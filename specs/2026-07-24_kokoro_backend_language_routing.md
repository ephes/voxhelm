# Kokoro TTS Backend + Automatic Language Routing

Status: DEPLOYED — implemented and live-verified on the studio 2026-07-24.
Active branch: PRIMARY (de→kokoro-martin, en→kokoro-af_heart). One open
item: the human listening test (Slice 6 item 7).

## Verified results (2026-07-24, studio, via loopback HTTP + HA API)

- Wyoming describe: Piper voices + kokoro-af_heart/kokoro-martin advertised,
  ASR intact.
- /v1/audio/speech: DE→kokoro-martin @24kHz, EN→kokoro-af_heart @24kHz
  (X-Voxhelm headers asserted); German rules check (numbers/abbreviations,
  pinned martin + routing:false) 7.0s audio, no errors; long-text chunking:
  65.9s of audio synthesized in 6.4s wall (>510-token input, pinned martin).
- Routing override: pinned de_DE-thorsten-high + English text →
  kokoro-af_heart with effective language en. Floor: "Okay." with pinned
  German voice stays on the pinned Piper voice.
- HA E2E (DE pipeline, intent→tts): German question answered in German and
  synthesized; English question answered in English and synthesized; TTS
  proxy audio fetched for both.
- Latency (median of 5 warm, ~15-word sentence): martin 0.43s,
  af_heart 0.44s — both far under the 2.0s threshold; the int8 fallback
  stays unused.
- Rollback rehearsal: deploy with both flags false → only Piper voices
  advertised, kokoro env absent; re-enable deploy restored routing; full
  battery re-passed afterwards.
- Tier-2 on-studio note: the `requires_models` suite was not re-executed on
  the studio directly (the venv is root-owned and interactive root SSH is
  not provisioned); equivalence holds because the deployed artifacts are
  sha256-verified at deploy against the same checksums the local tier-2
  runs passed with, and the live battery exercised both deployed voices
  end-to-end, including chunking and voicepack selection.

## Context

voxhelm serves Wyoming STT/TTS for Home Assistant from the Mac Studio
(launchd, uv-managed venv, Python `>=3.14,<3.15`). TTS today is Piper only:
`de_DE-thorsten-high` and `en_US-lessac-medium`, with the voice pinned per HA
Assist pipeline. Since the Voice PE routes free-form queries to an LLM agent
that answers in the language of the question, English answers on the German
pipeline are spoken by the German Piper voice with a heavy accent.

Goals:

1. Add a Kokoro ONNX backend with two models: the official Kokoro v1.0
   (English, default voice `af_heart`) and the community German fine-tune
   "Martin" (`kikiri` lineage, ONNX export by Godelaune).
2. Automatic language detection on outgoing TTS text, routing to the
   language-appropriate voice regardless of the pipeline-pinned voice.
3. Keep Piper installed and working as fallback; rollback is config-only.

Non-goals: streaming synthesis (voxhelm advertises
`supports_synthesize_streaming=False` today and that stays), voice cloning,
languages beyond de/en, replacing Piper for anything else.

## Constraints discovered up front (bind the design)

- Python 3.14 pin: `kokoro-onnx` caps `<3.14` and `misaki` (official English
  G2P) caps `<3.13` — NEITHER is installable. `onnxruntime` 1.27.0 ships a
  cp314 macosx-arm64 wheel. Therefore: talk to onnxruntime directly and
  phonemize BOTH languages with espeak-ng (`phonemizer-fork` +
  `espeakng-loader`, both uncapped). This mirrors the Martin repo's own
  FastAPI implementation (Apache-2.0), which is the reference for inference
  and German handling.
- espeak-ng is NOT currently installed on the studio; the deploy role must
  provision it via Homebrew and the runtime must locate `libespeak-ng.dylib`
  (espeakng-loader, with an env override).
- Model artifacts (checksums pinned at implementation time):
  - Official: `kokoro-v1.0.onnx` (325.5 MB) + `voices-v1.0.bin` (28.2 MB)
    from the `thewh1teagle/kokoro-onnx` GitHub release `model-files-v1.0`.
    The int8 variant (`kokoro-v1.0.int8.onnx`, 92.4 MB) is provisioned and
    checksum-pinned alongside it in Slice 4 so the latency fallback (Slice
    6 item 5) is a config-only switch for the OFFICIAL model. No int8
    Martin exists; Martin's latency/quality fallback is remapping
    de→`de_DE-thorsten-high` in `VOXHELM_TTS_LANGUAGE_VOICES` (one env
    var).
  - Martin: `kokoro-martin.onnx` + `voices-martin.npz` from HF
    `Godelaune/Kokoro-82M-ONNX-German-Martin`. (`german_text_rules.py` is
    NOT downloaded at deploy time — the vendored copy inside the voxhelm
    codebase (Slice 2) is the single authoritative runtime source, so
    deployed preprocessing can never drift from tested code.)
- `uv sync --frozen` in deploy means every dependency change must update
  `uv.lock` in the same slice.
- voxhelm deploys by rsync from the local working tree (not git), so deploys
  do not depend on pushes; slices are still committed after their Pi gate.

## Architecture

One new backend, one routing layer, minimal surgery elsewhere:

- **Voice registry across backends.** Voice discovery becomes
  backend-agnostic: each backend contributes `InstalledVoice`-style records
  (generalized: key, languages, backend name, backend-specific artifact
  paths). The requested/routed voice determines the backend — so
  `kokoro-martin`, `kokoro-af_heart`, and the Piper voices coexist, and
  `VOXHELM_TTS_BACKEND` degrades to "default backend for unprefixed/unknown
  cases" instead of a global switch. Wyoming `describe`
  (`transcriptions/wyoming.py:build_wyoming_info`) enumerates the registry
  instead of the Piper voice dir, with per-backend attribution.
- **KokoroBackend** (`synthesis/` module) implementing the existing
  `BackendProtocol`: espeak-ng phonemization (language `de` for Martin, `en-us`
  for af_heart) → token ids (vocab from the Martin reference implementation /
  kokoro tokenizer config) → `onnxruntime.InferenceSession.run` → 24 kHz WAV
  written the same way PiperBackend does (sample rate is read back from the
  WAV header downstream, so nothing else changes). Sessions cached per model
  file, synthesis under a per-model lock (mirrors `_PIPER_LOCK`), lazy
  imports with `BackendUnavailableError` when the `kokoro` extra is absent.
  **Voicepack key resolution (implementation-time fact-check, tested):**
  `voices-martin.npz` is npz-keyed; whether `voices-v1.0.bin` is npz-keyed
  (`np.load` name access) or a raw array must be established from the file
  itself during Slice 2. If keyed: select by name. If raw: vendor the
  official voice-name→index table from the kokoro-onnx project and select
  by pinned index. Two test tiers keep this honest without checking in
  large artifacts: (1) a unit test with a small CHECKED-IN synthetic pack
  fixture pins the selection LOGIC (keyed lookup and/or index-table path,
  shape validation, clamp behavior) and always runs in the Slice 2 gate;
  (2) an artifact-backed test marked `requires_models` pins the REAL
  `af_heart`/`martin` embedding checksums — it runs during Slice 2
  development against locally downloaded models (skipped when artifacts
  are absent, e.g. in the bare review environment) and again on the studio
  as part of Slice 6, so a silent wrong-voice regression is impossible on
  the deployed artifacts.
  **Style-vector indexing** follows the Martin reference implementation
  exactly: the style row is selected by the chunk's token count BEFORE
  boundary/padding tokens are added, and an index clamp to the pack's
  length guards the edge; a boundary test pins the indexing at the exact
  chunk limit (no off-by-two, no out-of-range).
  **Token limit & chunking (hard requirement):** Kokoro models accept ~510
  phoneme tokens per call, and the voicepack is length-indexed — the style
  vector is selected by the chunk's token count from the chosen voicepack
  entry. Free-form LLM answers routinely exceed one chunk, so the backend
  MUST: split input on sentence boundaries into chunks under the token
  budget (hard-splitting an over-long single sentence as last resort),
  select the per-chunk style vector from the configured `voicepack_key`
  entry by that chunk's token length, run inference per chunk, and
  concatenate the PCM into one WAV. Unit tests cover the boundaries: text
  just under the limit (single call), a two-sentence text crossing the
  limit (two calls, concatenated duration ≈ sum), and a single over-long
  sentence (hard split, no crash).
- **German preprocessing:** vendor `german_text_rules.py` (Apache-2.0,
  attribution comment with upstream URL + revision) and apply it before
  phonemization for German synthesis only. Vendored file is excluded from
  strict lint if needed but covered by tests.
- **Language routing** in `synthesis/service.py::synthesize_text` (so
  Wyoming, HTTP `/v1/audio/speech`, and batch jobs all benefit):
  - `lingua-language-detector` restricted to GERMAN/ENGLISH, lazy singleton.
  - New settings: `VOXHELM_TTS_LANGUAGE_ROUTING` (bool, default false) and
    `VOXHELM_TTS_LANGUAGE_VOICES` (language→voice map across all backends;
    the existing Piper-only `VOXHELM_PIPER_LANGUAGE_VOICES` stays for
    Piper-internal fallback resolution, unchanged).
  - Precedence WHEN ROUTING IS ON: detected language's mapped voice
    overrides the request-pinned voice, and the EFFECTIVE LANGUAGE becomes
    the detected language end-to-end — the `SynthesizeParams.language`
    passed to the backend, the `SynthesisResult.language`, and the
    `X-Voxhelm-Language` header all carry the detected language, never the
    request's original one (asserted by a routing test). WHEN OFF:
    behavior is byte-for-byte today's.
  - Deterministic routing floor (exact criteria, each with a unit test):
    routing applies only when ALL hold — (a) the stripped text is ≥ 20
    characters, (b) lingua's confidence for the top language is ≥ 0.90
    (`compute_language_confidence_values` on the de/en-restricted
    detector), (c) the detected language has a mapping in
    `VOXHELM_TTS_LANGUAGE_VOICES`, and (d) the mapped voice exists in the
    voice registry. Any condition failing → the request's pinned
    voice/language is used unchanged; (d) failing additionally logs a
    warning naming the missing voice. Routing NEVER raises at REQUEST
    time for routing-related reasons; the one exemption is STARTUP
    configuration validation — routing enabled without the `routing`
    extra installed fails fast at service startup (Slice 3), which is a
    deployment error, not a per-request condition.
- **Config surface (env, all optional):** `VOXHELM_KOKORO_MODEL_DIR`,
  `VOXHELM_KOKORO_MODELS` (voice-key→entry; each entry names model file,
  voicepack file, **voicepack_key** — the embedding to select inside the
  pack, e.g. `af_heart` in `voices-v1.0.bin`, `martin` in
  `voices-martin.npz` (exact key confirmed against the npz at
  implementation time) — and phoneme language; parsed from a compact env
  encoding consistent with existing `env_map` style),
  `VOXHELM_TTS_LANGUAGE_ROUTING`, `VOXHELM_TTS_LANGUAGE_VOICES`,
  `VOXHELM_ESPEAK_LIBRARY` (optional dylib override). Registry voice keys
  follow `kokoro-<voicepack_key>` by convention, but the explicit
  `voicepack_key` in the entry is authoritative — never parsed back out of
  the registry key.
- **Verification surface:** `/v1/audio/speech` responses gain
  `X-Voxhelm-Backend`, `X-Voxhelm-Voice`, and `X-Voxhelm-Language` headers
  populated from `SynthesisResult` (added in Slice 3; also serves ongoing
  debugging). The headers are UNCONDITIONAL new response surface —
  present regardless of the routing flag; the "routing off =
  byte-for-byte today's behavior" guarantee applies to synthesis
  parameters and audio, not to these additive headers. Slice 6's metadata
  assertions read them. When routing applies, it must also release an
  explicitly pinned `request_model` (reset to auto) so registry dispatch
  follows the mapped voice's backend — a language-routed voice can never
  be held on the wrong backend by a pinned model name.

## Slices

Each slice: implement → Pi review (`openai-codex/gpt-5.6-sol` via
pi-review-loop, fresh review after each fix round) until CLEAN → commit in
the owning repo (message prefixed `# `, no self-reference). Docs and
CHANGELOG updates belong to the slice that changes behavior, not a cleanup
slice at the end.

### Slice 1 — voxhelm: registry refactor (no new features)

Generalize voice discovery + backend selection: voice registry, per-voice
backend dispatch in `build_backend_service`/`resolve_voice` call sites,
Wyoming `describe` driven by the registry, `SynthesisResult.backend_name`
already flows. Piper behavior must be provably unchanged: existing tests
pass untouched except where they assert internals that moved; add registry
unit tests. No new deps.

### Slice 2 — voxhelm: KokoroBackend

Backend + vendored German rules + tokenizer/vocab + settings + `kokoro`
extra in `pyproject.toml` (onnxruntime, phonemizer-fork, espeakng-loader,
numpy if not present) + `uv.lock` update. (`lingua-language-detector` is
NOT part of this extra — routing is backend-agnostic and gets its own
`routing` extra in Slice 3, so language routing works with Piper-only
voices too.) Unit
tests: phonemize/tokenize with espeak mocked or skipped-if-absent, inference
via a stubbed ort session (fixture pattern like the existing fake-Piper
tests), German rules applied only for German. A dev smoke script
(`manage.py`-level or test marked `slow`) that runs real inference when the
models are present locally — used again in Slice 6 on the studio.

### Slice 3 — voxhelm: language routing

Detector, settings, precedence-with-floor logic, wiring into
`synthesize_text`; a `routing` extra in `pyproject.toml` carrying
`lingua-language-detector` (+ `uv.lock` update) — routing raises a clear
configuration error at startup if enabled without the extra installed, and
is inert otherwise. The `/v1/audio/speech` `X-Voxhelm-*` response headers
land here, plus an optional request field `routing` (boolean, default true)
on that endpoint: `routing: false` bypasses language routing for that
single request, so direct-backend checks and debugging can pin a voice
deterministically regardless of the active language map (the Wyoming path
has no bypass — HA always routes). Slice 6's explicit-voice checks (the
vendored-rules check and the >510-token long-text check) send
`routing: false` with `voice=kokoro-martin`, which reaches Martin in every
branch. Tests: routing off = identical params through to backend
(regression); routing on maps DE/EN texts to mapped voices; short/ambiguous
text falls back to pinned voice; detector restricted to two languages.
Wyoming handler untouched (service-layer feature) — confirmed by the
existing monkeypatch tests still passing.

### Slice 4 — ops-library: voxhelm_deploy role

- Homebrew `espeak-ng` (role targets macOS; follow existing brew usage in
  the repo's macOS roles, or `community.general.homebrew`).
- Model provisioning tasks mirroring the Piper download block: GitHub
  release assets + HF files with sha256 pins into
  `voxhelm_kokoro_model_dir`; idempotent (stat + checksum, no re-download).
- `kokoro` and `routing` uv extras wired into `uv_sync_args.yml` behind
  `voxhelm_tts_kokoro_enabled` / `voxhelm_tts_language_routing_enabled`
  (mirroring the diarization pattern) — independently toggleable.
- env template: the new VOXHELM_* vars; `validate.yml` assertions (models
  configured when kokoro enabled, language map sanity).
- README + CHANGELOG.

### Slice 5 — ops-control: wiring

`playbooks/deploy-voxhelm.yml`: enable kokoro, model entries
(`kokoro-martin` de / `kokoro-af_heart` en), `VOXHELM_TTS_LANGUAGE_ROUTING`
on, language map de→`kokoro-martin`, en→`kokoro-af_heart`. Piper vars stay
untouched (fallback + rollback). CHANGELOG; update
`docs/VOICE_ASSIST_CLANKY_BRIDGE.md` TTS section + `docs/HOMEASSISTANT.md`
voice section (they currently document the pinned-voice accent limitation —
that text changes, so it belongs to this slice).

### Integration review (orchestrator, not Pi)

Cross-slice pass by the orchestrating session before deploy: registry ↔
backend ↔ routing ↔ role env vars ↔ playbook values name-consistency; no
leftover plan-only names; `uv.lock` consistent; both repos' diffs read as
one coherent change. Findings fixed + re-gated before deploy.

### Slice 6 — deploy + live verification (definition of done)

Deploy voxhelm (`ops-control`: install-local-library + deploy voxhelm),
then, all via API from the workstation unless noted:

1. Wyoming `describe` lists Piper voices AND `kokoro-martin` +
   `kokoro-af_heart`, ASR section intact.
2. `/v1/audio/speech`: assertions follow the ACTIVE configuration branch —
   **primary** (de→`kokoro-martin`): German sentence → valid WAV, 24 kHz,
   headers backend `kokoro`, voice `kokoro-martin`; **fallback** (Martin
   failed latency or listening, de→`de_DE-thorsten-high`): German sentence
   → backend `piper`, voice `de_DE-thorsten-high`, 22.05 kHz. In both
   branches: English sentence → `kokoro-af_heart`, 24 kHz. "Done" means the
   active branch's assertions pass and the active branch is recorded here.
   The vendored-rules check EXPLICITLY pins `voice=kokoro-martin` AND sends
   `routing: false` (the Slice 3 bypass), and the assertion requires the
   `kokoro-martin` response header — so neither routing precedence nor a
   Piper response can falsely satisfy it in any branch: a short German sentence with
   numbers/abbreviations must synthesize without error and with plausible
   duration. A LONG text (>510 phoneme tokens,
   e.g. a multi-sentence paragraph) runs against the real model the same
   way — explicit `voice=kokoro-martin` + `routing: false`, `kokoro-martin`
   asserted via the response header: response succeeds and audio duration
   scales with length — exercising chunking, per-chunk voicepack indexing,
   and concatenation live, not just under the stubbed session tests.
3. Routing override: request with pinned voice `de_DE-thorsten-high` but
   English text → audio comes back from `kokoro-af_heart` (headers), proving
   detected language beats the pin (this assertion is branch-independent —
   English maps to Kokoro in both branches). Routing floor: a one-word text
   ("Okay") with a pinned German voice stays on the pinned voice.
4. E2E through HA: Assist pipeline run (DE) with a free-form query → TTS
   stage completes against voxhelm; resulting media URL fetches audio.
   STT sanity: pipeline STT stage or `describe` ASR + a voxhelm health check
   (STT code path is untouched; assert service-level health).
5. Latency: measurement protocol — per model: one warm-up request
   (discarded; session/model load happens here; its cold wall-time is
   recorded separately for information), then 5 requests with a fixed
   ~15-word sentence; the reported figure is the MEDIAN warm wall-time of
   the `/v1/audio/speech` round-trip. Pass threshold for BOTH models:
   median ≤2.0s. Pass criteria per branch: primary branch passes when both
   models meet the threshold; if Martin misses it, ACTIVATING the
   documented fallback (remap de→`de_DE-thorsten-high`, branch recorded in
   this spec) itself constitutes passing item 5 — the official model must
   still meet the threshold (via int8 if needed) in every branch. Official model failing → switch to the
   pre-provisioned int8 artifact (config change) and re-measure against the
   same threshold. Martin failing → remap de→`de_DE-thorsten-high` in the
   language-voice map (int8 cannot help Martin) and record the project as
   English-routing-only for German audio.
6. Rollback rehearsal: one deploy with `voxhelm_tts_kokoro_enabled=false` +
   routing off → Piper-only behavior returns; then re-enable. (Config-only,
   mirrors the Clanky-bridge rollback proof.)
7. Human listening test (Jochen, async): German answer quality vs Thorsten,
   English accent fix. Explicitly allowed to fail the project: if Martin
   loses by ear, flip `VOXHELM_TTS_LANGUAGE_VOICES` de→`de_DE-thorsten-high`
   (routing still fixes English accent) — one env var, no redeploy of code.

The task is DONE only when 1–6 pass live; 7 is the user's acceptance check
with a documented one-variable fallback.

## Risks

- **espeak-ng English G2P < misaki quality** (numbers, homographs): accepted;
  vendoring misaki later is a bounded follow-up. German is espeak-native for
  this model — that's what Martin was trained against.
- **Martin compound-word mispronunciations**: mitigated by vendored rules;
  residual risk accepted, fallback is the one-variable voice map change.
- **Community model provenance**: ONNX files from HF run under onnxruntime
  (no pickle execution), checksum-pinned at first download; Apache-2.0.
- **Tokenizer/vocab mismatch** between official Kokoro and Martin exports:
  the Martin reference implementation is authoritative for Martin; unit
  tests pin a few known text→token sequences per model.
- **launchd/env drift**: new env vars only exist after the role template
  renders; the backend must degrade with a clear `BackendUnavailableError`
  (and routing must no-op) when unconfigured, so a voxhelm deploy without
  the ops-control wiring cannot break existing Piper TTS.
- **Detection latency**: lingua two-language detector is ~ms on sentence
  input and loaded lazily; measured as part of Slice 6 item 5.

## Rollback

Config-only at every level, but ORDER MATTERS because enabled routing
overrides pinned voices and registry dispatch sends known Kokoro voice keys
to the Kokoro backend regardless of `VOXHELM_TTS_BACKEND`:

1. To stop AUTOMATIC Kokoro use: set `VOXHELM_TTS_LANGUAGE_ROUTING` off
   (or remap both languages to Piper voices) — pinned pipeline voices are
   Piper voices, so the HA path is restored by this alone. Explicit
   requests naming a `kokoro-*` voice (HTTP callers) still dispatch to
   Kokoro while models remain configured; that is intentional.
2. To remove Kokoro voices entirely: `voxhelm_tts_kokoro_enabled: false`
   in the role. The enable flag gates model REGISTRATION, not just the uv
   extra — with it false, the env template renders no
   `VOXHELM_KOKORO_MODELS`, so kokoro voices are neither advertised in
   `describe` nor dispatchable, and can never be advertised-but-broken.
   (Slice 4 requirement: extra install and model config render from the
   same flag; `validate.yml` asserts they cannot diverge.)
3. `VOXHELM_TTS_BACKEND` or HA pipeline-voice changes are NOT sufficient on
   their own while routing is enabled — never document them as the first
   rollback step.

Piper stays installed throughout. Code rollback = revert the voxhelm
commits (each slice is one commit).

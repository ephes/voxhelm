import pytest


@pytest.fixture(autouse=True)
def configure_tokens(settings):
    settings.VOXHELM_BEARER_TOKENS = {"archive": "test-token"}


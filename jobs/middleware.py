"""Response privacy at the private job resource boundary, including errors."""

from django.http import HttpRequest, HttpResponse
from django.utils.cache import add_never_cache_headers, patch_vary_headers
from django.utils.deprecation import MiddlewareMixin


class PrivateJobResponsesMiddleware(MiddlewareMixin):
    def process_response(self, request: HttpRequest, response: HttpResponse) -> HttpResponse:
        match = request.resolver_match
        name = match.url_name if match else None
        if name in {"uploads-collection", "jobs-collection", "job-detail", "job-artifact"}:
            add_never_cache_headers(response)
            patch_vary_headers(response, ("Authorization",))
        elif name in {"root", "operator-artifact"}:
            add_never_cache_headers(response)
            patch_vary_headers(response, ("Cookie",))
        return response

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

django_application = get_asgi_application()

# Imported after Django setup: the wrapper reads settings per request.
from config.body_limit import RequestBodyLimitMiddleware  # noqa: E402

application = RequestBodyLimitMiddleware(django_application)

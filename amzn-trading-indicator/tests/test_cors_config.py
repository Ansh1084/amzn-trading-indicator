import re

from backend.app.main import app


def test_local_vite_fallback_port_5174_is_allowed_by_cors():
    middleware = next(item for item in app.user_middleware if item.cls.__name__ == "CORSMiddleware")
    allowed = middleware.kwargs["allow_origin_regex"]
    assert re.fullmatch(allowed, "http://localhost:5173")
    assert re.fullmatch(allowed, "http://127.0.0.1:5174")
    assert not re.fullmatch(allowed, "http://localhost:5180")

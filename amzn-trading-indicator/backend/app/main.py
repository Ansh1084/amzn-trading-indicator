from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes import router

app = FastAPI(title="AMZN Trading Indicator API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1):517[0-9]$",
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(router, prefix="/api")
@app.get("/health")
def health():
    return {"status": "ok"}

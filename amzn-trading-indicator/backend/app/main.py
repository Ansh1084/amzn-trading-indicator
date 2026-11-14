import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes import router
from backend.app.services.market_stream import service as market_stream

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    configured = market_stream.start()
    reconcile_task = asyncio.create_task(_reconcile_loop()) if configured else None
    try:
        yield
    finally:
        if reconcile_task:
            reconcile_task.cancel()
            try:
                await reconcile_task
            except asyncio.CancelledError:
                pass
        market_stream.stop()

async def _reconcile_loop():
    # Recover missed/completed bars regularly; the stream provides minute updates only.
    while True:
        try:
            await market_stream.reconcile()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Keep periodic recovery alive even if an unexpected exception escapes
            # the service's own error boundary.
            logger.exception("Unexpected error in the AMZN market reconciliation loop")
        await asyncio.sleep(60)

app = FastAPI(title="AMZN Trading Indicator API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1):517[0-9]$",
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(router, prefix="/api")
@app.get("/health")
def health():
    return {"status": "ok"}

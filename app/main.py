from __future__ import annotations

from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .api import router


app = FastAPI(title="YOLO Label Studio Local", version="0.1.0")
app.include_router(router)
_frontend = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=_frontend), name="static")


@app.get("/")
def index():
    return FileResponse(_frontend / "index.html")


def run():
    uvicorn.run("app.main:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    run()

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag_agent import __version__
from rag_agent.api.schemas import HealthResponse, IngestResponse, QueryRequest
from rag_agent.container import Container
from rag_agent.domain.models import AgentAnswer, AgentEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def create_app(container: Container | None = None) -> FastAPI:
    container = container or Container()
    settings = container.settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(container.ensure_index)
        yield

    app = FastAPI(title="Agentic RAG Research Assistant", version=__version__, lifespan=lifespan)
    app.state.container = container
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def get_container(request: Request) -> Container:
        return request.app.state.container

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and not (x_api_key and secrets.compare_digest(x_api_key, settings.api_key)):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    @app.get("/health", response_model=HealthResponse)
    def health(c: Container = Depends(get_container)):
        return HealthResponse(
            status="ok",
            indexed_chunks=c.vector_store.count(),
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            retrieval_strategy=settings.retrieval_strategy,
        )

    @app.post("/query", response_model=AgentAnswer, dependencies=[Depends(require_api_key)])
    def query(req: QueryRequest, c: Container = Depends(get_container)):
        return c.agent(req.strategy).ask(req.question, req.history)

    @app.post("/query/stream", dependencies=[Depends(require_api_key)])
    def query_stream(req: QueryRequest, c: Container = Depends(get_container)):
        """Server-Sent Events: `route` -> many `token` -> `final` (or `error`)."""
        agent = c.agent(req.strategy)

        def events():
            for ev in agent.stream(req.question, req.history):
                yield f"data: {ev.model_dump_json()}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
    def ingest(files: list[UploadFile] = File(...), c: Container = Depends(get_container)):
        upload_dir = Path(settings.data_dir) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for f in files:
            name = Path(f.filename or "").name
            if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                raise HTTPException(415, f"Unsupported file type: {name!r}")
            data = f.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, f"{name!r} exceeds {settings.max_upload_mb} MB")
            target = upload_dir / name
            target.write_bytes(data)
            saved.append(target)
        report = c.pipeline.ingest_paths(saved)
        return IngestResponse(
            files=report.files, chunks_added=report.chunks_added, skipped=report.skipped,
            total_chunks=c.vector_store.count(),
        )

    return app


app = create_app()

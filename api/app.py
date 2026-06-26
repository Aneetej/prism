from __future__ import annotations

import asyncio

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router


def create_app(config_path: str = "config.yaml") -> FastAPI:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    api_cfg = cfg.get("api", {})
    cors_origins = api_cfg.get("cors_origins", ["*"])

    app = FastAPI(
        title="Inference-Time Safety Layer",
        description=(
            "LLM-agnostic two-stage safety layer. "
            "Stage 1 screens prompts before inference; "
            "Stage 2 verifies outputs before sending to user."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.on_event("startup")
    async def _startup():
        import logging
        log = logging.getLogger(__name__)
        try:
            from pipeline import from_config
            pipeline = from_config(config_path)
            app.state.pipeline = pipeline
            app.state.config = cfg

            #Eager load
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, pipeline.llm._load)
            log.info("LLM loaded at startup")

            if pipeline.pre_check is not None and pipeline.pre_check.mode in ("hybrid", "classifier"):
                await loop.run_in_executor(None, pipeline.pre_check._load_classifier)
                log.info("Stage 1 classifier loaded at startup")

            #Eagerly load Llama Guard if it's active (standalone or inside CascadeChecker)
            from checker.cascade import CascadeChecker
            from checker.llama_guard import LlamaGuardChecker
            if isinstance(pipeline.checker, CascadeChecker):
                if isinstance(pipeline.checker.slow, LlamaGuardChecker):
                    await loop.run_in_executor(None, pipeline.checker.slow._load)
                    log.info("LlamaGuardChecker loaded at startup")
            elif isinstance(pipeline.checker, LlamaGuardChecker):
                await loop.run_in_executor(None, pipeline.checker._load)
                log.info("LlamaGuardChecker loaded at startup")

        except Exception as exc:
            log.error("Pipeline init failed: %s", exc)
            raise

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    api_cfg = cfg.get("api", {})
    uvicorn.run(
        "api.app:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=False,
    )
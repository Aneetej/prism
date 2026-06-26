#Route handlers
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas import (
    ConfigResponse,
    HealthResponse,
    RunRequest,
    RunResponse,
)

router = APIRouter()


def _get_pipeline(request: Request):
    return request.app.state.pipeline


def _get_config(request: Request):
    return request.app.state.config


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Return model id, checker type, and pipeline mode."""
    pipeline = _get_pipeline(request)
    return HealthResponse(
        status="ok",
        model_id=pipeline.llm.model_id,
        checker=type(pipeline.checker).__name__,
        mode=pipeline.mode,
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config(request: Request):
    cfg = _get_config(request)
    llm_cfg = cfg.get("llm", {})
    chk_cfg = cfg.get("safety_checker", {})
    pre_cfg = cfg.get("pre_check", {})
    exp_cfg = cfg.get("experiment", {})
    return ConfigResponse(
        llm_provider=llm_cfg.get("provider", "unknown"),
        model_id=llm_cfg.get("model_id", "unknown"),
        checker_type=chk_cfg.get("type", "rule_based"),
        pre_check_enabled=pre_cfg.get("enabled", True),
        default_mode=exp_cfg.get("mode", "full_output"),
        buffer_size=exp_cfg.get("buffer_size", 30),
        overlap=exp_cfg.get("overlap", 5),
    )


#Could think about running in chunks instead of waiting for full model output. latency increase if step 2 is a lanuage model with many params  
@router.post("/run", response_model=RunResponse)
async def run(request: Request, body: RunRequest):
    pipeline = _get_pipeline(request)

    from llm.base import GenerationConfig
    gen_config = GenerationConfig(
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        system_prompt=pipeline.gen_config.system_prompt,
    )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, pipeline.run, body.prompt, gen_config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return RunResponse(
        output=result.output,
        passed=result.passed,
        blocked_at=result.blocked_at,
        blocked_category=result.blocked_category,
        latency_ms=result.latency_ms,
        checker_latency_ms=result.checker_latency_ms,
        pre_check_latency_ms=result.pre_check_latency_ms,
        llm_latency_ms=result.llm_latency_ms,
        mode=result.mode,
        model_id=pipeline.llm.model_id,
    )


@router.post("/stream")
async def stream(request: Request, body: RunRequest):
    """Run the pipeline in sliding-window mode; returns an SSE stream of verified chunks."""
    pipeline = _get_pipeline(request)

    async def event_generator() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _produce():
            try:
                for chunk in pipeline.stream(body.prompt):
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, f"error:{exc}")
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        _future = loop.run_in_executor(None, _produce)  # noqa: F841 — keep reference alive

        while (chunk := await q.get()) is not None:
            if isinstance(chunk, str) and chunk.startswith("error:"):
                yield f"event: error\ndata: {chunk[6:]}\n\n"
                break
            event = "blocked" if chunk == pipeline.error_message else "data"
            yield f"event: {event}\ndata: {chunk}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
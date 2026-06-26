"""Pydantic request/response models for the FastAPI layer."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192)
    mode: Literal["full_output", "sliding_window"] = "full_output"
    checker: Optional[Literal["rule_based", "classifier", "llm_judge", "probe"]] = None
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class RunResponse(BaseModel):
    output: str
    passed: bool
    blocked_at: Optional[Literal["pre_check", "safety_check"]]
    blocked_category: Optional[str] = None
    latency_ms: float
    checker_latency_ms: float
    pre_check_latency_ms: float
    llm_latency_ms: float
    mode: str
    model_id: str


class HealthResponse(BaseModel):
    status: str
    model_id: str
    checker: str
    mode: str


class ConfigResponse(BaseModel):
    llm_provider: str
    model_id: str
    checker_type: str
    pre_check_enabled: bool
    default_mode: str
    buffer_size: int
    overlap: int


class ErrorResponse(BaseModel):
    detail: str
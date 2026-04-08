from __future__ import annotations

import traceback
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from support_triage_env.models import TriageAction
from support_triage_env.server.environment import SupportTriageEnvironment

app = FastAPI(title="SupportTriageEnv", version="0.1.0")
env = SupportTriageEnvironment()


class ResetRequest(BaseModel):
    task_id: int = Field(default=1)
    seed: int = Field(default=42)


@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "support-triage-env"}


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "support-triage-env"}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    try:
        req = request if request is not None else ResetRequest()
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        return obs.model_dump(mode="json")
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/step")
def step(action: TriageAction) -> Dict[str, Any]:
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(mode="json"),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/state")
def state() -> Dict[str, Any]:
    return env.state().model_dump(mode="json")

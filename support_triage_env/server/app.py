from __future__ import annotations

from typing import Any, Dict, Tuple

from fastapi import FastAPI
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
def reset(request: ResetRequest = None) -> Dict[str, Any]:
    req = request or ResetRequest()
    obs = env.reset(task_id=req.task_id, seed=req.seed)
    return obs.model_dump(mode="json")


@app.post("/step")
def step(action: TriageAction) -> Dict[str, Any]:
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(mode="json"),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    return env.state().model_dump(mode="json")

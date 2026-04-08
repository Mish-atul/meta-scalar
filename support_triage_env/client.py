from __future__ import annotations

from typing import Any, Dict, Tuple

import httpx

from .models import TriageAction, TriageObservation, TriageState


class TriageEnv:
    """Thin HTTP client wrapper for the triage environment API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def reset(self, task_id: int = 1, seed: int = 42) -> TriageObservation:
        try:
            response = self.client.post("/reset", json={"task_id": task_id, "seed": seed})
            response.raise_for_status()
            return TriageObservation.model_validate(response.json())
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500] if exc.response else "no body"
            raise RuntimeError(
                f"reset() failed with HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"reset() could not connect to {self.base_url}: {exc}"
            ) from exc

    def step(self, action: TriageAction) -> Tuple[TriageObservation, float, bool, Dict[str, Any]]:
        try:
            response = self.client.post("/step", json=action.model_dump(mode="json"))
            response.raise_for_status()
            payload = response.json()
            obs = TriageObservation.model_validate(payload["observation"])
            return obs, float(payload["reward"]), bool(payload["done"]), dict(payload.get("info", {}))
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500] if exc.response else "no body"
            raise RuntimeError(
                f"step() failed with HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"step() could not connect to {self.base_url}: {exc}"
            ) from exc

    def state(self) -> TriageState:
        response = self.client.get("/state")
        response.raise_for_status()
        return TriageState.model_validate(response.json())

    def close(self) -> None:
        self.client.close()

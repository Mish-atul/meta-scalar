from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

IMAGE_NAME = "support-triage-env"
CONTAINER_NAME = "support-triage-env-ci"
HOST_PORT = 8010
CONTAINER_PORT = 8000
DOCKERFILE_PATH = "support_triage_env/server/Dockerfile"
HEALTH_URL = f"http://127.0.0.1:{HOST_PORT}/health"
RESET_URL = f"http://127.0.0.1:{HOST_PORT}/reset"
STEP_URL = f"http://127.0.0.1:{HOST_PORT}/step"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

    if stdout:
        print(stdout.strip())
    if stderr:
        print(stderr.strip())

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return subprocess.CompletedProcess(args=cmd, returncode=result.returncode, stdout=stdout, stderr=stderr)


def locate_openenv() -> list[str]:
    exe = shutil.which("openenv")
    if exe:
        return [exe]

    appdata = os.environ.get("APPDATA")
    if appdata:
        candidate = os.path.join(appdata, "Python", "Python312", "Scripts", "openenv.exe")
        if os.path.exists(candidate):
            return [candidate]

    raise RuntimeError("OpenEnv CLI not found. Install openenv-core and ensure openenv is on PATH.")


def wait_for_health(timeout_seconds: int = 40) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=3) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    if payload.get("status") == "ok":
                        print(f"Health OK: {payload}")
                        return
        except Exception:
            time.sleep(1)
            continue
    raise RuntimeError("Container health check timed out")


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"Request to {url} returned status {response.status}")
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Request failed for {url}: {exc.code} {body}") from exc


def ensure_docker_available() -> None:
    run(["docker", "version"], check=True)


def cleanup_container() -> None:
    run(["docker", "rm", "-f", CONTAINER_NAME], check=False)


def run_pipeline() -> None:
    python = sys.executable

    print("\n==== Gate 1/5: Tests + Coverage ====")
    run([python, "-m", "pytest", "--cov=support_triage_env", "--cov-report=term", "-q"])

    print("\n==== Gate 2/5: Docker Build ====")
    ensure_docker_available()
    run(["docker", "build", "-t", IMAGE_NAME, "-f", DOCKERFILE_PATH, "."])

    print("\n==== Gate 3/5: Container Smoke Tests ====")
    cleanup_container()
    run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"{HOST_PORT}:{CONTAINER_PORT}",
            IMAGE_NAME,
        ]
    )

    try:
        wait_for_health()
        reset_payload = post_json(RESET_URL, {"task_id": 1, "seed": 42})
        current = reset_payload.get("current_email") or {}
        email_id = current.get("email_id")
        if not email_id:
            raise RuntimeError("Reset did not return current_email.email_id")

        step_payload = post_json(
            STEP_URL,
            {
                "action_type": "request_info",
                "email_id": email_id,
                "question": "Please share invoice details.",
            },
        )

        if "reward" not in step_payload or "done" not in step_payload:
            raise RuntimeError("Step response missing reward/done fields")

        print(
            "Smoke test OK: "
            f"task={reset_payload.get('task_id')} "
            f"pending={reset_payload.get('pending_count')} "
            f"step_reward={step_payload.get('reward')}"
        )
    finally:
        cleanup_container()

    print("\n==== Gate 4/5: OpenEnv Validate ====")
    openenv_cmd = locate_openenv()
    run(openenv_cmd + ["validate"])

    print("\n==== Gate 5/5: Completed ====")
    print("All validation gates passed.")


def main() -> int:
    try:
        run_pipeline()
        return 0
    except Exception as exc:
        print(f"\nValidation pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

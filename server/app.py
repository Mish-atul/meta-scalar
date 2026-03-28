from support_triage_env.server.app import app

__all__ = ["app"]


def main() -> None:
	import uvicorn

	uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
	main()

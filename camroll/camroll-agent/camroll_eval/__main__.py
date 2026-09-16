"""Independent evaluator CLI; importing this module never starts a run."""
from __future__ import annotations

import argparse
import json

from .runner import Runner, configured_paths


def main(argv=None):
    parser = argparse.ArgumentParser(prog="camroll-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8772)
    run = commands.add_parser("run")
    run.add_argument("--run-id")
    run.add_argument("--config", choices=["M1", "M2", "M3", "M4"], default="M1")
    run.add_argument("--resume", action="store_true")
    history = commands.add_parser("import-history")
    history.add_argument("directory")
    replay = commands.add_parser("replay-history")
    replay.add_argument("directory")
    args = parser.parse_args(argv)
    root, source, albums = configured_paths()
    if args.command == "serve":
        from .service import create_app
        import uvicorn
        # Avoid access logs containing arbitrary request bodies or credentials.
        uvicorn.run(create_app(root, source, albums), host=args.host, port=args.port, access_log=False)
    elif args.command == "import-history":
        from .repository import Repository
        run_id = Repository(root).import_history(args.directory)
        print(json.dumps({"imported_run_id": run_id}))
    elif args.command == "replay-history":
        from pathlib import Path
        from .common import read_json
        from .metrics import summary
        state = read_json(Path(args.directory) / "run.json")
        state["system_id"] = "sentrix"
        print(json.dumps(summary(state), ensure_ascii=False, indent=2))
    elif args.command == "run":
        if args.resume and not args.run_id:
            parser.error("--resume requires --run-id")
        result = Runner(root, source, albums).run(run_id=args.run_id, config=args.config, resume=args.resume)
        print(json.dumps({"run_id": result["run_id"], "status": result["status"], "summary": result["summary"]},
                         ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

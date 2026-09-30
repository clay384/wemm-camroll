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
    run.add_argument('--experiment', choices=['Q37', 'Q38', 'W37', 'W38Max'])
    history = commands.add_parser("import-history")
    history.add_argument("directory")
    replay = commands.add_parser("replay-history")
    replay.add_argument("directory")
    repair = commands.add_parser("repair-qa")
    repair.add_argument("parent_id")
    repair.add_argument("--run-id", required=True)
    repair.add_argument("--qa-ids", nargs="+", required=True)
    caption = commands.add_parser("repair-caption")
    caption.add_argument("parent_id")
    caption.add_argument("--repair-id", required=True)
    caption.add_argument("--resume-preflight", action="store_true")
    args = parser.parse_args(argv)
    root, source, albums = configured_paths()
    if args.command == "repair-qa":
        from .repair import RepairRunner
        result = RepairRunner(root, source, albums).run_repair(args.parent_id, args.qa_ids, run_id=args.run_id)
        print(json.dumps({"run_id": result["run_id"], "status": result["status"], "repaired_valid_count": result["repaired_valid_count"]}))
        return 0
    elif args.command == "repair-caption":
        from .repair import repair_caption
        result = repair_caption(root, albums, args.parent_id, repair_id=args.repair_id, resume_preflight=args.resume_preflight)
        print(json.dumps({k: result.get(k) for k in ("repair_id", "status", "derived_index_version")}))
        return 0
    elif args.command == "serve":
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
        options = {'experiment': args.experiment, 'qa_concurrency': 1, 'judge_concurrency': 4} if args.experiment else {}
        result = Runner(root, source, albums).run(run_id=args.run_id, config='M4' if args.experiment else args.config,
                                                 resume=args.resume, **options)
        print(json.dumps({"run_id": result["run_id"], "status": result["status"], "summary": result["summary"]},
                         ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

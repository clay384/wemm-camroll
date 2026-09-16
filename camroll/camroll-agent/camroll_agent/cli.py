"""CamRoll cloud generation plus local WeMM CLI."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict


def main(argv=None):
    parser=argparse.ArgumentParser(prog="camroll-agent")
    sub=parser.add_subparsers(dest="command",required=True)
    inspect=sub.add_parser("inspect");inspect.add_argument("spec")
    for name in ["build","run","enrich"]:
        p=sub.add_parser(name)
        p.add_argument("spec");p.add_argument("-o","--output-dir",required=True)
        p.add_argument("--max-images",type=int);p.add_argument("--resume",action="store_true",default=True)
        p.add_argument("--force",action="store_true");p.add_argument("--concurrency",type=int,default=4)
        p.add_argument("--vlm-backend",choices=["qwen"],default="qwen")
        p.add_argument("--vlm-model",choices=["qwen3.7-flash"],default="qwen3.7-flash")
        p.add_argument("--embedding-model",choices=["wemm"],default="wemm")
    p=sub.add_parser("index");p.add_argument("memory_dir")
    p.add_argument("--embedding-model",choices=["wemm"],default="wemm")
    p=sub.add_parser("embed-images");p.add_argument("spec");p.add_argument("-o","--output-dir",required=True)
    p.add_argument("--max-images",type=int);p.add_argument("--resume",action="store_true",default=True)
    p.add_argument("--force",action="store_true")
    p=sub.add_parser("ask");p.add_argument("question");p.add_argument("--memory",dest="memory_dir",required=True)
    p.add_argument("--config",choices=["M1","M2","M3","M4"],default="M1")
    p.add_argument("--json",action="store_true");p.add_argument("--no-stream",action="store_true")
    p.add_argument("--max-steps",type=int,default=25);p.add_argument("--max-view-image-calls",type=int,default=5)
    p.add_argument("--llm-backend",choices=["qwen"],default="qwen")
    p.add_argument("--vlm-backend",choices=["qwen"],default="qwen")
    p.add_argument("--llm-model",choices=["qwen3.7-flash"],default="qwen3.7-flash")
    p.add_argument("--vlm-model",choices=["qwen3.7-flash"],default="qwen3.7-flash")
    sub.add_parser("budget")
    args=parser.parse_args(argv)
    from .pipeline import build, enrich, encode_images, inspect, reindex, BuildBudgetExhausted, BuildProviderFailed
    from .qwen import QwenClient
    from .wemm import EmbeddingClient
    if args.command=="inspect":
        result=inspect(args.spec)
    elif args.command in {"build","run","enrich"}:
        try:
            if args.command=='enrich':
                result=enrich(args.spec,args.output_dir,QwenClient(),max_images=args.max_images,resume=not args.force,concurrency=args.concurrency)
            else:
                result=build(args.spec,args.output_dir,QwenClient(),EmbeddingClient(),max_images=args.max_images,resume=not args.force,concurrency=args.concurrency)
        except (BuildBudgetExhausted, BuildProviderFailed) as exc:
            print(json.dumps(exc.summary,ensure_ascii=False,indent=2));return 3
    elif args.command=="embed-images":
        try:
            result=encode_images(args.spec,args.output_dir,EmbeddingClient(),max_images=args.max_images,resume=not args.force)
        except BuildProviderFailed as exc:
            print(json.dumps(exc.summary,ensure_ascii=False,indent=2));return 3
    elif args.command=="index":
        try:
            result=reindex(args.memory_dir,EmbeddingClient())
        except BuildProviderFailed as exc:
            print(json.dumps(exc.summary,ensure_ascii=False,indent=2));return 3
    elif args.command=="budget":
        result=QwenClient().usage_summary()
    else:
        from .agent import Agent
        agent=Agent(args.memory_dir,config=args.config,max_steps=args.max_steps,max_view_image_calls=args.max_view_image_calls)
        if args.json or args.no_stream:
            answer=agent.ask(args.question)
            if not args.json:
                print(answer.final_text);return 0
            result=asdict(answer)
        else:
            for event,value in agent.ask_streaming(args.question):
                print(json.dumps({"event":event,"data":value},ensure_ascii=False,default=str),flush=True)
            return 0
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    return 0

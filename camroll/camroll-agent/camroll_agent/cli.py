"""CamRoll CLI with generation and retrieval bound to album model profiles."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict

GENERATION_MODELS = ["qwen3.7-flash", "qwen3.8-flash", "qwen3.8-max"]
EMBEDDING_MODELS = ["wemm", "qwen", "Qwen3-VL-Embedding-2B"]


def _validate_models(parser, args, profile):
    generation = profile['generation_model'] if profile else 'qwen3.7-flash'
    embedding_names = {profile['embedding_backend'], profile['embedding_model']} if profile else {'wemm'}
    for argument in ('llm_model', 'vlm_model'):
        selected = getattr(args, argument, None)
        if selected is not None and selected != generation:
            parser.error('--' + argument.replace('_', '-') + ' differs from the selected or saved model profile: ' + generation)
    if getattr(args, 'embedding_model', None) is not None and args.embedding_model not in embedding_names:
        parser.error('--embedding-model differs from the selected or saved retrieval profile')
    if profile and getattr(args, 'config', None) not in (None, profile['mode']):
        parser.error('--config differs from the saved experiment mode: ' + profile['mode'])
    if profile and getattr(args, 'concurrency', profile['caption_concurrency']) != profile['caption_concurrency']:
        parser.error('--concurrency differs from the experiment caption concurrency')


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
        p.add_argument("--vlm-model",choices=GENERATION_MODELS,help="Validate the model selected by --experiment or the saved album")
        p.add_argument("--embedding-model",choices=EMBEDDING_MODELS,help="Validate the retrieval model selected by --experiment or the saved album")
        p.add_argument('--experiment', choices=['Q37', 'Q38', 'W37', 'W38Max'])
    p=sub.add_parser("index");p.add_argument("memory_dir")
    p.add_argument("--embedding-model",choices=EMBEDDING_MODELS)
    p.add_argument('--experiment', choices=['Q37', 'Q38', 'W37', 'W38Max'])
    p=sub.add_parser("embed-images");p.add_argument("spec");p.add_argument("-o","--output-dir",required=True)
    p.add_argument("--max-images",type=int);p.add_argument("--resume",action="store_true",default=True)
    p.add_argument("--force",action="store_true")
    p.add_argument('--experiment', choices=['Q37', 'Q38', 'W37', 'W38Max'])
    p=sub.add_parser("ask");p.add_argument("question");p.add_argument("--memory",dest="memory_dir",required=True)
    p.add_argument("--config",choices=["M1","M2","M3","M4"],help="Defaults to the saved experiment mode, or M1 for legacy albums")
    p.add_argument("--json",action="store_true");p.add_argument("--no-stream",action="store_true")
    p.add_argument("--max-steps",type=int,default=25);p.add_argument("--max-view-image-calls",type=int,default=5)
    p.add_argument("--llm-backend",choices=["qwen"],default="qwen")
    p.add_argument("--vlm-backend",choices=["qwen"],default="qwen")
    p.add_argument("--llm-model",choices=GENERATION_MODELS,help="Validate the saved album generation model")
    p.add_argument("--vlm-model",choices=GENERATION_MODELS,help="Validate the saved album visual model")
    sub.add_parser("budget")
    args=parser.parse_args(argv)
    from .pipeline import build, enrich, encode_images, inspect, reindex, BuildBudgetExhausted, BuildProviderFailed
    from .qwen import QwenClient
    from .wemm import EmbeddingClient
    if args.command=="inspect":
        result=inspect(args.spec)
    elif args.command in {"build","run","enrich"}:
        from .runtime_clients import build_clients
        cloud, embedding, reranker, profile = build_clients(args.experiment, args.output_dir)
        _validate_models(parser, args, profile)
        try:
            if args.command=='enrich':
                result=enrich(args.spec,args.output_dir,cloud,max_images=args.max_images,resume=not args.force,concurrency=args.concurrency,experiment_config=profile)
            else:
                result=build(args.spec,args.output_dir,cloud,embedding,max_images=args.max_images,resume=not args.force,concurrency=args.concurrency,experiment_config=profile)
        except (BuildBudgetExhausted, BuildProviderFailed) as exc:
            print(json.dumps(exc.summary,ensure_ascii=False,indent=2));return 3
    elif args.command=="embed-images":
        from .experiment import get_experiment_profile
        from .retrieval import make_retrieval_clients
        from .library import Library
        from pathlib import Path
        profile = get_experiment_profile(args.experiment).as_dict() if args.experiment else None
        namespace = Path(args.output_dir).resolve().name
        if (Path(args.output_dir) / 'CURRENT').is_file():
            saved = Library(args.output_dir, experiment_config=profile)
            profile = saved.experiment.as_dict() if saved.experiment else profile
            namespace = saved.cache_namespace or namespace
        if profile:
            profile['cache_namespace'] = namespace
        embedding = make_retrieval_clients(profile)[0]
        try:
            result=encode_images(args.spec,args.output_dir,embedding,max_images=args.max_images,resume=not args.force,experiment_config=profile)
        except BuildProviderFailed as exc:
            print(json.dumps(exc.summary,ensure_ascii=False,indent=2));return 3
    elif args.command=="index":
        try:
            from .index import run
            result=run(args.memory_dir, experiment=args.experiment, embedding_model=args.embedding_model)
        except BuildProviderFailed as exc:
            print(json.dumps(exc.summary,ensure_ascii=False,indent=2));return 3
    elif args.command=="budget":
        result=QwenClient().usage_summary()
    else:
        from .agent import Agent
        from .library import Library
        from .runtime_clients import build_clients
        cloud, embedding, reranker, profile = build_clients(memory=args.memory_dir)
        _validate_models(parser, args, profile)
        mode = args.config or (profile['mode'] if profile else 'M1')
        library = Library(args.memory_dir, embedding, config=mode, reranker=reranker, experiment_config=profile)
        agent=Agent(args.memory_dir,llm=cloud,vlm=cloud,library=library,config=mode,
                    max_steps=args.max_steps,max_view_image_calls=args.max_view_image_calls,
                    deadline_s=600 if profile else 180)
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

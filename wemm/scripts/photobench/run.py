"""Reusable stage CLI. A new run ID creates a separate immutable experiment."""
import argparse,re,time
from pathlib import Path
import bench

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','verify','ingest','evaluate','export','selftest'])
    ap.add_argument('--run-id',default=bench.RUN_ID);ap.add_argument('--source',type=Path)
    args=ap.parse_args();assert re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id)
    bench.RUN_ID=args.run_id;bench.OUT=bench.ROOT/'reports/photobench'/args.run_id
    if args.source:bench.SOURCE=args.source.resolve()
    if args.stage=='ingest':
        s,b=bench.client()
        for _ in range(720):
            st=bench.api(s,b,'/api/status')
            if st['state']=='error':raise RuntimeError(st['error'])
            if st['state']=='ready' and not st['busy_asset'] and not st['counts'].get('pending',0):break
            time.sleep(5)
        else:raise RuntimeError('Existing worker did not become ready in one hour')
    if args.stage=='export':
        import report
        report.export()
    else:getattr(bench,args.stage)()

if __name__=='__main__':main()

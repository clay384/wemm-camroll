"""Non-model checks; these results are not GPU inference evidence."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.common import ROOT,relative_name
from backend.benchmark import metric
checks=[]
for p in ['../private','/Users/example','a/../../b','C:/test','a\\b']:
 try:relative_name(p);checks.append({'name':'reject '+p,'passed':False})
 except ValueError:checks.append({'name':'reject '+p,'passed':True})
checks.append({'name':'preserve Chinese relative path','passed':relative_name('旅行/中文 文件.png')=='旅行/中文 文件.png'})
x=metric(['a','b','c','d','e','f'],{'b','f'})
checks.append({'name':'Hit differs from Recall','passed':x['Hit@5']==1 and x['Recall@5']==.5 and x['MRR']==.5})
checks.append({'name':'no fabricated NDCG','passed':'NDCG@10' not in x})
(ROOT/'reports/offline_checks.json').write_text(json.dumps({'status':'OBSERVED','scope':'path validation and metric arithmetic only; NOT model evidence','checks':checks,'time':time.time()},ensure_ascii=False,indent=2))
assert all(x['passed'] for x in checks)
print(len(checks),'offline checks passed')

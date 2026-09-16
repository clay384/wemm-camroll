"""Freeze source-derived relevance before any retrieval evaluation.
COCO categories are source annotations, not newly human-reviewed gold.
"""
from pathlib import Path
import json,hashlib,collections
R=Path(__file__).resolve().parents[1];B=R/'benchmark'
if (B/'FREEZE.json').exists():raise SystemExit('Already frozen; use existing suite')
photos=json.loads((B/'photo_evidence.json').read_text()); gen=json.loads((B/'generated_evidence.json').read_text()); videos=json.loads((B/'video_evidence.json').read_text())
by=collections.defaultdict(list)
for x in photos:
 for c in x['categories']:by[c].append(x['sha256'])
translation={'person':'人','bicycle':'自行车','car':'汽车','motorcycle':'摩托车','airplane':'飞机','bus':'公交车','train':'火车','truck':'卡车','boat':'船','traffic light':'交通信号灯','bench':'长椅','bird':'鸟','cat':'猫','dog':'狗','horse':'马','sheep':'羊','cow':'牛','elephant':'大象','bear':'熊','zebra':'斑马','giraffe':'长颈鹿','backpack':'背包','umbrella':'雨伞','handbag':'手提包','tie':'领带','suitcase':'行李箱','frisbee':'飞盘','skis':'滑雪板','snowboard':'单板滑雪板','sports ball':'球','kite':'风筝','baseball bat':'棒球棒','skateboard':'滑板','surfboard':'冲浪板','tennis racket':'网球拍','bottle':'瓶子','wine glass':'葡萄酒杯','cup':'杯子','fork':'叉子','knife':'刀','spoon':'勺子','bowl':'碗','banana':'香蕉','apple':'苹果','sandwich':'三明治','orange':'橙子','broccoli':'西兰花','carrot':'胡萝卜','hot dog':'热狗','pizza':'披萨','donut':'甜甜圈','cake':'蛋糕','chair':'椅子','couch':'沙发','potted plant':'盆栽','bed':'床','dining table':'餐桌','toilet':'马桶','tv':'电视','laptop':'笔记本电脑','mouse':'鼠标','remote':'遥控器','keyboard':'键盘','cell phone':'手机','microwave':'微波炉','oven':'烤箱','toaster':'烤面包机','sink':'水槽','refrigerator':'冰箱','book':'书','clock':'时钟','vase':'花瓶','scissors':'剪刀','teddy bear':'玩具熊','hair drier':'吹风机','toothbrush':'牙刷'}
categories=sorted([c for c in by if c in translation and len(by[c])>=2 and c!='person'],key=lambda c:(-len(by[c]),c))[:18]
queries=[];qrels=[]
def add(kind,text,targets,group,image=None,exclude=None,basis='source COCO object annotations; incomplete relevance judgments',page=None,no_match=False):
 qid=f'q{len(queries)+1:03d}'; split='dev' if int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%3==0 else 'test'
 queries.append({'id':qid,'type':kind,'text':text,'image_asset':image,'exclude_assets':exclude or [],'group':group,'split':split,'no_match':no_match,'label_status':'source_annotation_or_program_verified; NOT human_gold'})
 for a in targets:qrels.append({'query':qid,'asset':a,'page':page,'grade':1,'basis':basis,'judgment':'relevant'})
for c in categories:add('text_image','包含'+translation[c]+'的照片',by[c],'category:'+c)
for c in categories[:6]:add('text_image','A photo containing '+c,by[c],'category:'+c)
for c in categories[:6]:add('image_image','',by[c][1:],'category:'+c,image=by[c][0],exclude=[by[c][0]],basis='coarse same-object category from COCO; source image excluded')
for c in categories[:6]:add('joint_image','找出包含同类物体的其他照片',by[c][1:],'category:'+c,image=by[c][0],exclude=[by[c][0]],basis='joint image+text coarse object relevance; source excluded')
for x in [g for g in gen if g['type']=='pdf']:
 add('text_pdf',f"{x['topic']}活动在哪里集合，需要携带什么？",[x['sha256']],'document:'+x['topic'],basis='known generation parameters; visual page only',page=1)
for x in videos[:4]:add('text_video',x['description'],[x['sha256']],'synthetic-video:'+x['color'],basis='known generator conditions; static scene native video smoke')
for x in [g for g in gen if g['type']=='text'][:12]:
 add('text_text',f"寻找{x['topic']}中重点为{x['focus']}的活动记录",[x['sha256']],'text:'+x['topic'],basis='known generator conditions')
negative=['身穿宇航服的大象站在月球表面','紫色的飞龙飞过城堡','水下有发光的外星城市','企鹅在驾驶黄色宇宙飞船','绿色恐龙坐在办公室开会','北极熊在弹奏紫色钢琴','红色潜水艇停在沙漠里','长着翅膀的猫飞过火山','月球上的机器人婚礼','透明的鲸鱼漂浮在图书馆']
for i,t in enumerate(negative):add('text_image',t,[],f'negative:{i}',no_match=True,basis='visual corpus audit required')
assert len(queries)==70,len(queries)
(B/'queries.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in queries));(B/'qrels.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in qrels));(B/'splits.json').write_text(json.dumps({x['id']:x['split'] for x in queries},indent=2))
manifest=[]
for p in sorted((R/'data/manifests').glob('*.jsonl')):manifest+=p.read_text().splitlines()
(R/'data_manifest.jsonl').write_text('\n'.join(manifest)+'\n')
freeze={'seed':20260911,'hashes':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [B/'queries.jsonl',B/'qrels.jsonl',B/'splits.json',R/'data_manifest.jsonl']},'quality_scope':'independent small-scale validation, incomplete relevance labeling; not official MMEB','recall_scope':'annotated evidence only','aggregation':'max unit score per source file','negative_status':'pending corpus contact-sheet audit','no_configuration_selection_on_test':True}
(B/'FREEZE.json').write_text(json.dumps(freeze,indent=2));print(collections.Counter(q['split'] for q in queries))

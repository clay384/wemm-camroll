"""Deterministic synthetic functional samples; no model-generated labels."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
R=Path(__file__).resolve().parents[1]; D=R/'import/demo/generated'; D.mkdir(parents=True,exist_ok=True)
(R/'benchmark').mkdir(exist_ok=True); (R/'data/manifests').mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
records=[]; evidence=[]
def record(p,meta):
 sha=hashlib.sha256(p.read_bytes()).hexdigest();records.append({'path':str(p.relative_to(R/'import')),'sha256':sha,'license':'CC0-1.0','source':'scripts/generate_samples.py','seed':20260911,'kind':'synthetic'})
 evidence.append({'sha256':sha,**meta})
colors=[('red','红色'),('blue','蓝色'),('green','绿色'),('orange','橙色')]
for i,(color,zh) in enumerate(colors):
 for j,shape in enumerate(['circle','square','triangle']):
  im=Image.new('RGB',(512,384),'white');d=ImageDraw.Draw(im)
  if shape=='circle':d.ellipse((150,85,360,295),fill=color)
  elif shape=='square':d.rectangle((150,85,360,295),fill=color)
  else:d.polygon([(256,60),(125,300),(387,300)],fill=color)
  p=D/f'g{i*3+j:02d}.png';im.save(p);record(p,{'type':'graphic','color':zh,'shape':['圆形','正方形','三角形'][j]})
topics=[('城市骑行','头盔','周六','东门',18),('森林徒步','雨衣','周日','北门',12),('海边摄影','三脚架','周五','南门',24),('室内烘焙','面粉','周三','西门',16),('植物观察','放大镜','周二','温室',10),('铁路旅行','车票','周四','车站',32),('图书整理','标签纸','周一','图书馆',14),('羽毛球训练','球拍','周六','体育馆',20)]
for i,(topic,item,day,place,count) in enumerate(topics):
 p=D/f'd{i:02d}.pdf'; c=canvas.Canvas(str(p),pagesize=(595,842),invariant=1)
 for page in range(2):
  c.setFillColorRGB(.08,.13,.22);c.setFont('STSong-Light',24);c.drawString(52,770,topic+'活动手册')
  c.setFont('STSong-Light',14)
  lines=[f'集合日期：{day}    集合地点：{place}',f'计划人数：{count if page==0 else count+4} 人',f'需要携带：{item}', '安排：签到、讲解、实践、讨论。', '请提前十分钟到达，结束后整理个人物品。'] if page==0 else ['备用方案与物资分配',f'备用人数：{count+4} 人',f'主要物资：{item}', '第一组负责准备，第二组负责记录。', '遇到天气变化，以现场通知为准。']
  for y,line in enumerate(lines):c.drawString(52,712-y*34,line)
  c.setStrokeColorRGB(.6,.66,.72)
  for y in range(4):c.line(52,450-y*44,543,450-y*44)
  for x in [52,230,390,543]:c.line(x,318,x,450)
  for j,row in enumerate([['环节','时间','地点'],['准备','09:00',place],['活动','09:30',place]]):
   for x,txt in zip([64,242,402],row):c.drawString(x,420-j*44,txt)
  c.setFont('STSong-Light',12);c.drawString(52,50,f'独立生成的功能测试文档 - 第 {page+1} 页');c.showPage()
 c.save();record(p,{'type':'pdf','topic':topic,'item':item,'day':day,'place':place,'count':count})
for i in range(24):
 topic,item,day,place,count=topics[i%8]; version=i//8
 text=f'{topic}活动记录\n\n第{version+1}次活动在{day}举行，集合地点是{place}。参加人数为{count+version}人。\n\n参与者需要携带{item}。上午进行准备和实践，下午交流观察结果。'+['\n\n本次重点是安全检查。','\n\n本次重点是记录流程。','\n\n本次重点是小组协作。'][version]
 p=D/f't{i:02d}.txt';p.write_text(text);record(p,{'type':'text','topic':topic,'focus':['安全检查','记录流程','小组协作'][version]})
(R/'data/manifests/generated.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
(R/'benchmark/generated_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
print('generated',len(records))

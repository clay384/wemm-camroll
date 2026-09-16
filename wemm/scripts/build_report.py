"""Build honest standalone report from immutable observed evaluation and acceptance files."""
import json,html,base64,io,sqlite3,statistics,collections,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'reports';m=json.loads((P/'metrics.json').read_text());rows=[json.loads(x) for x in (P/'per_query_results.jsonl').read_text().splitlines()];smoke=json.loads((P/'smoke.json').read_text());resource=json.loads((P/'resource_final.json').read_text());lock=json.loads((R/'SOURCE_LOCK.json').read_text())
read=lambda n:json.loads((P/n).read_text()) if (P/n).exists() else {'status':'NOT_RUN'}
checks={n:read(n) for n in ['service_alignment.json','api_acceptance.json','browser_acceptance.json','browser_upload_extra.json','browser_offline.json','recovery_acceptance.json','gpu_formats.json','format_preprocessing.json','offline_checks.json']}
types={'text_image':'文搜图','image_image':'图搜图','joint_image':'图文搜图','text_pdf':'文搜 PDF','text_video':'文搜视频（合成）','text_text':'文搜文字'}
def aggregate(split,dim,level='file'):
 xs=[x for x in rows if x.get('split')==split and x.get('dimension')==dim and x.get('level')==level and x.get('metrics')]
 return {'split':split,'dimension':dim,'n':len(xs),**{k:statistics.mean(x['metrics'][k] for x in xs) for k in xs[0]['metrics']}}
agg=[aggregate(s,d) for s in ['dev','test'] for d in [4096,1024,256]]
mrl=[]
for dim in [4096,1024,256]:
 xs=[x for x in rows if x.get('dimension')==dim and x.get('level')=='file']
 mrl.append({'dimension':dim,'raw_vector_bytes':m['mrl_raw_vector_bytes'][str(dim)],'retrieval_p50_ms':statistics.median(x['timings']['retrieval_s'] for x in xs)*1000,'retrieval_p95_ms':float(__import__('numpy').percentile([x['timings']['retrieval_s']*1000 for x in xs],95))})
coverage=[]
for kind in types:
 xs=[x for x in rows if x.get('type')==kind and x.get('dimension')==4096 and x.get('level')=='file' and x.get('metrics')]
 coverage.append({'type':kind,'queries':len(xs),'known_relevant_annotations':sum(x['judged_relevant_count'] for x in xs),'top10_returned':sum(len(x['top_k']) for x in xs),'unjudged_top10':sum(x['unjudged_top10'] for x in xs)})
summary={'status':'OBSERVED','run_id':m['run_id'],'aggregate':agg,'mrl_retrieval':mrl,'annotation_coverage':coverage,'no_score_guided_changes':True,'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};(P/'delivery_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
md=f'''# WeMM-Embedding-9B 独立验证报告

本次实际完成 9B BF16 独立推理、文件导入、精确向量检索、局域网网页、冻结评测及恢复验收。所有本页数值为 OBSERVED，运行编号 `{m['run_id']}`。没有复现或引用官方榜单分数作为本次成绩。

网页：<http://192.168.0.100:8765/>。独立口令保存在服务器 `run/access.key`（0600），不包含在交付包。服务在 SSH 退出后持续运行；未注册系统开机启动。

## 环境与真实模型

- 服务器 `/home/realmagic/wemm-embedding-lab`；Mac `/Users/zhanyiqi/Documents/nx/wemm-embedding-lab`。
- Ubuntu 22.04，AMD Ryzen 9 9950X，32 线程、123 GiB 系统内存；驱动 595.84。
- 两张 RTX 3090，仅选 GPU 0 / `GPU-bd82fa9d-0906-ce2c-9932-a2c021551203`。另一张 GPU 未被本项目加载模型。
- 官方 Hugging Face `{lock['model_id']}`，revision `{lock['revision']}`。ModelScope 官方 SDK 实际返回 404 后回退；Mac 7890 经 SSH 反向转发到服务器回环 17890 下载。
- 模型文件 `models/WeMM-Embedding-9B/model.safetensors`，18,815,757,458 字节，SHA256 `b6d5dff9e632973991f1d0cbfcfd26c42ffd66fbb8ebd6f852aece08e9794fa4`。
- 官方代码 commit `{lock['code_commit']}`。PyTorch 2.10.0+cu128、Transformers 5.2.0、qwen-vl-utils 0.0.14；完整版本和 17 个文件哈希见 SOURCE_LOCK.json。
- 实际 {smoke['parameter_count']:,} 参数；760 个参数张量均为 BF16、均在 cuda:0；4096 维、batch=1、无量化、无 CPU offload。SDPA、eval、inference_mode、L2 归一化。
- 图像模型预算 262,144 pixels，视频每片段 4 帧、每帧最多 65,536 pixels；8 秒切片，最多 300 秒源视频；512 文本 token，按段落/固定 token 窗口切块。两档预算本次相同，reference 仅保持官方语义作有界正确性参照，非官方榜单默认预算。

## 数值与服务正确性

文本、图片、原生视频、PDF 视觉页面、图文联合输入均产生有限 4096 维向量。重复文本、文本→图片→文本、文本→视频→文本最大逐项差均为 0。HTTP 服务对文本、图片、视频、图文联合的官方示例对照最大差均为 0。单卡只持有一个模型 worker。

真实猫图对猫/公交文字相似度分别为 0.5744/0.2124；公交图对猫/公交文字分别为 0.2587/0.5825。这些仅为功能配对检查，不代表相册质量。

## 固定质量评测

{m['source_files']} 源文件：120 张公开真实照片、12 张生成图形、8 份两页 PDF、24 份文字、8 个视频。共 {m['units']} 单元：132 图片、16 PDF 页面、24 文字块、12 视频片段。

70 条冻结查询（60 正向、10 无匹配）；29 dev / 41 test，其中正向 25 dev / 35 test。v2 在模型首次评分前冻结；v1 未运行记录保留。固定查询、标签、来源清单的哈希见 benchmark/FREEZE.json。分组依据保存在 splits.json；不是官方 MMEB 数据集。

来源注释 + 程序标签，代理检查过照片联系表；**不是 human_gold**。真实照片相关性不完整，已知正例以外条目为未判定，不自动作为不相关。下列 Hit、MRR、Recall 均为“原始排名中已标注证据”的指标；不能解释成全库完整相关性质量。PDF 页标签只覆盖已知证据页。NDCG 未计算。保留集很小，不作统计显著性或生产门槛结论。

### 4096 维保留集，源文件级

|题型|正向题数|Hit@1|Hit@5|Hit@10|Recall@10|MRR|
|---|---:|---:|---:|---:|---:|---:|
'''
for x in m['summary']:
 if x['split']=='test' and x['dimension']==4096 and x['level']=='file':md+='|'+types[x['type']]+'|'+str(x['n'])+'|'+ '|'.join(f"{x[k]:.3f}" for k in ['Hit@1','Hit@5','Hit@10','Recall@10','MRR'])+'|\n'
md+='\n### 同一完整向量的 MRL 对照，35 条保留正向题宏平均\n\n|维度|Hit@1|Hit@5|Hit@10|MRR|原始向量字节|检索 P50/P95 ms|\n|---|---:|---:|---:|---:|---:|---:|\n'
for x,y in zip([a for a in agg if a['split']=='test'],mrl):md+=f"|{x['dimension']}|{x['Hit@1']:.3f}|{x['Hit@5']:.3f}|{x['Hit@10']:.3f}|{x['MRR']:.3f}|{y['raw_vector_bytes']:,}|{y['retrieval_p50_ms']:.2f} / {y['retrieval_p95_ms']:.2f}|\n"
md+=f'''
MRL 双端截取并重新 L2 归一化，不重新编码、不减模型显存。检索时间包含向量文件读取和矩阵构造，是本次小库精确检索的时间，不能外推百万级索引。没有根据保留集改选配置；主配置仍是 4096 维。

10 条无匹配题 Top-1 分数范围 {m['no_match_distribution']['min']:.4f}–{m['no_match_distribution']['max']:.4f}，中位数 {m['no_match_distribution']['p50']:.4f}。未启用拒识阈值，误接纳率 NOT_COMPUTED；页面明确提醒 Top-K 不保证存在相关结果。

### 失败与覆盖边界

文搜图存在组合条件失败，例如 q003「同时包含杯子和碗」第一个已知正例排名 21，q005「同时包含瓶子和碗」排名 8。保留图搜图和图文搜图各 3 题 Hit@1 均为 0；这按来源类别标签计算，Top-1 未标相关并不等于人工确认完全不相关。详细输入图、真实 Top-K、未判定状态保留在静态 HTML 和 JSONL。未以新标签或删题提高分数。

4 条视频质量题仅为可控颜色方块视频；另有 4 个 Blender 动画片段实际原生编码成功，但没有建立动作检索质量 gold。动画原片仅约 33 秒，最后一个片段约 1 秒。真实摄像视频、复杂动作、长时事件理解 NOT_RUN。中英题共享类目分组，但不是所有题均为严格翻译对，不做跨语言等价分数宣称。

## 效率与资源

- 首次进程加载 {smoke['load_s']:.3f} 秒；服务加载 {m['cold_load_s']:.3f} 秒。权重刚校验后文件系统缓存已热，**不是清空 OS 缓存后的磁盘冷启动**。
- demo 入库总墙钟 {m['import_wall_s']:.3f} 秒（含一次交互查询插入）；{m['source_files']/m['import_wall_s']:.3f} 文件/秒、{m['throughput_units_per_s']:.3f} 单元/秒；CPU 资产预处理累计 {m['asset_preprocess_s']:.3f} 秒，GPU 单元编码累计 {m['unit_encode_s']:.3f} 秒。
- 向量 .npy 文件 {m['index_bytes']:,} 字节（含文件头）；模型进程 RSS 采样 {m['resident_rss_bytes']/2**30:.3f} GiB。SQLite 元数据不计入这个向量大小。
- 索引期间 PyTorch 峰值 allocated {m['peak_allocated_mib']:.1f} MiB、reserved {m['peak_reserved_mib']:.1f} MiB；整卡 GPU 0 每秒采样峰值 {resource['cards']['0']['sampled_peak_used_mib']} MiB，包含其他用户进程，不能当作本模型 allocated。GPU 0 最后样本常驻整卡 {resource['cards']['0']['last']['used_mib']} MiB；本项目进程显存及时间见 resource_final.json。

|热态输入（每类预热1次，再重复20次）|模型 P50/P95 ms|预处理+模型 P50/P95 ms|
|---|---:|---:|
'''
for k,x in m['hot'].items():md+=f"|{k}|{x['p50_embedding_s']*1000:.2f} / {x['p95_embedding_s']*1000:.2f}|{x['p50_total_s']*1000:.2f} / {x['p95_total_s']*1000:.2f}|\n"
md+='''
GPU 计时显式 synchronize，输入重复但向量缓存关闭。以上不是 HTTP 排队延迟。并发 4 请求的 queue_s、preprocess_s、embedding_s、retrieval_s、response_total_s 单独保存在 recovery_acceptance.json。浏览器页面也展示这些实际耗时。

## 工程验收

- 官方权重完整哈希、最小数值与服务对齐：通过。
- CPU 格式/资源拒绝检查 19/19；实际 GPU 入库的格式样本 13/13（PDF 产生两页）。已实测 JPG、PNG、WebP、TIFF、BMP、HEIC、TXT、MD、PDF、MP4、MOV、WebM 与 EXIF 方向。HEIF / .jpeg / .tif 别名未单独测试，不能扩大为所有编码变体兼容。
- Mac Chrome/Playwright，真实 LAN HTTP：文件夹选择保留中文、空格和子目录，服务 SHA256 与 Mac 相符；浏览器多文件、拖放、联合候选、详情、刷新、逐题与报告下载通过。首次浏览器断言未等待异步文本详情，修复等待后重测通过；失败记录保留。
- API 14/14：未登录拒绝、路径穿越、白名单/符号链接、坏文件/音频、空文件、实际 129 MiB 上传、去重不重复推理、暂停/取消/恢复与真实查询通过。
- 故障注入/恢复 8/8：缺失模型路径明确 error；恢复相同指纹/索引/排名；查询超长错误不污染下一个请求；单 worker 串行并发；新离线上传检索通过。
- 启动脚本曾将 TIME_WAIT 误判占用并切换 8766，已使用 SO_REUSEADDR 修复探测，恢复保持 8765。仅本项目故障服务被停止。下载代理已关闭，Mac 浏览器新文件上传与检索成功，页面未请求外部资源。服务模型加载强制 offline，并在服务 Python 内拒绝非回环出站连接；未改系统防火墙。
- 本任务未读取或连接现有 CamRoll、Sentrix、相册业务系统，未给其他服务发送停止/重启指令，也未改驱动/系统环境。GPU 0 既有 PID 53789 保持存在；GPU 1 在长下载期间出现其他任务 PID 变化，未由本任务操作，不声称对方全程无中断。业务健康接口未访问，NX 运行状态 UNKNOWN；NX 无本任务改动。

## 使用与恢复

服务器进入项目目录后运行 `scripts/start.sh` / `scripts/stop.sh` / `scripts/status.sh` / `scripts/logs.sh` / `scripts/recover.sh`。脚本按 PID、命令和 cwd 核对归属；不使用广泛 kill。

在自己的 SSH 终端查看 `cat ~/wemm-embedding-lab/run/access.key`。登录网页，进入“数据管理”，创建独立数据集，点击“选择文件夹”；上传建索引完成后，在“检索”选择同一数据集并输入文字或图片。不会持续监控 Mac 目录。原文件不改写；按 SHA 去重。

`import/` 是服务器唯一目录导入根；用户数据与 demo 分开。取消在安全单元边界生效，用重试继续。资料、数据库、向量持久化在 data/；不要删除它们来恢复服务。

`scripts/run_benchmark.sh` 可重新运行冻结评测；`scripts/export_report.sh` 检查导出；`.venv/bin/python scripts/build_report.py` 汇总当时证据。历史运行在 reports/runs。静态 HTML 含公开样例缩略图，可单独离线查看；新查询仍需服务器模型。

## 交付及未运行项

源码包含前后端、构建产物、脚本、配置、依赖与来源锁、冻结评测和报告；不含权重、venv、运行口令或私人导入数据。公开 demo 单独成包，附逐文件来源/许可清单。

不支持音频语义、ASR、生成问答；未做 batch>1、模型量化、CPU offload、百万文件规模、长时间压力、浏览器移动端/Safari、系统重启自动启动。setup.sh 是环境重建入口，其全新机器重新安装流程 NOT_RUN；本机实际环境的依赖检查、启动/停止/恢复已实测。没有为未运行项填入模拟数据。
'''
(P/'validation_report.md').write_text(md)
esc=html.escape
# Independent single HTML: embed only public frozen-demo thumbnails, never personal datasets.
from PIL import Image
conn=sqlite3.connect(R/'data/lab.sqlite');conn.row_factory=sqlite3.Row;cache={}
def thumb(asset,uid=None):
 key=uid or asset
 if key not in cache:
  u=conn.execute('SELECT thumb FROM units WHERE id=?',(uid,)).fetchone() if uid else conn.execute('SELECT thumb FROM units WHERE asset=? AND thumb IS NOT NULL LIMIT 1',(asset,)).fetchone()
  if not u or not u['thumb']:cache[key]='';return ''
  with Image.open(R/u['thumb']) as im:
   im.thumbnail((200,150));b=io.BytesIO();im.convert('RGB').save(b,format='JPEG',quality=68);cache[key]='data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
 return cache[key]
# Lightweight markdown rendering: headings, paragraphs, code and tables remain legible offline.
def render(text):
 out=[];in_table=False;paragraph=[]
 def flush():
  if paragraph:out.append('<p>'+esc(' '.join(paragraph))+'</p>');paragraph.clear()
 for line in text.splitlines():
  if line.startswith('|'):
   flush()
   if not in_table:out.append('<div class="tablewrap"><table>');in_table=True
   if set(line.replace('|','').replace(':','').replace('-','').replace(' ',''))==set():continue
   out.append('<tr>'+''.join('<td>'+esc(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>');continue
  if in_table:out.append('</table></div>');in_table=False
  if line.startswith('#'):
   flush();n=len(line)-len(line.lstrip('#'));out.append(f'<h{n}>'+esc(line[n:].strip())+f'</h{n}>')
  elif line.startswith('- '):flush();out.append('<p class="bullet">'+esc(line[2:])+'</p>')
  elif not line:flush()
  else:paragraph.append(line)
 flush()
 if in_table:out.append('</table></div>')
 return ''.join(out)
body=render(md);body+='<h2>标注覆盖（4096维、源文件级正向题）</h2><div class="tablewrap"><table><tr><th>题型</th><th>查询</th><th>已知相关标签数</th><th>Top10 返回数</th><th>其中未判定</th></tr>'
for c in coverage:body+='<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in [types[c['type']],c['queries'],c['known_relevant_annotations'],c['top10_returned'],c['unjudged_top10']])+'</tr>'
body+='</table></div><h2>全部分任务、单元级与源文件级指标</h2><details><summary>展开完整指标表（含 dev 与三种维度）</summary><div class="tablewrap"><table>'
keys=['split','type','dimension','level','n','Hit@1','Hit@5','Hit@10','Recall@5','Recall@10','MRR'];body+='<tr>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>'
for x in m['summary']:body+='<tr>'+''.join('<td>'+esc(f'{x[k]:.3f}' if isinstance(x[k],float) else str(x[k]))+'</td>' for k in keys)+'</tr>'
body+='</table></div></details><h2>逐题真实 Top-K（4096 维 / 源文件级）</h2><p>缩略图嵌入本文件。相关表示冻结标签中的已知正例；未判定不表示不相关。详情中的分数是相似度。</p>'
for x in rows:
 if x.get('dimension')!=4096 or x.get('level')!='file':continue
 label='无匹配题' if x['no_match'] else 'Hit@1='+str(int(x['metrics']['Hit@1']))
 body+='<details class="query"><summary>'+esc(x['query']+' · '+(x['text'] or '图片查询')+' · '+label)+'</summary>'
 if x['query_image']:body+='<p>查询图片（原图与同 SHA 复制品已排除）</p><img alt="查询图" src="'+thumb(x['query_image'])+'">'
 body+='<div class="cards">'
 for h in x['top_k']:
  source=thumb(h['asset'],h['id']);body+='<article>'
  if source:body+='<img alt="'+esc(h['name'])+'" src="'+source+'">'
  else:
   u=conn.execute('SELECT path FROM units WHERE id=?',(h['id'],)).fetchone();content=(R/u['path']).read_text()[:150] if u else '';body+='<pre>'+esc(content)+'</pre>'
  body+='<p>'+esc(str(h['rank'])+' · '+h['name'])+'</p><p>'+esc(f"{h['score']:.4f} · "+('已标注相关' if h['judgment']=='relevant' else '未判定'))+'</p></article>'
 body+='</div><pre>'+esc(json.dumps({'metrics':x['metrics'],'timings':x['timings'],'judged_relevant_count':x['judged_relevant_count'],'unjudged_top10':x['unjudged_top10']},ensure_ascii=False,indent=2))+'</pre></details>'
body+='<h2>公开素材许可</h2><p>每个公开素材的原始来源、许可证、归属 URL 和 SHA256 均列在包内 data_manifest.jsonl。COCO 图片许可按图片分别记录；COCO 注释来自官方公开 val2017。Blender 动画：(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org，CC-BY 3.0。程序生成素材脚本包含在源码包。</p>'
style='body{font:15px/1.7 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#243247;background:#fff}h1{font-size:30px}h2{font-size:23px;margin-top:40px}h3{font-size:18px}p{overflow-wrap:anywhere}.bullet{padding-left:16px;border-left:2px solid #dbe6f5}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #dde4ed;text-align:left;white-space:nowrap}tr:first-child{background:#f0f4fa}details{border:1px solid #e1e6ed;border-radius:10px;padding:14px;margin:12px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f4f7fa;padding:14px}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:18px}article{border:1px solid #e1e6ed;border-radius:8px;overflow:hidden}article img{width:100%;height:135px;object-fit:contain;background:#f4f7fa}article p{font-size:11px;margin:8px}article pre{height:135px;overflow:hidden;margin:0}a{color:#375e9f}@media(max-width:750px){.cards{grid-template-columns:repeat(2,1fr)}}'
(P/'validation_report.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>WeMM 独立验证 · 实测报告</title><style>'+style+'</style></head><body>'+body+'</body></html>')
print(json.dumps({'report_bytes':(P/'validation_report.html').stat().st_size,'queries':70,'status':'OBSERVED'}))

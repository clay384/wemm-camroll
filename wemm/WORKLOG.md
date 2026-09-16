## 2026-09-11 — 完成交付与恢复验收

- 18,815,757,458字节官方9B完整SHA校验通过；BF16单卡实际推理/四模态HTTP官方对齐通过。
- 172文件/184单元完成入库，冻结70题×3维度×2级别实际运行，无异常；低分与未判定保留。
- 完成浅色中文网页、浏览器目录/多文件/拖放/详情/联合候选/报告及离线新上传检索。
- API14项、恢复8项、GPU格式13样本、CPU19项通过。首次浏览器文本详情断言缺异步等待，修正后重测。
- 故障恢复发现TIME_WAIT误判端口，修复项目manage.py的SO_REUSEADDR后验证8765恢复；仅重启本项目PID。故意错误路径显示error后恢复相同指纹、索引、排序与分数。
- 整理时发现Mac旧检查点覆盖了远端恢复JSON，保留通过日志并重新执行恢复8项生成新证据；质量评测未重跑或更改。
- 关闭本任务反向代理，停止专属GPU监测器；服务保留后台运行。未读取或修改既有业务项目，未发付费/API请求。
- 同步源码、构建产物、报告与公开demo；源码包排除权重、环境、口令、私人导入。全新机器setup安装未重跑，明确NOT_RUN。

# Worklog
## 2026-09-11 — Environment and preprocessing verified
OBSERVED: torch 2.10.0+cu128 / transformers 5.2.0 / BF16 CUDA tensor test on selected RTX 3090; pip check passed. Text/image/native-video processor ends at <embedding>. CPU format checks 19/19 passed, including JPEG/PNG/WebP/TIFF/BMP/HEIC, orientation, TXT/MD/PDF, MP4/MOV/WebM and rejection cases. Fixed PyMuPDF deprecated-import stdout contamination and reproduced/fixed OpenBLAS thread startup under the child memory limit. Failed evidence retained. Full 9B inference remains NOT_RUN until final checksum.
Frozen dataset v2 before any inference: deterministic PDF metadata/footer fix, measured trailer endpoint, six object-cooccurrence queries; v1 preserved. No model scores were available or used.

## 2026-09-11 — Isolated preparation and download recovery
Locked official code 6f4f007b9105fe2d0a3176d719b8799e3bf6974e and HF model revision 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482. Reviewed model code, processor, tokenizer postprocessor and license. Xet stalled after a 4,112,443,339-byte sequential prefix; stopped only task-owned downloader, preserved prefix and began bounded 8-way HTTP Range continuation with full final SHA256 required.
Prepared 120 licensed COCO photos, 44 generated image/PDF/text sources, 4 synthetic videos and 4 licensed Blender animation excerpts. Visual contact-sheet audit performed by assistant (not human gold). Frozen 70 queries before model evaluation. Added bounded preprocessing, persistent metadata/vector pipeline, single worker service source, exact retrieval/evaluator and project-only process scripts. None of these constitute live GPU or browser success yet.

## 2026-09-11 — Initialization
Read full task brief. Completed SSH key preflight; created isolated directories and venv. ModelScope SDK metadata lookup failed with 404 record not found; initiated official HF fallback using project SSH reverse tunnel. No password used. No existing service modified.

## 2026-09-14 PhotoBench preparation
OBSERVED: authorized .100 GPU1 and .153 GPU0; 344 media copied to independent import; 864 question-only inputs frozen; scorer tests passed. Rewrite RUNNING on .153; WeMM indexing/evaluation NOT_RUN. Old demo reports preserved.

## 2026-09-14 — Rewrites frozen and GPU1 selected
OBSERVED: 864 rewrites finished; original/semantic/keyword queries frozen before retrieval. 17 WeMM source hashes verified; physical GPU1 UUID selected. Indexing/evaluation next.

OBSERVED import rejection: one 42,368,256-pixel panorama exceeded 40M decode cap. Increased bounded source decode cap to 45M before any retrieval scores; model image budget unchanged at 262,144 pixels. Initial failure retained, retry task dataset only.

OBSERVED native-video regression: three 8.042s videos produced one-frame terminal clips. Repeated last frame to reach fixed four-frame budget; no source frames discarded, source originals unchanged. CPU regression covers three failing clips plus ordinary video. Preprocess fingerprint changes; rebuild all task-service vectors before retrieval, no query scores seen.

## 2026-09-14 — Unified index complete; evaluation running
OBSERVED: all 344 PhotoBench assets indexed: 334 image units + 18 native video units = 352. Finite normalized 4096-vectors verified; fingerprint cb0b50fe0c768df7ec200dcc15160643ce11660e0ffea39e48eafc451e6a4cf7. 4,992 fixed-query API requests in progress; no prompt or scoring adjustments after retrieval began.

## 2026-09-14 — PhotoBench complete
OBSERVED：WeMM 网页 http://192.168.0.100:8765/ 正在运行，物理 GPU1 UUID GPU-4ef9200c-d8f3-638a-3eeb-26fe4fa3d7b0。BF16 9B / 4096维 / 单worker，无量化、无CPU offload。GPU0现有任务未控制。

独立运行 reports/photobench/20260914-115425：344媒体（334图+10视频）/352单元；864条问题的A原文、B语义改写、C关键词已冻结；800题图片主测+859题混合补测+5题补充，共4992次检索，零失败。

图片主测 A Hit@1/5/10=80.25%/95.625%/97.625%，Recall@10=94.639%。B Hit@10相同，C为96.875%；两种改写Hit@1均低于原问题。视频目标题39题在混合库中的Hit@10为A53.85%、B56.41%、C69.23%。完整结论见同目录 ANALYSIS.md。

验收：模型17文件、数据354文件SHA256一致；352向量有限/归一化；客户端1086条后实际中断续跑无重复；服务重启后9项抽样Top50排名分数完全一致（最大差0），向量哈希不变，旧demo报告未改。浏览器筛选、搜索、Top50、图片预览、视频完整播放、分页通过。

导入修复：原图解码上限45M以覆盖42.37MP全景图（模型预算仍262144）；不足4帧的视频尾片段重复末帧补足，保留原始内容和补帧元数据。修复后指纹 cb0b50fe0c768df7ec200dcc15160643ce11660e0ffea39e48eafc451e6a4cf7，统一重建后开始评分。失败和调整记录保留。

.153 Qwen3-4B-Instruct-2507独立改写进程已完成并退出；无业务服务修改。NX业务代码未改，NX运行状态未探测。

恢复：scripts/status.sh / start.sh / stop.sh；保持GPU1配置，不自动改回GPU0。复跑：scripts/photobench/README.md。服务未注册开机自启。报告本机预览127.0.0.1:8877只对Mac本机开放；report.html同目录媒体可离线查看。

限制：仅针对已标注证据；图片QA human_verified=false；视频为照片生成素材；没有router、元数据/身份增强、真实长视频或长时压力验证。


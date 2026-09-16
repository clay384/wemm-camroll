# WeMM 检索实验室

## 最新 PhotoBench 对照（2026-09-14）

已完成334图+10视频建库与4992次检索，服务改为物理GPU1。800题图片主测原问题Hit@1/5/10为80.25%/95.625%/97.625%；默认保留原问题，固定改写未改善整体图片召回。细节、逐题图像及视频对照见 `reports/photobench/20260914-115425/report.html`，结论见同目录 `ANALYSIS.md`，复跑见 `scripts/photobench/README.md`。9月11日的公开demo报告与包保留为历史交付，未覆盖。

独立验证 Tencent WeMM-Embedding-9B。与 CamRoll、Sentrix、NX 和已有相册服务无连接。运行状态以 CURRENT_STATE.md 和本次实测报告为准。

## 运行

服务器目录 `/home/realmagic/wemm-embedding-lab`。进入目录后：

```sh
scripts/start.sh
scripts/status.sh
scripts/logs.sh
scripts/stop.sh
scripts/recover.sh
```

网页 `http://192.168.0.100:8765` 已从 Mac Chrome 实际访问、上传和检索。 项目口令首次启动生成在服务器 `run/access.key`，权限 0600；在自己的 SSH 终端查看即可。不使用 SSH 密码。源码包不包含该文件。会话有效期 24 小时。

## 导入与检索

打开“数据管理”，创建或选择数据集，再“选择文件夹上传”或“选择多个文件”。文件夹内容会真实上传到服务器，保留相对目录；不会监控 Mac 文件系统。每个文件最多 128 MB。也可导入服务器项目 `import/` 内的相对路径，不允许任意服务器目录。

“检索”支持文字、图片以及同次编码的图文联合查询。Top-K 是相似度排序，不是正确率，也不保证库中存在相关结果。照片、PDF 页、视频片段、文本块可过滤；按文件聚合使用最高单元分数，展开查看其他命中。图像自匹配默认排除原图内容 SHA256。

默认是纯 WeMM：无 Caption/OCR 补充、BM25、外部 LLM、重排和文件名语义输入。PDF 页作为视觉输入；视频不理解音频。视频模式为原生视频编码，转码仅是明确记录的导入预处理。

## 恢复与评测

```sh
scripts/import_demo.sh
scripts/run_benchmark.sh
scripts/export_report.sh
```

同一 SHA256 文件不重复编码；暂停/继续和取消/重试在安全单元边界生效。数据库和向量保存在 `data/`。重启恢复中断资产；已完成单元跳过。模型、预处理或编码代码指纹变化后旧索引不参与新查询，相关资产重新入队。

固定评测保存在 `benchmark/`，与索引内容分离。源注释、程序生成标签和代理视觉复核均不称本轮人工 gold。Recall 仅针对已标注证据。测试集不用于反复选择输入配置。MRL 1024/256 从同一批 4096 向量截取后重新归一化，不改变模型权重、编码时间或显存。

静态 `reports/validation_report.html` 可离线查看已有结果；交互式新检索必须运行服务器。逐查询证据和历史运行保存在 `reports/runs/`。

## 环境与模型

`SOURCE_LOCK.json` 固定来源、revision、文件 SHA256、官方代码 commit 与实际软件版本。仅使用项目 `.venv`，不修改系统 Python/CUDA/驱动。批次固定 1，单 worker 单卡；参考和交互配置都显式限制输入，非官方榜单配置。

Hugging Face 回退只通过已验证的代理。若服务器没有本机代理，可在 Mac 建立仅回环反向转发：

```sh
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -R 127.0.0.1:17890:127.0.0.1:7890 realmagic@192.168.0.100
```

服务器只给下载进程传代理：

```sh
HTTPS_PROXY=http://127.0.0.1:17890 scripts/download_model.sh
.venv/bin/python scripts/lock_sources.py
```

不关闭主机密钥校验。服务启动脚本清除代理并强制模型离线加载。部署管理只检查本项目 PID、命令与工作目录，不停止他人服务。服务未注册为系统启动项；服务器重启后运行 `scripts/start.sh` 恢复。

## 源码包

包含 backend、静态前端源码/构建产物、脚本、配置、版本锁、任务记录与报告。不包含模型权重、venv、缓存、访问口令或私人导入文件。公开 demo 单独提供 `wemm-public-demo.tar.gz`，来源与逐文件许可在 `data_manifest.jsonl`；源码包不含任何运行访问口令。


## 本次实际结果

172 个源文件 / 184 个单元，冻结 70 查询（60 正向、10 无匹配）。4096 维保留正向题 Hit@1=0.771、Hit@5=0.943、Hit@10=1.000；只衡量已标注证据，非完整人工相关性判断。保留文搜图 Hit@1=0.857；图搜图和图文搜图各 3 题 Hit@1=0，不能据总体均分掩盖失败。详见 `reports/validation_report.md` 与可单独打开的 `reports/validation_report.html`。

本次通过真实 BF16 编码、参考/HTTP 对齐、13 种格式/方向样本入库、Mac 文件夹/多文件/拖放、坏文件拒绝、取消恢复、故障状态、离线新导入与恢复后同排序。下载代理已经关闭。未做长时压力、大规模相册、真实长视频动作质量或 Safari 兼容验证。

## 环境重建入口

当前服务器无需重新安装。新副本的入口如下，依赖全新安装流程未在第二台机器重跑；启动/停止/恢复已在目标机实测：

```sh
scripts/setup.sh
scripts/build_frontend.sh
# 配置授权 GPU UUID；模型下载仍须使用已验证代理：
HTTPS_PROXY=http://127.0.0.1:17890 scripts/download_model.sh
.venv/bin/python scripts/lock_sources.py
# 在项目根解开可选公开 demo 包后：
scripts/start.sh
scripts/import_demo.sh
```

`frontend/src/` 是无外部依赖的原始前端；`frontend/dist/` 是同源托管产物。源码包含固定的官方推理示例及许可，`official-locked/COMMIT` 和 `HASHES.json` 记录来源。不要把 `run/access.key`、私人 `data/` 或任何权重加入源码包。

主报告生成器 `.venv/bin/python scripts/build_report.py` 根据已经运行的 metrics 和验收证据生成静态报告，不重新跑模型或改变评分。新评测后可再次执行。原始质量运行在 `reports/runs/20260911-135239/`，其中保留当时配置和锁文件。

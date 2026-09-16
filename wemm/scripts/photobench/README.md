# PhotoBench 原问题／语义改写／关键词对照

本目录是已有 WeMM 服务的独立评测客户端，不启动第二份 WeMM 模型，不改公共 API，不覆盖历史 demo 评测。

本次运行 ID：`20260914-115425`。服务器工作根目录：`/home/realmagic/wemm-embedding-lab`。

## 流程与恢复

以下命令在 `.100` 工作根目录使用项目 `.venv` 执行：

```sh
.venv/bin/python scripts/photobench/run.py selftest
.venv/bin/python scripts/photobench/run.py verify
.venv/bin/python scripts/photobench/run.py ingest
.venv/bin/python scripts/photobench/run.py evaluate
.venv/bin/python scripts/photobench/run.py export
.venv/bin/python scripts/photobench/acceptance.py --restart
```

`evaluate` 按 `(qa_id, track, arm)` 跳过已持久化记录；每条结果写入后 fsync。输入、查询、索引或评分脚本哈希变化会停止运行，不静默混用。记录失败请求，最多原样重试一次。重跑报告不调用模型。

新实验用 `run.py prepare --run-id 新ID`，后续各阶段传相同 `--run-id`。默认读取同一媒体快照；换快照用 `--source` 并在各阶段保持一致。本工具针对本次 album3 三集合规范，数量断言是有意的。新实验必须重新冻结改写文件；没有改写完成锁时不能运行评测。`acceptance.py` 默认针对本次固定运行。

不直接运行原项目 `/api/benchmark`：该接口绑定旧 demo，会导出覆盖旧报告。本评测只使用现有登录、数据集、目录导入、检索与媒体读取接口。

## 查询改写

`.153` 独立目录：`/home/asus/wemm-photobench-rewrite/20260914-115425`。模型：`/home/asus/models/Qwen3-4B-Instruct-2507`；解释器：`/home/asus/miniconda3/envs/sentrix-vllm/bin/python`。

```sh
python rewrite.py --input questions_only.jsonl --out output \
  --model /home/asus/models/Qwen3-4B-Instruct-2507 \
  --wemm-tokenizer tokenizer.json \
  --gpu GPU-930c71f5-9fed-cfdd-1977-1b4b8e4e1cc5
```

只把 `questions_only.jsonl`、改写脚本和 WeMM tokenizer 提供给改写进程。生成目录不需要 QA、答案、证据或排名文件。输出及模型哈希在 `rewrite_lock.json`，输出完成摘要在 `rewrite_complete.json`。将 output 中四个 JSON/JSONL 文件复制到 `.100` 对应运行目录再评测。两种输出一次联合生成，耗时是共享成本。greedy 重试不会保证修复格式问题；失败如实保留。

## GPU 与输入配置

WeMM 物理 GPU 1：`GPU-4ef9200c-d8f3-638a-3eeb-26fe4fa3d7b0`，进程内逻辑设备 `cuda:0`。保持官方权重、BF16、4096 维与已有模型预算。只使用项目管理脚本启动／停止。

原图解码限制由 40M 调整到 45M pixels，以接纳实测 42,368,256 像素的全景图；模型图片预算仍是 262,144 pixels。输入原件没有改写；原失败记录和调整说明随运行保存。恢复配置前先停止本项目服务，检查目标 GPU 空闲，不能自动恢复到有任务的 GPU 0。

三条约 8.042 秒视频会产生不足四帧的末尾切片。预处理现在对这种切片重复末帧补足四帧，并在单元元数据记录补帧数、原起止时间和原帧数；不丢弃原始内容。此修复改变预处理指纹，因此本轮全部候选在修复后重建索引；旧 demo 报告不变。具体回归证据见 `short_tail_regression.json`。

## 指标与数据边界

- 图片主测：800 题，334 张候选图；混合补测：859 题，344 个候选媒体；缺失 expected_action 的五题单列。303 条非正向交互单列排除。
- SHA256 同内容资产去重；视频按最高片段分数聚合。检索目标使用 typed media refs，答案字段只用于报告核查。
- Hit/Recall/All@1,5,10,20，MRR@50。所有指标只衡量已标注证据；未标注不等于不相关。失败计零，另外提供共同成功题对照。
- 事件级 bootstrap：固定 seed=20260914，2,000 次有放回抽取事件，保留簇内问题并按题数计算均值；2.5/97.5 百分位区间。区间仅描述此固定小数据集，不用于官方榜单或生产门槛声明。
- 本轮不调参、不选最佳提示词、不做 router。图片 QA 的 human_verified=false；大量问题带事件描述依赖。视频是照片生成素材。

## 查看交付

报告是 `reports/photobench/20260914-115425/report.html`，同目录包含缩略图、预览图和视频，可离线查看。也可在 Mac 对该运行目录启动仅回环的静态 HTTP 服务，以浏览器查看。报告不需要服务口令，不包含口令；不要将含私人媒体的报告混入原公开 demo 源码包。

主要记录：`freeze.json`、`protocol.json`、`source_transfer_verified.json`、`weights_verified.json`、`evaluation_lock.json`、`index_verified.json`、`results.jsonl`、`summary.csv`、`comparisons.csv`、`errors.json`、`live_before_restart.json`、`live_after_restart.json`。浏览器实际验收另存 `browser_acceptance.json`。

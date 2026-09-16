# 接手清单与证据位置

## 接手时应知道

**2026-09-16 更新：当前禁止修改 `.100` 和 `.153` 的文件。只读本地资料，服务器操作和接口验证不在本次范围。** 新交接入口为 [CamRoll 搭建与改动总结](CAMROLL_BUILD_SUMMARY_20260916.md)，包含源码、接口、运行数据及报告位置。下文的实施授权和运维步骤属于历史记录，不能覆盖此最新边界。

本页保留先前仅文档交接阶段的 WeMM 历史基线。2026-09-15 用户随后明确授权实施 CamRoll → Qwen3.7-Flash + WeMM；当时的实施约定见 [AGENTS.md](../AGENTS.md)，当前范围以本页顶部最新说明为准，新代码与部署记录从 [CamRoll README](../camroll/README.md) 进入。以下 2026-09-14 冻结验收仍保留，不能替代新服务的实时健康检查。

| 项目 | 精确值 |
|---|---|
| WeMM 服务器 | realmagic@192.168.0.100 |
| QA / Qwen 服务器 | asus@192.168.0.153 |
| 远端项目 | /home/realmagic/wemm-embedding-lab |
| Mac 副本 | /Users/zhanyiqi/Documents/nx/wemm-embedding-lab |
| 实验 run_id | 20260914-115425 |
| dataset_id | ad8f153d865a4d0caa5d1704b0874426 |
| dataset 名称 | PhotoBench album3 · 20260914-115425 |
| 远端报告 | /home/realmagic/wemm-embedding-lab/reports/photobench/20260914-115425/ |
| Mac 报告 | /Users/zhanyiqi/Documents/nx/wemm-embedding-lab/reports/photobench/20260914-115425/ |
| 登录密钥 | 各项目 run/access.key；内容不纳入交接文档 |

## 已交付

[可交互 HTML 报告](/Users/zhanyiqi/Documents/nx/wemm-embedding-lab/reports/photobench/20260914-115425/report.html)提供原问题、B/C 对照、目标图、Top-K、过滤和视频详情。旁边的文件共同组成可追溯交付：

| 类型 | 文件 |
|---|---|
| 阅读结论 | RESULTS.md、ANALYSIS.md |
| 全量逐题结果 | results.jsonl、errors.json、evaluation_complete.json |
| 汇总 / 对照 / 耗时 | summary.csv/json、comparisons.csv/json、timings.json |
| 冻结规则 | freeze.json、protocol.json、evaluation_lock.json |
| 数据映射与选题 | media.json、eligible.jsonl、excluded.jsonl、questions_only.jsonl、dataset.json |
| 来源证明 | source_hashes_asus.json、source_transfer_verified.json、weights_verified.json |
| 改写 | rewrites.jsonl、rewrite_lock.json、rewrite_runtime.json、rewrite_complete.json |
| 验收 | index_verified.json、scorer_tests.json、resume_verified.json、live_after_restart.json、browser_acceptance.json |
| 同步证明 | delivery_manifest.json、delivery_verified.json |
| 失败历史 | import_failure_history.json、import_video_failures_before_fix.json、short_tail_regression.json |
| 配置及修改前记录 | config.yaml.before、CURRENT_STATE.md.before、WORKLOG.md.before、preprocess.py.before |

原项目 [CURRENT_STATE.md](/Users/zhanyiqi/Documents/nx/wemm-embedding-lab/CURRENT_STATE.md) 和 [WORKLOG.md](/Users/zhanyiqi/Documents/nx/wemm-embedding-lab/WORKLOG.md) 是运行历史；本次没有更新它们。复跑代码在 [scripts/photobench](/Users/zhanyiqi/Documents/nx/wemm-embedding-lab/scripts/photobench)。旧 deliveries 下源码/demo 压缩包属于旧交付，不应假设包含后来 PhotoBench 的全部修改。

## 接手顺序

1. 阅读本页、API 和测试文档；确认要看历史结果还是执行新的实验。
2. 只看结果时打开 Mac 本地报告，不需要启动模型或访问 `.100`。
3. 新运行先检查资源、模型配置、原始文件哈希、目标映射与新 run_id；本次CamRoll按已批准D1实施，其他系统仍不扩展。
4. 需要恢复时区分客户端中断、资产失败、服务故障；分别对应评测续跑、资产 retry、项目 recover。不要把所有问题都处理成重启模型。
5. 新管线改变会影响 fingerprint，必须重新核对索引，不能把新旧向量或结果混用。

## 容易误判的地方

- `/api/status` 是全局计数；旧 demo 故意失败的文件不等于新数据集索引失败。
- `asset_id` 是内容哈希，不是文件名；别名去重后必须仍按规范资产评分。
- Top-K 命中一张图不等于多证据答案完整；要一起看 Recall 与全目标覆盖率。
- B/C 是预先固定的离线对照，不是系统自动 query router。默认使用 A 有本轮结果支持。
- 视频是短合成基准素材；人物关系、精确时间/GPS 等没有作为检索能力实现。
- 旧 API benchmark 和 alignment 会写根报告；PhotoBench 有独立脚本与目录。
- recover 会重启，status 脚本也经过管理器；当前禁止写文件时不要执行来“顺便检查”。

后续可单独立项验证真实视频、身份/元数据能力、更多事件与完整人工标注；这些属于未完成的新实验，不能写成已具备能力。

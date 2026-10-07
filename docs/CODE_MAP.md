# CamRoll–WeMM 源码阅读地图

2026-10-07。本文所有链接均相对仓库定位；总体数据流与当前实验状态见 [ARCHITECTURE_CURRENT.md](ARCHITECTURE_CURRENT.md)。仓库之外的权重、媒体、QA、数据库及逐题记录不是缺失源码。

## 推荐阅读顺序

1. [根README](../README.md)、[AGENTS.md](../AGENTS.md)、[RESTORE.md](../RESTORE.md)：范围、操作边界、恢复材料。
2. [当前架构](ARCHITECTURE_CURRENT.md)：区分冷建库四模型实验和共享WeMM快照的Flash Gate实验。
3. [pipeline.py](../camroll/camroll-agent/camroll_agent/pipeline.py) → [library.py](../camroll/camroll-agent/camroll_agent/library.py)：资产如何变成版本化记录、矩阵和FTS，查询如何读取。
4. [agent.py](../camroll/camroll-agent/camroll_agent/agent.py) → [schemas.py](../camroll/camroll-agent/camroll_agent/schemas.py) → [tools.py](../camroll/camroll-agent/camroll_agent/tools.py)：工具协议、状态机及看图。
5. [gate.py](../camroll/camroll-agent/camroll_agent/gate.py) → [evidence_gate.py](../camroll/camroll-agent/camroll_agent/evidence_gate.py) → [budget.py](../camroll/camroll-agent/camroll_agent/budget.py)：输出尝试、事实判定、请求身份和资金状态。
6. [gate_experiment.py](../camroll/camroll-agent/camroll_eval/gate_experiment.py) → [judge.py](../camroll/camroll-agent/camroll_eval/judge.py) → [repository.py](../camroll/camroll-agent/camroll_eval/repository.py)：完整计划、失败归类、评分、报告。
7. [four_gpu_window.py](../deployment/four_gpu_window.py) 及当前所用监督器：GPU互斥、租约、恢复与源文件保护。

## 目录与关键实现

### 相册核心：`camroll/camroll-agent/camroll_agent/`

| 文件 | 关键对象/函数 | 责任 |
|---|---|---|
| [cli.py](../camroll/camroll-agent/camroll_agent/cli.py) | `main` | inspect、建库/补充描述/图片编码、索引、问答、预算与服务CLI |
| [pipeline.py](../camroll/camroll-agent/camroll_agent/pipeline.py) | `build`、`enrich`、`encode_images`、`reindex`、`_publish` | 全部建库阶段、缓存、失败覆盖、原子发布 |
| [metadata.py](../camroll/camroll-agent/camroll_agent/metadata.py) | `extract_exif`、`OfflineGeocoder` | 时间/GPS来源、离线地点、可信过滤 |
| [identity.py](../camroll/camroll-agent/camroll_agent/identity.py) | `FaceEngine`、`seed_references`、`photo_faces`、`resolve_person` | 外置CPU模型、登记参考照匹配、身份不确定性及合法裁剪 |
| [library.py](../camroll/camroll-agent/camroll_agent/library.py) | `Library`、`search`、`_rank`、`_fuse`、`_rerank` | 固定快照、向量/指纹校验、过滤、M1–M4检索及六工具读操作 |
| [experiment.py](../camroll/camroll-agent/camroll_agent/experiment.py) | `ExperimentProfile`、`PROFILES` | Q37/Q38/W37/W38Max不可变冷建库配置 |
| [retrieval.py](../camroll/camroll-agent/camroll_agent/retrieval.py) | `make_retrieval_clients`、`validate_embedding`、`split_document` | 后端选择、角色/维度/指纹和文档分块 |
| [runtime_clients.py](../camroll/camroll-agent/camroll_agent/runtime_clients.py) | `build_clients` | 通用入口与已发布快照绑定 |
| [wemm.py](../camroll/camroll-agent/camroll_agent/wemm.py) | `EmbeddingClient` | 8765认证、原4096维内部编码、512token分块、deadline继承 |
| [qwen_retrieval.py](../camroll/camroll-agent/camroll_agent/qwen_retrieval.py) | `QwenRetrievalClient` | 18778回环调用、2048维图文编码/重排、隔离缓存及遥测 |
| [agent.py](../camroll/camroll-agent/camroll_agent/agent.py) | `Agent`、`AgentResult` | 问答状态机、题内历史、工具去重、最终复核和失败来源 |
| [schemas.py](../camroll/camroll-agent/camroll_agent/schemas.py) | `TOOLS`、`validate_arguments`、`validate_schema` | 六工具及结构化输出契约 |
| [tools.py](../camroll/camroll-agent/camroll_agent/tools.py) | `search`、`grep`、`list_by_date`、`get`、`view_image`、`query_photo_people` | Library到工具观察、VLM图片调用适配 |
| [prompts.py](../camroll/camroll-agent/camroll_agent/prompts.py) | `SYSTEM_PROMPT`、`format_observation`、`public_evidence` | 冻结Agent提示、工具观察/证据投影 |
| [gate.py](../camroll/camroll-agent/camroll_agent/gate.py) | `OutputGate`、`GatedVisionClient` | 最多4次输出尝试、原样重发核对、分类修复及候选审计 |
| [evidence_gate.py](../camroll/camroll-agent/camroll_agent/evidence_gate.py) | `EvidenceGateVerifier` | 无GT的Doubao事实判定、来源与引用校验、一次格式修复 |
| [qwen.py](../camroll/camroll-agent/camroll_agent/qwen.py) | `QwenClient`、`RequestScope`、`MODEL_LIMITS` | 云型号、上下文/请求/时间限额、流响应、实际型号、费用计量 |
| [budget.py](../camroll/camroll-agent/camroll_agent/budget.py) | `BudgetLedger`、`RequestJournal`、`UncertainRequestAcknowledgement` | 预留/结算、幂等回放、未知请求阻断及精确授权例外 |
| [service.py](../camroll/camroll-agent/camroll_agent/service.py) | `create_app` | `/api/ask`、会话/请求持久化、健康与请求状态 |

[index.py](../camroll/camroll-agent/camroll_agent/index.py) 是重建索引兼容入口；[build_memory.py](../camroll/camroll-agent/camroll_agent/build_memory.py)、[store.py](../camroll/camroll-agent/camroll_agent/store.py)、[vector.py](../camroll/camroll-agent/camroll_agent/vector.py) 和 [llm/](../camroll/camroll-agent/camroll_agent/llm/) 保留上游/旧接口。理解当前WeMM/Qwen实验时应优先跟踪 `pipeline → Library → retrieval`，不能仅凭旧类名判断运行路径。

### 评测与报告：`camroll/camroll-agent/camroll_eval/`

| 文件 | 责任 |
|---|---|
| [__main__.py](../camroll/camroll-agent/camroll_eval/__main__.py) | 独立评测CLI、服务、历史导入/回放和历史修复入口 |
| [dataset.py](../camroll/camroll-agent/camroll_eval/dataset.py) | 私人输入读取、题目/轮次结构、媒体快照及完整性校验 |
| [runner.py](../camroll/camroll-agent/camroll_eval/runner.py) | 基础建库/问答/评分调度，阶段记录和取消收尾 |
| [gate_experiment.py](../camroll/camroll-agent/camroll_eval/gate_experiment.py) | `IndexBinding`、`InferenceConfig`、`GateBackend`、`GateRunner`、`FATAL_CODES`及固定分母统计 |
| [gate_continuation.py](../camroll/camroll-agent/camroll_eval/gate_continuation.py) | `ContinuationParent`：不可变父run、继承尝试/评分/题内对话校验 |
| [judge.py](../camroll/camroll-agent/camroll_eval/judge.py) | 最终质量/行为Judge调用、费用journal及错误传播 |
| [judge_prompts.json](../camroll/camroll-agent/camroll_eval/judge_prompts.json)、[task_actions.json](../camroll/camroll-agent/camroll_eval/task_actions.json) | 冻结评分提示和动作定义，不是私人参考答案 |
| [compat.py](../camroll/camroll-agent/camroll_eval/compat.py)、[provenance.json](../camroll/camroll-agent/camroll_eval/provenance.json) | 原评分兼容结构与来源存证 |
| [metrics.py](../camroll/camroll-agent/camroll_eval/metrics.py)、[gate_metrics.py](../camroll/camroll-agent/camroll_eval/gate_metrics.py) | 质量/行为、固定81题交付、召回/重排、Gate尝试与失败指标 |
| [telemetry.py](../camroll/camroll-agent/camroll_eval/telemetry.py) | 实际采样资源与运行遥测 |
| [repository.py](../camroll/camroll-agent/camroll_eval/repository.py) | 只读结果投影、逐题轨迹、来源允许集合和比较边界 |
| [service.py](../camroll/camroll-agent/camroll_eval/service.py)、[static/](../camroll/camroll-agent/camroll_eval/static/) | API与报告页面；8774使用只读模式且不挂Agent |
| [repair.py](../camroll/camroll-agent/camroll_eval/repair.py)、[supplement.py](../camroll/camroll-agent/camroll_eval/supplement.py) | 历史独立修复/补测，不自动覆盖首跑结果 |
| [common.py](../camroll/camroll-agent/camroll_eval/common.py) | 安全ID、原子JSON、JSONL追加、哈希和时间 |

### WeMM原系统：`wemm/`

| 文件/目录 | 责任 |
|---|---|
| [backend/app.py](../wemm/backend/app.py) | 旧API、会话、上传/索引/检索和静态前端 |
| [backend/camroll_internal.py](../wemm/backend/camroll_internal.py) | CamRoll回环内部编码/tokenize，复用既有worker |
| [backend/worker.py](../wemm/backend/worker.py) | 单worker、编码队列与任务恢复 |
| [backend/encoder.py](../wemm/backend/encoder.py) | WeMM模型加载、向量与指纹 |
| [backend/preprocess.py](../wemm/backend/preprocess.py) | 原图片/文档/视频预处理 |
| [backend/retrieval.py](../wemm/backend/retrieval.py)、[backend/benchmark.py](../wemm/backend/benchmark.py) | 旧检索和原WeMM benchmark |
| [frontend/](../wemm/frontend/)、[scripts/](../wemm/scripts/) | 旧页面及既有运维/导入工具；不将旧打包脚本当新Git发布白名单 |
| [config.yaml](../wemm/config.yaml)、[SOURCE_LOCK.json](../wemm/SOURCE_LOCK.json)、[official-locked/](../wemm/official-locked/) | 部署配置、来源/revision/环境锁及官方代码；权重外置 |
| [model-code/](../wemm/model-code/README.md)、[SOURCE_MANIFEST.json](../wemm/model-code/SOURCE_MANIFEST.json) | 本版补齐外置模型目录原有的自定义Python、结构配置和模板；逐文件哈希固定，不修改实际加载路径 |

模型内部实现从 [modeling_st_wemm.py](../wemm/model-code/modeling_st_wemm.py) 和 [modeling_wemm_embedding.py](../wemm/model-code/modeling_wemm_embedding.py) 阅读；底层通用Transformer/PyTorch实现由外部依赖提供。Qwen的40个模型文件只提供 [公开来源清单](QWEN_MODEL_FILES.json)，权重不在仓库。详细安装材料与清单不可互换的约束见 [MODEL_DEPENDENCIES.md](MODEL_DEPENDENCIES.md)。

### 部署与独立GPU窗口：`deployment/`

| 入口 | 责任及阅读说明 |
|---|---|
| [eval_config.py](../deployment/eval_config.py)、[with_eval_config.py](../deployment/with_eval_config.py) | 加载受限配置，固定相册/结果路径与共享累计账本 |
| [manage_eval.py](../deployment/manage_eval.py)、[run_eval.py](../deployment/run_eval.py) | 8772进程管理；[运行说明](../deployment/README.md) |
| [manage_chat.py](../deployment/manage_chat.py)、[run_chat.py](../deployment/run_chat.py) | 8773独立页面/代理；[运行说明](../deployment/CHAT.md) |
| [qwen_worker.py](../deployment/qwen_worker.py) | 18778回环Qwen检索worker；串行单模型驻留、图文分块、rope状态修复 |
| [vendor/qwen_vl/](../deployment/vendor/qwen_vl/) | 固定Qwen Embedding/Reranker代码、许可及兼容补丁锁 |
| [four_gpu_window.py](../deployment/four_gpu_window.py) | 共用GPU1所有权、租约、watchdog、互斥换载及恢复 |
| [four_config.py](../deployment/four_config.py)、[four_suite.py](../deployment/four_suite.py) | 历史四模型独立冷建库套件；[运行说明](../deployment/FOUR_MODELS.md) |
| [four_cloud_smoke.py](../deployment/four_cloud_smoke.py)、[four_gpu_acceptance.py](../deployment/four_gpu_acceptance.py) | 云能力及真实本地GPU组件验收，真实执行需显式付费/运行参数 |
| [four_export.py](../deployment/four_export.py)、[four_report.py](../deployment/four_report.py) | 历史四组费用归属、导出和8774报告 |
| [four_supplement.py](../deployment/four_supplement.py)、[four_supplement_export.py](../deployment/four_supplement_export.py) | 已授权历史403补测及独立导出，当前Flash任务不调用Max |
| [gate_config.py](../deployment/gate_config.py)、[gate_validate.py](../deployment/gate_validate.py) | 当前固定路径/参数、离线验证与源文件清单 |
| [gate_cloud_smoke.py](../deployment/gate_cloud_smoke.py)、[gate_suite.py](../deployment/gate_suite.py) | 原Flash四组验收/监督器；[运行说明](../deployment/FLASH_GATE.md) |
| [gate_continuation_smoke.py](../deployment/gate_continuation_smoke.py)、[gate_continuation.py](../deployment/gate_continuation.py) | 精确授权未知请求保留后的独立续测；[运行说明](../deployment/GATE_CONTINUATION.md) |
| [gate_remaining.py](../deployment/gate_remaining.py) | 仅原先尚未开始的F38-On独立入口；[运行说明](../deployment/GATE_REMAINING_RUNBOOK.md) |
| [gate_export.py](../deployment/gate_export.py)、[gate_continuation_export.py](../deployment/gate_continuation_export.py)、[gate_remaining_export.py](../deployment/gate_remaining_export.py) | 各代私人报告/费用导出，保留来源与不完整状态 |
| [gate_report.py](../deployment/gate_report.py)、[gate_continuation_report.py](../deployment/gate_continuation_report.py)、[gate_remaining_report.py](../deployment/gate_remaining_report.py) | 各代只读8774入口；当前归档状态对应remaining入口 |

### 8773问答页面：`camroll/camroll-chat/`

[camroll_chat/config.py](../camroll/camroll-chat/camroll_chat/config.py) 固定上游和私人配置，[album.py](../camroll/camroll-chat/camroll_chat/album.py) 固定相册/快照及资产边界，[app.py](../camroll/camroll-chat/camroll_chat/app.py) 提供页面、图片和流式代理，[static/](../camroll/camroll-chat/static/) 显示答案、图片、实时步骤及用量。它不是8774比较报告，也不持有第二套预算账本。

## 关键调用链

| 要回答的问题 | 从哪里跟踪 |
|---|---|
| 原图怎样进入可查询相册？ | `pipeline.build → metadata/identity → QwenClient → EmbeddingClient → pipeline._publish → Library` |
| search为什么返回这张图？ | `tools.search → Library.search → _eligible → _rank/_fuse → _rerank → _combine`；查看私有retrieval_trace |
| Qwen检索怎样不污染WeMM？ | `make_retrieval_clients → QwenRetrievalClient → qwen_worker.Runtime`；进程互斥由`GpuWindow`负责 |
| Agent如何看图并交付？ | `Agent → schemas/tools.view_image → QwenClient.generate_json → 最终复核 → 引用/交付校验` |
| Gate重试是否真是原样发送？ | `OutputGate.run → QwenClient.chat → RequestJournal`；不同semantic identity、相同完整payload哈希 |
| 事实Gate是否用了答案？ | `Agent证据装配 → EvidenceGateVerifier.verify`；该模块不依赖dataset/finalJudge，并拒绝GT字段 |
| 为什么超限会停整组？ | `QwenClient._preflight → ProviderError(context_limit) → GateRunner.infrastructure_failure/FATAL_CODES` |
| 旧失败会不会被成功续测覆盖？ | `ContinuationParent → 新run继承来源 → gate_summary/Repository`；父run/原始尝试/评分SHA校验 |
| 一次请求怎样收费？ | `RequestJournal.claim → BudgetLedger.reserve → 发送 → journal完成/未知 → BudgetLedger.finish` |
| 中断后谁清GPU？ | `监督器 → GpuWindow/register_child → keepalive/watchdog → restore`，进入/退出哈希及进程身份核验 |
| 8774能否产生付费调用？ | `gate_remaining_report → create_app(read_only=True,mount_agent=False)`，写请求被中间件拒绝 |

## 测试入口与可运行边界

| 测试位置 | 主要覆盖 |
|---|---|
| [camroll/tests/](../camroll/tests/) | Agent、Qwen、预算、WeMM API、服务、部署、检索客户端及实验入口 |
| [camroll-agent/tests/](../camroll/camroll-agent/tests/) | Library、建库、元数据身份、Gate/事实证据、Judge journal、续测、固定指标与报告 |
| [deployment/test_four_gpu_window.py](../deployment/test_four_gpu_window.py) | 原服务已停/运行、冲突、租约、进程身份与恢复 |
| [deployment/test_gate_continuation_supervisor.py](../deployment/test_gate_continuation_supervisor.py)、[test_gate_remaining.py](../deployment/test_gate_remaining.py) | 父来源冻结、未知请求例外、独立窗口及失败收尾 |
| [deployment/test_qwen_worker.py](../deployment/test_qwen_worker.py) | worker输入/分块/模型状态及接口；真实GPU验收另走受控入口 |
| [camroll-chat/tests/](../camroll/camroll-chat/tests/) | 8773配置、资产读取、同源代理和流式接口 |

离线运行应使用独立Python环境与已有依赖锁。主回归命令的来源是 [gate_validate.py](../deployment/gate_validate.py)，其测试目录为 `camroll/camroll-agent/tests camroll/tests deployment`，并设置仓库、Agent和deployment的模块路径。源码测试不会替代真实模型能力、权重哈希或旧服务恢复验收。

真实部署还需私有配置、原累计账本、相册/QA、指定GPU与外置权重。部分监督器刻意绑定原服务器和冻结输入，未宣称能在空白云容器中直接运行正式benchmark。云端代码阅读者应先静态阅读和运行合成/模拟测试，不能用缺少运行数据解释为架构不完整。

## 历史与来源入口

- [baseline/source_manifest.json](../baseline/source_manifest.json)：2026-09-16基线采集来源和逐文件哈希；只用于其对应基线，不要求后续修改后的当前源码仍等于旧哈希。
- [baseline/environment/](../baseline/environment/)：基线Python和依赖存证，未包含虚拟环境。
- [PHOTOBENCH_PROTOCOL.md](PHOTOBENCH_PROTOCOL.md)：早期独立PhotoBench协议。
- [PHOTOBENCH_QWEN_PROTOCOL_V1.md](PHOTOBENCH_QWEN_PROTOCOL_V1.md)：历史Qwen/WeMM四模型协议。
- [docs/project/README.md](project/README.md)：原始交接索引，状态与配置均按其日期理解。
- [THIRD_PARTY.md](../THIRD_PARTY.md)、[CamRoll许可证](../camroll/LICENSE)、[WeMM官方许可证](../wemm/official-locked/LICENSE)、[Qwen vendor许可证](../deployment/vendor/qwen_vl/LICENSE)：保留各自来源与许可，不用本次架构文档重新授权第三方代码。

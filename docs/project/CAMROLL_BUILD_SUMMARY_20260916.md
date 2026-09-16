# CamRoll → Qwen3.7-Flash + WeMM：搭建与改动总结

整理日期：**2026-09-16**。实现与验收依据为本地保存的 **2026-09-15 23:13（北京时间）**记录，源码交付核对记录截至当日 23:16。本文不代表服务器当前健康状态。

## 1. 当前边界与交付结论

**用户最新要求：不修改 `.100` 和 `.153` 的任何文件。本次整理只读本地资料，只修改 `/Users/zhanyiqi/Documents/camwwem/docs`。未连接服务器，未调用远端接口或计费模型。** 此要求覆盖此前实施阶段的服务器操作授权；历史文档中的命令不代表当前允许执行。

此前已完成：在 `.100:/home/realmagic/camroll-wemm` 独立部署 CamRoll，全部生成角色替换为 Qwen3.7-Flash，检索复用原 WeMM-Embedding-9B 单 worker。保留五工具、Python/CLI 问答，新增 HTTP/SSE 接口，完成增量建库、有限纠错、预算账本和四配置开关。没有下载模型权重，也没有加载第二份 WeMM。

默认保留 **M1（原图检索＋按需看图）**。334 张照片、800 道冻结图片问题的检索对照已完成；图文双路没有带来总体收益。**最终回答仅做少量接通及证据传递检查，没有批量评测答案正确率。**

上游：`https://github.com/thaoshibe/camroll`，基线提交 `1967f977dbd51ce3f3b3b0d3213d21a514e55dfd`。本地源码改动尚未提交或推送到 GitHub。

## 2. 项目、数据和部署路径

| 用途 | 精确位置 |
|---|---|
| 本地交接工作区（不是线上源码根目录） | `/Users/zhanyiqi/Documents/camwwem` |
| 本地改造源码 | `/Users/zhanyiqi/Documents/camwwem/camroll` |
| 本地旧 WeMM 源码与冻结报告副本 | `/Users/zhanyiqi/Documents/nx/wemm-embedding-lab` |
| `.100` CamRoll 项目 | `/home/realmagic/camroll-wemm` |
| `.100` 独立 Python | `/home/realmagic/camroll-wemm/.venv/bin/python` |
| `.100` 既有 WeMM 项目 | `/home/realmagic/wemm-embedding-lab` |
| `.100` 本轮使用的冻结照片副本 | `/home/realmagic/wemm-embedding-lab/import/photobench-20260914/album3` |
| `.153` 原始资料（只读） | `/home/asus/Github/Sentrix-Home-Web/services/photobench/data/album3-max-video10/` |
| `.100` 旧冻结报告 | `/home/realmagic/wemm-embedding-lab/reports/photobench/20260914-115425/` |
| 本地本轮报告 | `/Users/zhanyiqi/Documents/camwwem/camroll/reports/20260915-221637/` |
| `.100` 本轮报告 | `/home/realmagic/camroll-wemm/reports/20260915-221637/` |

本轮实现使用 `.100` 已有冻结照片副本，未修改 `.153` 原始资料。WeMM 固定物理 GPU 1，UUID `GPU-4ef9200c-d8f3-638a-3eeb-26fe4fa3d7b0`；逻辑 `cuda:0` 是可见卡映射。GPU 0 不受本项目控制。

## 3. 架构与模型职责

```text
照片清单 → SHA256资产登记、来源/日期保存、EXIF方向处理、派生图
          ├─ WeMM原图编码 ────────────────────┐
          └─ Qwen中文caption → 时间顺序事件整理 → WeMM文本分块编码
                                             ↓
                         SQLite状态 + 不可变向量/关键词索引快照
                                             ↓
用户问题 → Qwen原生tool_calls → 校验/纠错 → 五工具 → 新证据 → 下一轮
                                             ↓
                     V0：根据工具文字证据回答
                     V1：最多6张证据图再次交给Qwen生成最终答案
```

- Qwen 请求模型固定 `qwen3.7-flash`，默认 `enable_thinking=False`；统一 caption、事件、工具选择、看图与最终回答，不自动切换其他模型。
- WeMM 维持 BF16、4096 维归一化向量、既有预处理及像素预算。CamRoll 独立进程访问云端，避免与 WeMM 禁外网保护冲突。
- 原版主循环接收 `view_image` 的文字分析，不会自动再次看到原图；V1 明确补充最终图片请求。
- 日期只来自清单或 EXIF，缺失记未知；不让模型猜日期。没有用户参考照时不推断相册主人。
- 路径、照片 ID、事件 ID、成员关系由程序维护；事件重命名不改变关联。

## 4. 源码改动位置

以下相对路径同时对应两个根目录：本地 `/Users/zhanyiqi/Documents/camwwem/camroll/`，服务器 `/home/realmagic/camroll-wemm/`。

| 路径 | 职责与改动 |
|---|---|
| `camroll-agent/camroll_agent/qwen.py` | Qwen 统一适配：图片、原生工具调用、结构化输出、usage、网络重试、云端流式支持 |
| `camroll-agent/camroll_agent/budget.py` | SQLite 原子预留与结算，跨并发请求和重试共享 300 元额度 |
| `camroll-agent/camroll_agent/pipeline.py` | 清单、去重、元数据、派生图、阶段缓存、事件整理、增量建库和发布 |
| `camroll-agent/camroll_agent/library.py` | 固定快照检索、日期预过滤、原图/双路、中文 FTS5、ID 获取 |
| `camroll-agent/camroll_agent/wemm.py` | 内部接口客户端、认证、分块、向量及指纹校验 |
| `camroll-agent/camroll_agent/agent.py` | 普通/流式共用执行循环、限额、纠错、去重和最终图片复核 |
| `camroll-agent/camroll_agent/schemas.py` | 工具参数类型、范围等校验 |
| `camroll-agent/camroll_agent/tools.py` | 五工具执行及 Qwen 看图 |
| `camroll-agent/camroll_agent/prompts.py` | 基于证据的提示词、路径信息清理 |
| `camroll-agent/camroll_agent/service.py` | FastAPI 问答、健康检查及 SSE 入口 |
| `camroll-agent/camroll_agent/cli.py` | 建库、续跑、发布、问答与账本 CLI |
| `camroll-agent/camroll_agent/build_memory.py`、`index.py` | 兼容入口 |
| `deployment/wemm_internal.py` | 安装到既有 WeMM 的内部 API 模块模板 |
| `deployment/install_wemm_api.py` | 对照原始源码哈希、备份并安装已知 API 补丁 |
| `deployment/snapshot_wemm.py`、`restart_wemm.py` | 旧库保护记录与受控重启脚本；当前禁止执行 |
| `deployment/with_config.py`、`run_service.py`、`manage_camroll.py` | 受限配置加载、回环地址启动、仅管理 CamRoll 自身进程 |
| `deployment/prepare_photobench.py` | 从冻结副本生成白名单建库输入，排除 QA/答案/人工事件 |
| `deployment/probe_qwen.py`、`probe_internal_live.py`、`probe_agent.py`、`probe_http.py` | 真实能力、内部编码、四配置与 HTTP 验证 |
| `deployment/export_receipt.py` | 导出脱敏用量及费用记录 |
| `scripts/evaluate_retrieval.py` | 800 题冻结检索对照、逐题差异与分组 bootstrap |
| `tests/`、`camroll-agent/tests/` | 组件与回归验证 |
| `pyproject.toml`、`requirements-wemm.txt` | 运行依赖；CamRoll 不自行加载检索模型 |

既有 WeMM 服务源码只改 API 层：

- `/home/realmagic/wemm-embedding-lab/backend/app.py`：注册内部接口。
- `/home/realmagic/wemm-embedding-lab/backend/camroll_internal.py`：新增内部接口模块。
- `encoder.py`、`worker.py`、`preprocess.py`、`config.yaml` 保持不变。

注意：本地 `deployment/install_wemm_api.py` 已支持在原始版本和精确已知补丁状态之间受哈希保护地更新；旧部署说明中“不能重复执行”的描述偏保守，不能代替实际脚本检查。当前禁止运行此脚本。旧后端文件仍有保留，不能理解为所有历史实现均已删除。

## 5. 接口位置与请求约定

### 5.1 CamRoll 问答 HTTP

历史监听位置：**`.100` 本机 `http://127.0.0.1:8770`**。入口源码：`/home/realmagic/camroll-wemm/camroll-agent/camroll_agent/service.py`；启动脚本：`/home/realmagic/camroll-wemm/deployment/run_service.py`。未建设前端或账号系统，不直接对局域网开放。

| 方法与路径 | 输入 | 返回/行为 |
|---|---|---|
| `GET /api/health` | 无 | `status`、`configs`、已发布相册列表；不证明模型可用或答案正确 |
| `POST /api/ask` | `album_id`、`question`、`config`（默认 M1）、`stream`（默认 false） | 普通 JSON 或 SSE；会调用模型并写入费用账本，不是只读检查 |

请求示例（仅展示协议，本次未执行）：

```json
{"album_id":"photobench-album3","question":"照片中有哪些动物？","config":"M1","stream":false}
```

普通响应包含 `answer`（`final_text` 的别名）、`final_text`、`album_id`、`tool_trace`、`steps`、`view_image_calls`、`stopped_reason`、`latency_s`、`evidence_image_ids`、`index_version`、`config`、`usage`。

相册 ID 只允许安全标识符；问题非空且不超过 20000 字符；未发布相册返回 404，非法请求返回 422。`stream=true` 返回 `text/event-stream`，事件包括 `status`、`tool_call`、`tool_result`、`answer`，异常可返回 `error`。**这是执行进度和工具事件流，不是逐 token 的答案文本流。**

### 5.2 WeMM 内部 API

CamRoll 使用 `http://127.0.0.1:8765`。既有 WeMM 服务仍保留原监听配置；新增内部路由额外限制本机访问，并沿用 `wemm_session` 认证。

| 方法与路径 | 输入 | 输出 |
|---|---|---|
| `POST /api/internal/embeddings` | 二选一：`{"text":"查询或文本块"}` 或 `{"image_path":"程序管理的图片绝对路径"}` | `embedding`（4096 维）、`dimension`、`fingerprint`、`timings` |
| `POST /api/internal/tokenize` | `{"text":"待分块文本"}` | `tokens`、无损 `chunks`、`fingerprint`、`limit`（512） |

图片仅允许 `/home/realmagic/camroll-wemm/runtime` 受限根目录内的合规文件，实际使用相册资产派生图；只接受 JPG/JPEG/PNG/WEBP，限制文件大小。磁盘路径由程序映射，模型不能自行提供。编码文本超过 512 token 明确拒绝；建库文本经分块后编码，查询超限交给 Agent 缩短，不能静默截断。接口调用现有 `worker.request()`，不加载第二份模型。

既有 `/api/login`、`/api/status`、`/api/search` 仍属 WeMM 原服务；尤其 `/api/search` 会写 SQLite 事件，不能在当前边界下作为“只读验证”。

### 5.3 Python、CLI 与五工具

- Python `Agent.ask()`、`Agent.ask_streaming()`：`camroll-agent/camroll_agent/agent.py`。
- CLI `ask`、建库/续跑/索引管理：`camroll-agent/camroll_agent/cli.py`。`enrich` 可单独完成云端预处理，`embed-images` 可单独编码原图，随后 `build` 汇合缓存并发布；`index` 处理已有发布数据。
- `search / grep / list_by_date / get / view_image` 是 **Agent 工具名，不是五个新增 HTTP 路由**。注册与执行在 `tools.py`，检索在 `library.py`。
- `get`、`view_image` 只通过服务器资产 ID 映射；关键词零结果只表示索引文本未匹配，不能据此断言照片内容不存在。

## 6. 运行配置、数据库与索引

所有下列服务器路径均是历史部署位置，本次未访问：

| 内容 | `.100` 路径 |
|---|---|
| 受限运行配置（0600） | `/home/realmagic/camroll-wemm/runtime/secrets.json` |
| 费用账本 | `/home/realmagic/camroll-wemm/runtime/budget.sqlite` |
| 相册输入清单 | `/home/realmagic/camroll-wemm/runtime/photobench-album3.json` |
| 相册根目录 | `/home/realmagic/camroll-wemm/runtime/albums/photobench-album3` |
| 服务记录 / 日志 | `/home/realmagic/camroll-wemm/runtime/camroll-service.json`、`camroll-service.log` |
| 既有 WeMM 认证文件 | `/home/realmagic/wemm-embedding-lab/run/access.key`（不复制内容） |
| 既有 WeMM 数据库 | `/home/realmagic/wemm-embedding-lab/data/lab.sqlite` |

配置变量包括 `DASHSCOPE_API_KEY`、`QWEN_BASE_URL`（北京授权业务空间兼容入口）、`CAMROLL_BUDGET_DB`、`CAMROLL_BUDGET_CNY`、`CAMROLL_ALBUM_ROOT`、`WEMM_BASE_URL`、`WEMM_KEY_PATH`。密钥不进入源码、本文或调用日志。

相册目录主要结构：

```text
photobench-album3/
  state.sqlite                 阶段状态、来源及运行记录
  source_spec.json              规范化输入
  assets/<SHA256>/              原图、Qwen派生图、*.wemm.jpg
  enriched_records.json         caption与事件结果
  last_enrich.json              预处理阶段记录
  last_image_encoding.json      图片编码阶段记录
  snapshots/<version>/
    records.json               照片/事件元数据
    vectors.npz                向量矩阵
    search.sqlite              中文关键词索引
  CURRENT                      原子切换的已发布版本
```

已验收索引：`34c35f4d5a9ff16a506eecfc6e6b66d2`。预处理版本 `exif-rgb-view95-wemm2048-jpegdefault-v2`，caption 版本 `zh-visible-evidence-v1`，事件版本 `stable-anchor-three-lookback-v1`，索引版本 `wemm-rrf60-jieba0.42.1-v1`。

同内容照片复用编码，但保留来源路径、文件名和成员关系。缓存包含资产、模型、提示词和预处理版本。单次查询固定不可变快照，发布只切换 `CURRENT`；活动请求不会混用新旧向量。Caption 失败不写入伪描述，原图仍可被检索。

## 7. 检索、纠错与预算实现

- 原图路：WeMM 查询向量直接比较图片向量；事件按成员最高分聚合。
- 双路：图片与 caption 各 Top-50，按照片 ID 去重，以等权 `1/(60+rank)` 融合；事件融合成员聚合排名和事件文本排名。文本多块取最高分。
- 日期条件先筛选候选；`kind=both` 保留类型并交错输出，照片优先，稳定 ID 决定同分顺序。评测固定照片类型。
- `grep` 使用固定 jieba 0.42.1 分词及 SQLite FTS5，caption 可见文字进入索引。
- 普通与流式共用状态机：消息协议 → JSON → 字段类型/范围 → ID/日期/存在性 → 执行 → 明确成功或错误反馈。坏参数不执行，也不替换为空对象。
- Caption 最多 3 次结构化生成尝试；网络层每次请求最多 3 次尝试。429、超时及暂时服务错误退避；认证、权限、模型不存在等永久错误直接停止。结构化输出只有明确不支持时才降级 JSON Object 并本地校验。
- 每题默认 25 决策轮、50 实际云请求、5 次看图、180 秒；工具参数错误最多 2 次纠正机会。成功重复工具调用复用，失败重复受控；连续无新证据先要求换策略，再终止。
- V1 最多一次额外最终复核，最多 6 张去重图片，优先已引用证据。引用必须来自实际取得、传入的证据，不能编造照片 ID。
- 300 元共享账本发请求前原子预留，全部重试计费，接近上限停止新请求并保留进度，不自动开启新预算。

## 8. 配置矩阵与部署备选

| 配置 | 检索 | 最终回答 | 本轮状态 |
|---|---|---|---|
| M1（默认） | 原图单路 | V0 按需看图，基于工具文字回答 | 检索对照＋少量问答验证 |
| M2 | 原图单路 | V1 最终证据图复核 | 少量问答及图片传递验证 |
| M3 | 原图＋caption 双路 | V0 | 检索对照＋少量问答验证 |
| M4 | 原图＋caption 双路 | V1 | 少量问答及图片传递验证 |

| 部署方案 | 取舍 | 状态 |
|---|---|---|
| D1 扩展既有 API、共用单 worker | 支持直接编码和自主索引；需 API 修改及重启 | 已实施 |
| D2 仅用原数据集导入和 search API | 无需改 WeMM 源码；双路/筛选/索引控制受限 | 备选，未实施 |
| D3 独立服务复用权重、与旧服务互斥运行 | 隔离更完整；同卡两服务不能同时加载 | 备选，未实施 |

P0 模型替换与可靠性、P1 增量/缓存/中文关键词/成本记录已实施；P2 双路与最终图片复核为开关。P3 局部裁剪、多跳、联合向量及 P4 ANN、降维、批量编码尚未实施；没有新增人脸识别、OCR 或重排模型。

## 9. 验证结果及答案质量边界

### 建库与接入

334 张照片全部完成原图编码；333 张有效 caption，文本覆盖率 **99.70%**；生成 147 个事件。1 张照片连续两次 HTTP 400，保留失败状态及图片向量。原图编码约 112.67 秒；后续发布复用成功阶段缓存。

Mac 与 `.100` 的最终组件验证均 **175/175 通过**。覆盖坏 JSON、类型/缺字段/ID/日期错误、工具错误与零结果区分、重试、重复调用、续跑、稳定关联、日期过滤、RRF、分块、快照和最终图片请求。实际北京业务空间验证通过文本、图片、JSON Schema、原生 Function Calling 往返和云端流式能力；四配置及 HTTP 普通/SSE 验证通过。

首轮 M4 曾因本地引用校验过严失败：正文举例一张，证据列表包含五张实际已传入图片，被集合相等规则误拒绝。现改为正文引用必须属于已声明且已复核证据，保留原失败和复现记录，修复后重新验证。建库期间 WeMM 曾被外部任务停止；云端预处理独立保存进度，待资源持有方恢复后续跑，未控制外部进程。

### 冻结检索对照

run_id：`20260915T150213Z-74ae8a76`。334 张照片、800 道图片主测问题，两路共用原问题与评分器。建库没有使用 QA 答案、目标标签或人工事件。RRF 固定不针对本集调参；失败固定分母计零。

| 指标 | M1 原图 | M3 双路 |
|---|---:|---:|
| Hit@1 | 80.250% | 77.500% |
| Hit@5 | 95.625% | 93.750% |
| Hit@10 | 97.625% | 96.875% |
| Hit@20 | 99.000% | 98.500% |
| Recall@10 | 94.639% | 93.235% |
| Recall@20 | 97.034% | 96.456% |
| 全目标覆盖@10 | 91.000% | 88.625% |
| 全目标覆盖@20 | 94.625% | 94.000% |
| MRR@50 | 0.868248 | 0.846068 |
| 失败率 | 0% | 0% |

按 225 个冻结事件做 2000 次配对 bootstrap；不是按新生成的 147 个事件。M3−M1 的 Hit@1 95% 区间为 [−6.188, +0.385] 个百分点，跨零。Hit@5、Recall@10、全覆盖@10、MRR 单项区间低于零，但没有多重比较校正。保留 M1 默认，不宣称所有差异均显著。

查询编码 P50 210.79 ms；本地排名 P50：M1 1.79 ms、M3 2.45 ms，**不是端到端问答耗时**。M1 全部逐题指标与历史 WeMM 原问题 A 组一致。本次未重新对照原版 `text-embedding-3-small`，不能量化声称超过原版 CamRoll。

### 检索与答案的关系

检索决定可用证据，答案还取决于 Agent 是否获取足够证据、Qwen 是否看懂图片、是否正确整合并限制结论。Hit@10 97.625% 只说明前十张至少命中一个已标注目标，**不等于答案准确率 97.625%**。多证据问题更需要看全目标覆盖；Agent 多轮累积证据也不等同于单次查询指标。

目前可确认“功能接通、组件验证通过、冻结集检索表现已测”；无法给出整体答案准确率，不能宣称 M2/M4 优于 M1/M3。没有开展批量最终答案评分，符合本轮范围。唯一缺 caption 图片不在 800 题标注目标中，这一缺失场景的质量也未被该对照验证。

## 10. 旧库保护与云端费用

历史验收核对：旧 542 个资产状态（540 indexed、2 failed）、585 个向量文件、900 个报告文件及受保护模型源码保持不变。既有指纹：

`cb0b50fe0c768df7ec200dcc15160643ce11660e0ffea39e48eafc451e6a4cf7`

334 张新 WeMM 派生图与冻结库逐字节一致；新旧图片向量最大浮点差约 `1.49e-8`。源码交付清单核对 108 个文件，无不匹配。这些是历史对照证据，本次没有远端复核。

截至 2026-09-15 23:12:46，账本记录 738 次实际云请求尝试，输入 1,783,658、输出 185,406 token；按当日核实的官方价格估算 **0.505353 元**，预留与不确定请求均为零。300 元额度剩余 **299.494647 元**，这是本轮预算余量，不是阿里云账户余额或正式账单。未假定缓存折扣和免费额度。

## 11. 证据文件与交接入口

下列链接均指向本地已有文件，可离线阅读：

- [实现与升级矩阵](../camroll/docs/IMPLEMENTATION.md)、[部署与恢复历史说明](../camroll/docs/DEPLOYMENT.md)、[Qwen 运行说明](../camroll/docs/QWEN_RUNTIME.md)、[本轮验收总表](../camroll/docs/VALIDATION_20260915.md)。
- [检索报告](../camroll/reports/20260915-221637/retrieval/REPORT.md)、[差异分析](../camroll/reports/20260915-221637/retrieval/ANALYSIS.md)、[逐题结果 CSV](../camroll/reports/20260915-221637/retrieval/per_question.csv)。
- [Qwen 能力实测](../camroll/reports/20260915-221637/qwen_capabilities.json)、[WeMM 内部接口实测](../camroll/reports/20260915-221637/wemm_api_probe.json)。
- [四配置最终检查](../camroll/reports/20260915-221637/agent_probe_v2/summary.json)、[HTTP/SSE 最终检查](../camroll/reports/20260915-221637/http_probe_v2/summary.json)。
- [组件验证](../camroll/reports/20260915-221637/components_final.xml)、[Mac 组件验证](../camroll/reports/20260915-221637/components-local-final.xml)。
- [最终验收状态](../camroll/reports/20260915-221637/final_acceptance.json)、[源码交付核对](../camroll/reports/20260915-221637/source_delivery_verified.json)、[派生图一致性](../camroll/reports/20260915-221637/preprocess_parity.json)、[向量数值核对](../camroll/reports/20260915-221637/image_vector_parity.json)。
- [费用汇总](../camroll/reports/20260915-221637/usage_summary.json)、[逐请求费用记录](../camroll/reports/20260915-221637/usage_calls.jsonl)。

报告目录含私有相册证据，默认不进入 Git。配置和认证文件只记录路径，不记录内容。

## 12. 后续操作边界

当前停留在本地文档交接。不得因阅读历史运维说明而执行上传、建库、评测、问答、管理脚本、启动/停止/重启或远端接口探测。后续用户明确重新授权服务器操作时，再核实实时进程、资源与哈希，使用新 run_id，保持旧冻结报告，继续使用原 300 元账本。

恢复方案入口是 `camroll/docs/DEPLOYMENT.md`，原 API 备份位于本轮报告的 `wemm_app.py.before`。恢复涉及 API 文件及进程变化，不能在当前边界下执行；未知源码状态不能直接覆盖。`.153` 原始资料继续只读，GPU 0 不属于本项目。

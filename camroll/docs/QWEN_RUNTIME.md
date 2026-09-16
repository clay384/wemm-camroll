# Qwen3.7-Flash 运行与费用记录

实现日期：2026-09-15。所有生成角色固定使用 `qwen3.7-flash`，请求明确设置 `enable_thinking=False`、`temperature=0`。本文件不包含账户密钥、业务空间地址或相册内容。

## 配置与调用

| 配置 | 行为 |
|---|---|
| `DASHSCOPE_API_KEY` | 从受限运行环境读取；真实调用必填，不写入源码或日志 |
| `QWEN_BASE_URL` | 填交接指南指定的北京业务空间兼容接口；必填，没有公共端点回退 |
| `CAMROLL_BUDGET_DB` | 所有进程和角色共享的 SQLite 账本绝对路径；默认 `~/.local/state/camroll/budget.sqlite3` |
| `CAMROLL_BUDGET_CNY` | 可进一步降低预算；默认 300 元，不能提高本轮已保存的额度 |

`QwenClient.chat(messages, tools=None, role="agent", ...)` 返回原生 assistant 消息和工具调用。附加 `_usage`、`_finish_reason` 用于本地观测，重新发送消息时应去掉这两个字段。

`generate_json(prompt, image_paths, schema, role="caption", max_attempts=3)` 返回通过本地 JSON Schema 校验的对象。`generate(prompt, image_paths)` 保留旧 VLM 调用入口。`image_messages()` 读取程序已解析的资产路径，执行 EXIF 方向纠正、RGB 转换和长边不超过 2048 的 JPEG 编码；原图不会被覆盖。多图按传入顺序附带编号，调用方在提示词中维护图片 ID 对应关系。

`chat(..., stream=True)` 调用云端原生流式接口并合并文本、分段工具参数和末尾 usage，返回相同消息结构。问答服务的 SSE 工具事件与云端 token 流是两个不同接口。

## 300 元预算与费用含义

每次实际 HTTP 尝试之前，账本通过 SQLite `BEGIN IMMEDIATE` 原子预留费用。并发 caption、服务问答和重试必须使用同一个账本；重启不会清零费用，重新打开账本不能提高已经保存的上限。

预留采用完整 1,048,576-token 上下文与最高输入单价，再加本次最大输出的最高单价。默认输出上限 2048 token 的单次预留约 **1.268122 元**。这是用于控制最坏情况的临时占用，**不是每次请求实际花费**。客户端另外限制文本与图片的输入上界、图片尺寸及最多 32768 的输出上限。

有完整 usage 时，按输入 token 数选择以下实现报价档位；图片 token 已包含在 usage 中，不重复计费，也不假定缓存优惠：

| 输入长度 | 输入元/百万 token | 输出元/百万 token |
|---|---:|---:|
| ≤ 32768 | 0.2 | 0.8 |
| ≤ 262144 | 0.6 | 2.4 |
| 更长 | 1.2 | 4.8 |

价格版本为 `qwen3.7-flash-cn-2026-09-15`。执行方应核实该业务空间适用的官方计费；账本是按 usage 计算的保守估算，阿里云账户账单才是最终支付记录。

- `estimated_spent_cny`：有 usage 的请求按报价计算的费用估算。
- `reserved_cny`：仍在进行、结果不确定或缺失 usage 的请求占用。
- `remaining_cny`：额度减去以上两者后还能预留的金额。
- `uncertain_requests`：网络中断、5xx 或缺失 usage 等不能确定最终费用的请求数。
- `usage`：汇总输入、输出与总 token 数。
- `roles`：按 `caption`、`event`、`agent`、`view`、`final`、`probe` 等角色汇总请求次数、费用估算及预留。

明确被拒绝的 400/401/403/404/422/429 请求释放预留；超时、连接中断、5xx 和缺失 usage 保留全部预留。不能把不确定请求当成免费请求。只有取得可核实的使用记录后才能对相应调用进行账本结算。不要删除账本、更换账本路径或新建额度以自动继续已耗尽的本轮预算。

账本仅存角色、模型、提示词版本、生成参数、usage、耗时和错误代码等白名单元数据；不存提示词、图片 base64、工具内容或凭据。`usage_summary()` 返回汇总，`capabilities()` 返回该端点与模型的结构化输出能力记录。

## 纠错、重试与停止

| 场景 | 行为 |
|---|---|
| 429、超时、连接中断、5xx | 一次 `chat` 最多 3 次实际尝试；首次与第二次重试前分别退避 0.5 秒、1 秒 |
| 认证、权限、模型不存在等永久错误 | 立即停止，不重试，不暴露 SDK 原始异常中的数据 |
| 坏 JSON、`null`、非对象、字段或类型不符合 schema | 将失败字段路径和校验规则反馈给模型；默认最多 3 次结构生成尝试 |
| 明确返回不支持 `json_schema` | 持久化该端点/模型能力为不支持，固定使用 `json_object` 加本地完整 schema 校验 |
| schema 本身无效，或普通 400 错误 | 不误判为能力缺失，不自动回退 |
| 预算不足 | 请求发出前抛出 `BudgetExceeded`；已完成工作由调用方保存 |
| 本题请求次数或时间达到上限 | 分别抛出 `RequestLimitExceeded`、`DeadlineExceeded` |

结构生成尝试与传输尝试独立计数：一般最多 3 × 3 = 9 次实际请求；首次探测出不支持 JSON Schema 时，显式拒绝分支最多额外 3 次，因此最坏上限为 12 次。所有尝试都进入共享账本；有问答 scope 时同时受每题总请求上限约束。最终图片复核使用 `max_attempts=1`，传输故障仍受上述最多 3 次尝试约束。

`request_scope(max_requests=50, deadline_s=180)` 将 Agent、看图和最终回答的实际请求数及重试合并计算。嵌套的 Qwen 客户端复用当前 scope。`scope.summary()` 返回本题请求数、耗时、usage、费用估算和不确定占用。scope 通过 `ContextVar` 传递；跨线程或分次恢复的同步生成器必须显式保持同一上下文，不能假定线程池自动延续上下文。

## 验证与接入探针

组件验证使用假 SDK，不产生云端费用：

```bash
PYTHONPATH=camroll-agent .venv/bin/python -m pytest -q tests/test_budget.py tests/test_qwen.py
```

覆盖线程和独立进程并发预留、账本重开、未知费用保留、重试与永久错误、坏 JSON/错误类型/空结果、Schema 回退、图片编码、流式工具参数合并，以及 scope 限制。

`deployment/probe_qwen.py` 是实际计费的小样本接入检查，需由执行方加载受限环境配置并指定本次独立输出目录。其每个请求同样计入共享账本；检查结果不能代替照片检索质量或最终回答质量评测。

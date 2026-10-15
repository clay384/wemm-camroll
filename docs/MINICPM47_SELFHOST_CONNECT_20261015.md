# 自建 MiniCPM Q6 Thinking 连接验证（2026-10-15）

## 当前状态

2026-10-15 09:59–10:00（Asia/Shanghai）已按用户提供的部署说明恢复SSH隧道并完成三项真实调用。服务器为 `zhanyiqi@ih484318ke39.vicp.fun:59922`；主机ED25519指纹与既有known_hosts一致。凭据仅用于交互认证，不保存到脚本、运行记录或文档。

本机API：`http://127.0.0.1:18080/v1`，转发到该服务器的 `127.0.0.1:8080`。模型名：`minicpm-v4.7-35b-a3b-q6-thinking`。隧道保持后台运行，可通过原部署目录的 `tunnel.sh status` 检查；断开后用 `tunnel.sh start` 恢复。

部署工具：`/Users/zhanyiqi/.codex/.chatgpt-projects/g-p-6acf2c724600819185c868fa1be8417f/deployment-tools/`。

该服务器与原WeMM服务器 `.100` 是不同机器。这里的GPU0为部署中的RTX5880，不代表允许操作 `.100` 的GPU0。本轮没有重启模型或更改服务器启动参数。

## 实际验证

健康、模型列表、配置及槽位接口均HTTP200。服务报告Q6_K、262144上下文、单槽位、视觉支持和llama.cpp构建 `df18bee11e96557b07aa7b12db0976f3c7ac9165`。调用前后槽位空闲、服务健康。

| 独立请求 | 结果 | 输入／输出tokens | 本机测得耗时 |
|---|---|---|---|
| 文本计算 | HTTP200，最终结果964正确 | 31／246 | 2.218秒 |
| 原第2张图片＋严格JSON | HTTP200，完整schema和原ID通过 | 1012／1023 | 11.771秒 |
| 原完整六图＋严格JSON | HTTP200，完整schema、六个唯一原ID及顺序通过 | 4937／1846 | 47.630秒 |

三项均为精确模型alias、`finish_reason=stop`、完整usage；总9095 tokens，无自动重发。所有响应均有独立非空 `reasoning_content`，最终内容取 `message.content`。只对完整content做严格JSON解析，不提取片段，不删除解释文字；思考不作为答案或下一轮上下文。未返回独立reasoning token数，不能按字符数估算或重复计费。

图片请求直接复用此前保存的原data URL、提示、schema、ID与顺序，仅更换模型并明确 `temperature=0`、`max_tokens=8192`、`stream=false`。文本额度2048；服务启动时已开启Thinking，本轮未额外指定Thinking字段。三个请求均采用180秒读取超时，无重试。本次没有调用ModelBest或Doubao；自建推理的服务器成本未估算，也未伪造为共享云费用账本的已结算请求。

原ModelBest对第2张图片及六图的拒绝历史保持不变；新服务本次成功不改写旧失败。新服务的量化、推理后端、Thinking、上下文及生成参数均不同，不能将结果差异仅归因于模型能力。

## 尚未完成的评测接入

这三项是连接与接口探针，**不等于原30项验收、六工具完整链路、3题WeMM端到端或100题／111轮通过**，100题尚未启动。当前 `127.0.0.1:18080` 是Mac上的地址，不能直接让 `.100` 上的评测进程使用同一localhost地址。

后续如沿用原100题目标，应建立独立自建Q6客户端和版本化协议，不直接改写ModelBest客户端的默认行为：

- 固定自建服务器身份、GGUF/视觉投影器及后端版本，冻结Q6 Thinking配置与原相册快照。
- 读取 `reasoning_content`，明确本地推理与Doubao的费用口径，保留请求journal及未知请求停止规则。
- 为 `.100` 建立明确、受限的推理连接；不能把HTTP API直接公开到网络。
- 根据已知约250K输入验收曾耗时126–130秒，单独冻结合理超时，不能沿用云端90秒却声称支持同样长输入。
- 新runtime、claim和run_id下完成完整接口验收及3题真实WeMM链路，通过后再跑100题；保留历史Qwen与云端MiniCPM结果。

## 私有证据

本地目录：`runtime_refs/minicpm-selfhost-connect-20261015-verified/`。

- `claim.json`、`probe.py`：固定三项计划、调用参数和无重试执行器。
- `preflight.json`、`postflight.json`：模型/配置与健康/槽位。
- `request-journal.jsonl`、各项request、raw response和result：完整请求身份、原图片输入、响应及解析结果，均为私人运行材料，不进入Git。
- `result.json`：三项通过；总9095 tokens；正式评测未启动。
- `independent-audit.json`：独立重新核验原图字节、完整JSON/schema/ID与usage；SHA256 `503e84f28715af9b1dd5c76813e523de41fc02c4f0abb05c8bb3c94bd4d08a83`。

先前隧道关闭与免密认证失败单独保存在 `runtime_refs/minicpm-selfhost-connect-20261015-3b7654/connection-check.json`，不覆盖原失败。

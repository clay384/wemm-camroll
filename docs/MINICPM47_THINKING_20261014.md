# MiniCPM-V 4.7 Thinking On 接入与验收交接

后续用户已授权省略Thinking参数；最新独立接入与验收记录见[SDK默认协议交接](MINICPM47_SDK_DEFAULTS_20261014.md)。以下首次失败及追加记录保留。

观测日期：2026-10-14，Asia/Shanghai。本次实施的是独立新协议；当前采用的 Qwen3.7 Flash / M4 / Gate-Off 默认入口保持兼容。

## 当前结论

接入代码、独立执行监督器及8774只读报告已实现。**真实图片能力探针未通过，30次稳定性验收、3题端到端和100题／111轮正式评测均未启动。** 没有新增 MiniCPM 正式成绩，不能与 Qwen 计算质量差值。

固定真实图片上依次验证 `json_schema → json_object → 提示词约束`，各一次独立请求。三次均 HTTP 200、型号正确、usage 完整，`finish_reason=stop`，但 SDK 保存的 message 只有 role/reasoning，content 为 None 或缺失。三次分别为 prompt 2671、completion 18（含 reasoning 17）tokens，严格 JSON 均失败。没有提取 reasoning 冒充答案、删除说明文字或整批重跑筛选成功。

离线使用真实 OpenAI SDK 的 MockTransport 已核验：顶层 `chat_template_kwargs.enable_thinking=true`、`max_tokens=8192`、`stream=false` 均发送正确；非空 content 完整保留。`model_dump(exclude_none=True)` 只省略空字段。因此本次可确认此请求配置下未获得最终内容，尚不能定位 ModelBest 服务端根因，也不能泛化为模型完全不具备图像理解能力。

GPU1 本次仍由其他任务占用，约15,975MiB；未取得 GPU 窗口，未启动 WeMM、未终止他人进程、未切换GPU0或8773。8774已明确显示验收阻断及正式未启动，全部历史13条运行保留。

## 离线验证

最终Mac全量回归：**1048 passed，1 skipped**。JS语法与diff空白检查通过；覆盖旧Qwen/Gate兼容、严格JSON、reasoning分离、零价unknown、重放/预算、参数/ID/资产、上下文单轮失败继续、报告受限比较和GPU占用拒启。Linux结果与最终源码清单见私有 `validation.json`；本次真实探针仍失败，离线通过不能替代云端验收。

## 代码与冻结协议

协议：`minicpm47-thinking-m4-off-v1`。

| 模块 | 职责 |
|---|---|
| `camroll_agent/modelbest.py` | 独立传输、Thinking、8192非流式输出、strict JSON、native/JSON decision、用量与请求journal |
| `camroll_eval/minicpm_experiment.py` | 显式MiniCPM配置、复用原索引、原Doubao质量/行为协议、单轮context_limit失败后继续 |
| `deployment/minicpm_probe.py` | 一次性能力探针、精确旧unknown例外、模式选择；失败不进入验收 |
| `deployment/minicpm_acceptance.py` | 固定10类×3次、图片事实/协议/长历史、完整失败与未执行状态 |
| `deployment/minicpm_config.py`、`minicpm_suite.py` | 独立runtime、e2e/formal门槛、GPU1归属、child监督、冻结/恢复/费用守卫 |
| `deployment/minicpm_report.py` | 8774只读状态；严格限定新组与原F37-Off两份固定来源比较 |

端点 `https://api.modelbest.cn/v1`，精确型号 `MiniCPM-V-4.7-35B-A3B`，温度0，Thinking On，非流式，8192输出，256000上下文。目录中的最大输出0只表示未说明；本次请求未验证有效8192最终输出能力。Agent 600秒、25决策步、5次看图、50次真实云请求，单请求90秒并受剩余轮次时间限制。SDK自动重试关闭；明确429/空404只允许一次补发，未知请求停止。reasoning只存原始响应journal，不进入答案或后续工具上下文，已包含在completion计费中。非流式TTFT不可用。

固定原快照 `bcc1cd1176ab39697c789a4f44c7426e`，source run `w38max-formal-20260930-122300-813923`。原355图的路径、描述、事件、向量保持原位。历史对照仅 `f37-off-gate-20261006-181014-8bd9a5`，其run.json SHA256为 `8ee4497f779148783241dc0c2add0574d727e8a368e96687b6aca0bdbcc91d37`；本次只读重验一致，没有重跑或重新评分。

Thinking、输出额度、JSON/tool模式、非流式传输和上下文失败处理与旧Qwen不同；即使后续完整执行，也只能解释为同快照推理配置对比，不能把差值全部归因于模型或作严格速度排名。

## 证据位置与费用

私有runtime：`.100:/home/realmagic/camroll-wemm/runtime/minicpm47-thinking-m4-off-20261014-6675f0/`。

Mac回执：`/Users/zhanyiqi/Documents/camwwem/runtime_refs/minicpm47-thinking-m4-off-20261014-6675f0/`。凭据、图片、完整响应及私人用例均不进入Git。

- `acceptance/probe-results.json`、`probe-calls.jsonl`、`probe-journal.sqlite`：三次尝试、实际型号、reasoning与usage、错误完整保存。
- `acceptance/status.json`、`status.json`：验收阻断、正式未开始；0/111轮执行。
- `acceptance/case-plan.json`、`fixture-records.json`、`nonformal.jsonl`：七张目视预标注原图、30用例准备稿、3个非正式问题。未使用正式QA/GT选样；未声称完成30次或模式冻结。
- `pricing-evidence.json`：14:20官方控制台原始模型配置，input/output价格均为字符串0、`is_free=false`；不假设积分与人民币换算或永久免费。
- `acknowledgement.json`、`authorization.json`：只对旧请求 `011a8df1906847719e30c959cfaec785` 建立本runtime精确例外，原1.268122元预留及证据保留、不重发、不释放；新unknown仍阻断。
- `acceptance/attempt-audit.json`、`completion-audit.json`：调用、费用、保护文件及服务状态回执。
- `report-deployment/`：新旧报告进程身份、19份历史文件哈希、API/DOM验证及报告回滚依据。
- `validation.json`、`final-source-manifest.json`：最终离线验证及独立发布清单；探针另存当时源码清单，不能用后续代码覆盖它。

本次3次请求共 **8067 tokens**（输入8013，输出54，reasoning51已含输出；图片tokens7635已含输入），共享账本增加3条已结算零价调用，新增unknown为0。累计仍为 **600元上限／已结算375.308192元／未知预留1.268122元／可用223.423686元**。

调用后官方控制台显示5次成功请求、11211 tokens（输入10681、输出530）、消耗0.00积分、余额110.00积分，与原独立3144-token探针加本次8067吻合。原3144-token探针仅关联原记录及账号证据，不补写成共享账本已结算。

## 15:05–15:06 追加诊断与GPU释放

用户随后明确指令“gpu1直接杀掉有关进程”。新核验PID、启动时间、父子关系和`CUDA_VISIBLE_DEVICES=1`后，仅对GPU1的Qwen3.5-4B vLLM父进程90931和EngineCore 92444发送SIGTERM；两者及资源追踪子进程92443均退出，无需SIGKILL。显存从15975MiB降至18MiB，随后复查无GPU1计算进程。未控制GPU0、未启动WeMM。该明确授权只用于本次已核验任务，不作为今后任意终止其他进程的授权。

15:05:51重新读取官方模型目录，输入/输出仍为0、`is_free=false`。15:06:41对“请仅返回 OK”执行一次纯文本Thinking On诊断（无图片、工具和JSON约束；温度0、8192、非流式），HTTP200、型号正确、finish_reason=stop，仍缺少content。prompt47、completion2（其中reasoning1）共49 tokens，已结算0元，无新增unknown。它表明问题不限于图片或JSON约束；本次没有定位服务端根因，没有重发空响应，也没有绕过验收启动正式评测。

新增证据保存在原runtime下独立的`diagnostics/gpu1-stop-20261014T070518Z-8103d0/`和`diagnostics/text-recheck-20261014-150641-4b1243/`，Mac私有回执同步保留。原三次失败、冻结文件和证据归档不改写；本文前面的GPU占用描述为本次释放之前的历史观测。

## 15:09 官方最小SDK示例复核

用户提供只含`model`和`messages=[{"role":"user","content":"Hello"}]`的OpenAI SDK示例。按此请求体独立执行一次，未传Thinking、temperature、max_tokens、stream、工具或response_format；仅在客户端关闭自动重试并设置90秒超时。实际HTTP请求体已保存，HTTP200、型号正确、finish_reason=stop，content正常返回`Hello! How can I help you today?`，无reasoning字段，reasoning_tokens=0。输入46、输出10，共56 tokens，零价已结算，无新增unknown。

因此当前凭据、端点及默认文本调用正常。此前显式Thinking On／温度0／8192／非流式的组合仍无content；这两次诊断没有单独隔离所有参数，不能把根因确定归于Thinking一个参数。默认示例成功不代替Thinking On真实图、JSON及稳定性验收，100题继续未启动。

新增证据：`diagnostics/sdk-defaults-20261014-150938-53ec0b/`，含实际请求体、原始HTTP响应、journal、usage、计价及预算前后回执，已同步Mac私有目录。本轮两次追加诊断共105 tokens，均零价；共享600元账本余额仍223.423686元、旧未知预留1.268122元。

## 后续执行和恢复

先解决当前配置下 content 为空的问题，再创建新的独立验收身份和版本；保留本次失败runtime及claim。不能修改旧请求、降低严格解析要求或跳过模式/30用例门槛。模式通过后冻结 actual decision/json模式，再执行一次30用例及完整输出复核。GPU1归属重新核验可用后，监督器才允许3题端到端；正式入口同时要求云验收与端到端均通过。

入口均要求显式 `--runtime`，网络执行另要求 `--execute-paid`。以下是门槛已满足时的接口示意，**当前runtime不满足前置条件，不应直接执行**：

```bash
python deployment/minicpm_probe.py --runtime "$NEW_RUNTIME" --execute-paid
python deployment/minicpm_acceptance.py --runtime "$NEW_RUNTIME" --execute-paid
python deployment/minicpm_suite.py --runtime "$NEW_RUNTIME" --stage e2e --execute-paid --acknowledgement "$NEW_RUNTIME/acknowledgement.json" --smoke-questions "$NEW_RUNTIME/acceptance/nonformal.jsonl"
python deployment/minicpm_suite.py --runtime "$NEW_RUNTIME" --stage formal --execute-paid --acknowledgement "$NEW_RUNTIME/acknowledgement.json"
```

每阶段重新保存带时间的价格证据，非时间字段变化或证据过期停止。freeze固定源码、相册、输入、原对照哈希、Judge协议及费用例外。正式Agent并发1/Judge4；上下文超限是明确单轮失败，其余计划继续；新unknown、预算、基础设施错误停止。只有终态来源、100题/111轮/81题图片分母和相同Judge协议全部核验后，`write_comparison(runtime,run_id)` 才发布对照清单。

本次没有进入GPU窗口，原停止的问答/检索服务保持停止；只更换8774报告进程。报告回滚按 `report-deployment/before-switch.json` 与 `active-report-process.json` 联合核验PID、start tick、cwd及命令后，停止新报告并恢复原报告命令。只回滚报告代码，不回滚预算、运行证据、相册或历史结果。

官方接口资料：[OpenBMB API说明](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/api.md)；当前价格以登录后的 [ModelBest模型目录](https://platform.modelbest.cn/console/models) 原始响应为证据。

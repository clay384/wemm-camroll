# 自建 MiniCPM Q6 Thinking 接入与评测（2026-10-15）

## 验收终态（10月15日10:50）

自建客户端及独立协议已接入；Mac/Linux 各1162项通过、1项跳过。基础图片JSON Schema及原生工具往返探针通过，但完整30项验收为25项通过、5项失败，未达到冻结门槛。**3题端到端和100题／111轮未启动，当前无新正式对比成绩。** 本轮不重复同一配置筛选通过结果。

8774已使用新只读报告入口，原13个历史运行保留。报告启动曾因端口尚未释放失败，失败与回退尝试日志均保留，随后正常启动。

## 独立运行与固定参数

- 协议：`minicpm47-q6-thinking-m4-off-v1`。
- 型号：`minicpm-v4.7-35b-a3b-q6-thinking`；Q6_K、F16 KV、262144上下文、单槽位。
- 服务端Thinking开启；请求不添加云端Thinking字段。温度0、stream=false、max_tokens=8192，单请求600秒／单轮1200秒，25决策步／5次看图／50次真实请求。
- 冻结模式：`native_tools` + `json_schema`。
- llama.cpp：`df18bee11e96557b07aa7b12db0976f3c7ac9165`；模型revision：`f244654312a11af8c31835ed51a4753a094310bc`。模型和视觉投影器实际SHA256已重新核验。
- `.100`通过独立SSH隧道直达模型服务器，API绑定`.100`回环地址`http://127.0.0.1:18080/v1`，不依赖Mac持续在线。密码未保存。原Mac隧道与模型进程保持运行。
- release：`.100:/home/realmagic/camroll-wemm/releases/minicpm-q6-20261015-102743-a4c6c7`，430文件冻结。
- runtime：`.100:/home/realmagic/camroll-wemm/runtime/minicpm47-q6-thinking-m4-off-20261015-102743-a4c6c7`。
- Mac私有镜像：`runtime_refs/minicpm47-q6-thinking-m4-off-20261015-102743-a4c6c7/`。

新客户端`camroll_agent/minicpm_selfhost.py`复用原请求journal、零价记账、严格JSON及工具执行链路；单独处理`reasoning_content`，原始思考只留私有journal，不进入后续上下文。旧Qwen和ModelBest默认配置保持兼容。

## 门槛与历史对照

原355图快照`bcc1cd1176ab39697c789a4f44c7426e`不重建；原相册描述、事件、向量和资产路径保持不变。Agent、看图和M4均绑定自建模型，Gate-Off保留原有有限修复及外部Doubao质量／行为评分。

30项为10类各3次独立请求或完整会话，包含最多12图片输入、六工具、M4及合成长历史。全部接口、格式、图片ID、引用、usage和Thinking证据通过后，进行独立审查，再做3道非正式真实WeMM链路测试。事实错误只作能力观察，不修改答案，不阻断接口验收。正式100题/111轮只能在这些门槛通过后开始。

历史只读绑定`f37-off-gate-20261006-181014-8bd9a5`，SHA256 `8ee4497f779148783241dc0c2add0574d727e8a368e96687b6aca0bdbcc91d37`；不重新调用或评分Qwen。比较明确披露量化、后端、Thinking、输出额度、传输方式及600/1200秒时限差异，不作严格速度排名或单模型因果归因。

## 预算、GPU与失败保留

进入本轮前账本上限600元，已结算375.308192元、旧未知预留1.268122元、可用223.423686元。自建API费用记0，服务器成本未核算；Doubao按原规则计费。旧未知请求精确例外保持，新未知即使金额为0仍阻断。前三次连接探针9095 tokens独立关联原记录，不伪造成已结算。

10月15日10:18 GPU1的新vLLM任务已获用户明确授权临时停止并恢复。已只读确认当前模型为`qwen3.5-4b`、本地8103端口、CUDA设备1，记录可恢复配置；本轮未进入GPU窗口、未发送停止信号。每次GPU执行窗口前重新核验进程身份，不使用文档旧PID，不操作GPU0。

此前ModelBest HTTP400、旧30项未完成状态和逐张／组合诊断全部保留。本轮独立保留SSH控制socket路径过长、报告端口释放延迟等部署失败。原8773不变；报告回滚只切代码，不回退账本或实验数据。

## 私有证据入口

- `source-manifest.json`、`validation.json`、`local-validation.json`和两端测试日志。
- `deployment-manifest.json`、`connection.json`、`pricing-evidence.json`、`budget-before.json`。
- `acceptance/probe-results.json`、`acceptance/protocol.json`、`acceptance/status.json`、逐项cases和原始request-journal。
- `report-before.json`、`report-after.json`及全部启动日志。
- `gpu_stage.py`与`gpu1-restoration-preflight.json`为本轮私有运维材料；仅在验收通过后运行，结束恢复原任务。

所有图片、QA/GT、原始请求、推理文本、数据库及运维私有记录均不进入公开源码仓库。

## 30项结果与独立审计

| 用例 | 结果 |
|---|---|
| 场景、物体、OCR、人物外观 | 各3/3结构通过；事实差异原样保留 |
| 六图、原图＋合法裁剪共12输入 | 各3/3结构与ID交付通过，无400 |
| 六工具完整会话 | 0/3最终格式通过；六工具本身正常，最终outcome写成答案文字 |
| 最终回答JSON | 3/3最终通过，但每例前两次空content，第三次有限修复才成功 |
| M4最终复核 | 3/3，均仅一次调用；OCR字形偏差保留 |
| 合成长历史 | 1/3通过；最大规模两例缺少answer，outcome填成marker |

- 共30项全部执行，25通过、5失败，最终结构通过率83.33%。
- 首次结构通过43/51单元（84.31%），首次全部结构通过22/30用例（73.33%）；均低于95%要求。
- 验收57次请求全部HTTP200，包含6次已知空content格式失败；无HTTP400、输出截断、usage缺失或新增未知请求。
- 基础能力探针另3次；本轮合计60次请求，389507输入tokens、19545输出tokens、409052总tokens；每次均有独立Thinking证据。
- 非正式预标注事实观察89/98（90.82%），仅作记录，不是正式100题准确率，也不能替代接口门槛。
- 最大合成历史347380字节、24条消息，实际49592输入tokens；不是上下文超限。已有部署约250K输入验收属于另一份历史证据。

`completion-audit.json`独立复核全部30项、60份journal响应、usage、资产和模型绑定。图片data URL从原图和已记录裁剪区域重建，所有完整请求哈希均与发送前journal一致，另存私有`reconstructed-exact-requests/`；审计未新增模型调用。

审计文件SHA256：`c0e680cfb71f612e248676aab1c733fad4e00ed0a8c371c663d9409f65a7fbd7`。审计通过只表示记录完整可信，**不表示模型验收通过**。`failed-acceptance-observation-review.json`保留逐项人工观察和字段名误报等说明，不改写验收状态。

## 费用与恢复回执

本轮自建API记账0元，服务器成本未核算；未调用Doubao。共享账本仍已结算375.308192元、原未知预留1.268122元、可用223.423686元，无新增未知请求。两份请求journal已生成一致性快照。

10:50完成恢复：仅关闭本轮从`.100`新建的直连隧道；原Mac隧道与自建模型服务仍健康，GPU1原任务保持原进程，WeMM未启动。5个保护文件、1167个索引文件、原相册绑定及历史F37-Off哈希核验通过。8774保留新只读阻断报告与原13个历史运行，8773不变。

后续需分别诊断原生工具长链的最终字段错误、JSON Schema下空content问题。若调整工具或JSON模式，必须建立新的冻结配置与验收记录；不能把本轮失败改为通过或绕过门槛直接运行100题。

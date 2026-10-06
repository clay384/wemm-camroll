# Flash Gate 四组运行

本轮只调用 `qwen3.7-flash`、`qwen3.8-flash` 和独立 Doubao Seed 2.0 Lite。新入口不会执行 Max 测试。索引中历史生成模型仍原样保留，仅作为索引来源记录。

固定顺序为 F37-Off、F37-On、F38-Off、F38-On；正式组共享快照 `bcc1cd1176ab39697c789a4f44c7426e`，不重新建库。新目录是 `.100:/home/realmagic/camroll-wemm/runtime/photobench-flash-gate-20261006f`。原相册、原运行、历史费用和无效补测均不重写。

`photobench-flash-gate-20261001` 保留首次启动的完整证据：真实小样通过，正式入口在首题和付费调用前因新评测根目录缺少 `runs/` 而失败，监督器已恢复。修复后使用上述新目录、重新验证源码和真实小样；不删除旧claim，不覆盖旧记录，也不把启动失败记作模型答题错误。

`photobench-flash-gate-20261002` 保留第二次小样。该次Off模型的引用错误已被正确拦截和保留，但小样入口错误地把“所有回答正常”作为链路验收要求而阻断。验收现在区分完整测量与答题正确：允许明确归类的模型契约失败或Gate耗尽，要求按规则跳过评分并记录错误；基础设施／校验器故障、缺轮、缺评分、缺看图或最终复核链路仍阻断。On还须观测到有效事实Gate判定。禁止通过重跑选择无错误基线；此前小样状态和费用保持原样，正式模型提示词未调整。

`photobench-flash-gate-20261002b` 保留文字探针返回 `OK.` 而非 `OK` 导致的误阻断。文字能力探针接受末尾句号，并另记 `exact_output_match=false` 与原输出；空响应或其他内容仍失败。图片、工具和结构化输出能力门槛未变，正式请求及提示词不变。

`photobench-flash-gate-20261002c` 保留Doubao两次把问题上下文用作answer结论主要证据而被拒绝的校验器故障。格式修复提示现在明确区分原图／可信元数据、派生观察和上下文，提供允许引用的来源ID；不能支持结论时必须保留证据不足，不得制造支持。每个候选仍最多一次格式修正，原解析规则和事实标准不放宽，旧故障记录不覆盖。

`photobench-flash-gate-20261002d` 的小样通过后于10月6日正式执行，但Gate证据装配的独立48 KB限制会拒绝仍处于模型上下文容量内的合法多图证据，且裸输入异常被误记为模型输出错误。发现后停止套件、结清已发请求并恢复GPU1，全部记录保留。修复移除这个重复且不一致的阈值，原样保留证据，继续使用含图片、文字、输出及安全余量的保守上下文检查；输入契约错误带 `gate_input_error` 并停止对应组。真实模型输出格式错误仍按原规则最多修正一次。未改变模型提示词、事实标准、输出上限或最终Judge协议，也没有按正式答案调参。新目录重新验收并以新run执行四组，不续写或覆盖旧首跑。

上下文估算中的单图预留由32768修正为8192 token，仍高于[火山官方图片说明](https://docs.volcengine.com/docs/ark/image-understanding)所列Seed 2.0默认high的1280及xhigh上限5120（2026-10-06核对）。此改动仅修正本地容量估计，图片JPEG预处理、请求中的默认detail和原始证据内容均不改变；文字JSON仅无损去掉分隔空白，不截断文字。按[官方型号表](https://docs.volcengine.com/docs/82379/1553576?lang=zh)同时检查229376 token最大输入与262144总上下文。完整窗口最高档价格的费用预留保持原样，真实超限仍在发送前停止。

派生的视觉观察保留VLM返回的完整分析、逐图观察、局限以及工具关联的图片ID和裁剪信息；不再把工具另外拼接的整份 `images` 相册记录重复包装成模型观察。本轮复核图的登记资料、身份来源和caption仍按既有来源分类分别提供，不降低真实性标准。未删除任何视觉观察正文，原始完整工具结果也继续保存在审计轨迹中。此改动只在新版本生效，旧记录不改写。

`get` 的已返回ID集合同时包含工具记录明确给出的 `event_id` 关联；此前遗漏会把合法事件读取误判为未知ID。只承认实际返回记录的关联字段，不扫描模型正文、不放开整个索引中的事件。未返回或虚构的事件仍被执行前拒绝。

`photobench-flash-gate-20261006e` 保留修复后的真实合成小样：两种Flash能力探针通过，但Doubao两次输出的顶层引用与各理由引用的并集不一致；已完成结算并恢复，尚未发送新正式题。旧解析器有此约束，而 `inconsistent_verdict` 的格式修正提示未明确说明。f版本只为这一格式错误补充已有的总状态优先级和引用并集要求，不改事实判定、初始SYSTEM_PROMPT或最终Judge，仍仅允许一次格式修正，程序不代改模型结论或引用。正式答案未用于调整；旧小样不覆盖、不伪装为通过。

## 门槛与执行

源码需发布为独立 `releases/gate-<commit>`，使用现有 `.venv-photobench/bin/python`。不使用开发目录启动正式测试。

```bash
cd /home/realmagic/camroll-wemm/releases/gate-<commit>
/home/realmagic/camroll-wemm/.venv-photobench/bin/python deployment/gate_validate.py
/home/realmagic/camroll-wemm/.venv-photobench/bin/python deployment/gate_cloud_smoke.py --execute-paid
/home/realmagic/camroll-wemm/.venv-photobench/bin/python deployment/gate_suite.py --execute-paid
/home/realmagic/camroll-wemm/.venv-photobench/bin/python deployment/gate_export.py
```

`gate_validate` 运行原有与新增回归，记录对应源码清单。`gate_cloud_smoke` 在独立合成两图／三题上验证两种 Flash 的文字、图片、工具、结构化输出、Doubao 双图事实判定和最终评分，再用真实 WeMM Backend → Library → Agent → 看图 → 最终复核链路运行四个小组。合成父相册只用 W37，与正式相册隔离。所有调用使用原账本且计费。

正式入口要求对应源码的离线验证与真实小样全部通过；冻结源码、输入、索引、模型文件和参数。四组各自使用新 run、journal、查询客户端和题内上下文。100题／111轮／81个媒体指标分母固定；19道不可回答题保留。禁止正式答案指导提示词调参或自动补跑。

所有阶段使用原 `runtime/budget.sqlite` 累计600元上限；不重置、不自动提额。实际返回型号必须与请求 Flash 一致；缺失／混用型号阻断。结果未知或用量缺失保留预留并阻止后续提交。费用按请求 journal 归属，套餐实付未接入。

## Gate 语义

On 每个逻辑输出最多4次生成：首次、原样重发、分类提示修复、最后修复。第一次与第二次完整请求载荷一致，使用不同持久化请求身份；回放不作为网络重发。工具批次先全部验证，成功工具结果去重。所有物理请求（含 Doubao 格式修复）共用50次额度及600秒轮次期限。

规则检查 JSON、参数、实际返回ID、资产路径、裁剪及引用／交付／复核集合。事实 Gate 只收到当前问题、候选答案、本轮已复核原图（最多6张）与标明来源的元数据／派生文字，不接收参考答案、GT或最终评分。Gate与最终Judge使用独立提示词和请求身份；同为Doubao存在评判关联，通过Gate不保证真实。

实际交付图片ID与其余复核上下文图分开传给事实Gate，不能用另一张正确的上下文图片为错误交图背书。同一轮中，问题／候选答案／交付ID／证据及原图内容完全一致时，复用已有有效判定；正负判定均缓存，异常不缓存。改变候选或证据才产生新判定。复用判定和journal回放均单列，不计入新的判定或原样重发恢复率。

正常回答、澄清和拒答可评分。Gate耗尽与程序兜底记为执行无效、预期跳过评分；固定分母不因此缩小。合法资产缺失、索引／配置变动、401/403、检索服务故障、未知请求等停止新增付费提交。格式错误的校验器只修复一次，之后明确记录校验器故障。

## 监督、恢复与报告

新窗口ID沿用已有 GPU 所有权登记和租约机制（600秒、30秒续租），只使用GPU1。共享所有权登记用于防止旧／新监督器同时占卡。本轮只切到WeMM，不加载Qwen检索worker。任何退出路径均恢复进入前服务状态；直接子进程清理失败也必须尝试监督器恢复。守护进程可在主进程退出或租约超时后恢复。不要根据历史PID控制进程。

`gate_report.py` 只读绑定8774，不读取云凭据，不挂载Agent；端口冲突直接失败，不杀占用者。显式核验并停止自己拥有的旧8774报告进程后，才可部署新报告；不切换8773。新报告同时只读访问历史报告目录，但禁止把新Gate协议与旧协议混为同一次比较。

```bash
/home/realmagic/camroll-wemm/.venv-photobench/bin/python deployment/gate_report.py
```

保存每次候选、校验依据、完整调用和失败状态：各run的 `gate_events.jsonl`、`model_calls.jsonl`、`verifier_calls.jsonl`、`judge_calls.jsonl`、`turn_attempts/`、`results.jsonl`、`request-journal.sqlite`。这些都是私人运行产物，不提交Git。`gate_export.py` 生成私有 Markdown／JSON、成对差值与费用归属；只在四组完整、应评分轮均完成且恢复核验通过时标记完整交付。album3是已见基准，不作为独立泛化能力证据。

`formal-suite.claim`、每组claim、smoke记录均防止误重跑。已有文件或未知请求必须先人工检查；不要删除claim以绕过付费请求结果未知保护。预算阻断时保留全部计划和未执行状态。

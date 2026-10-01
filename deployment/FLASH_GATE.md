# Flash Gate 四组运行

本轮只调用 `qwen3.7-flash`、`qwen3.8-flash` 和独立 Doubao Seed 2.0 Lite。新入口不会执行 Max 测试。索引中历史生成模型仍原样保留，仅作为索引来源记录。

固定顺序为 F37-Off、F37-On、F38-Off、F38-On；正式组共享快照 `bcc1cd1176ab39697c789a4f44c7426e`，不重新建库。新目录是 `.100:/home/realmagic/camroll-wemm/runtime/photobench-flash-gate-20261001`。原相册、原运行、历史费用和无效补测均不重写。

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

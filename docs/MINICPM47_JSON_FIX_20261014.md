# MiniCPM 4.7：HTTP400根因与JSON兼容修复（2026-10-14）

承接[SDK默认调用及原失败](MINICPM47_SDK_DEFAULTS_20261014.md)。本轮用户授权：补全错误日志、重复验证接口错误、修复接口和JSON；Thinking保持未指定。识图及回答能力错误接受并如实评分，接口验收通过后继续原100题／111轮。旧失败、原相册与历史Qwen评分保留。

## 已验证的根因

原JSON决策请求连续三次独立调用均HTTP400，服务端正文为 `System message must be at the beginning.`。适配器新增system后与原system形成连续两条；只合并开头system且保留全部内容后，错误变化为 `Grammar error: Unimplemented keys: ["uniqueItems"]`。

随后只从发给服务端的response_format schema移除uniqueItems，保留提示里的完整schema和本地完整JSON校验，三次均HTTP200并返回正确get决策，完整usage及精确型号正确。当前证据没有表明oneOf不受支持。诊断共7次物理请求：4次明确400、3次成功；7225 tokens、0元，无新增unknown。

原始请求失败正文未保存的日志缺口已修复：新journal保存脱敏完整HTTP正文、错误结构、状态码、白名单request-id、异常类型、配置及请求哈希；旧记录不补造、不覆盖。429/空404有限补发的首次失败也保留。

## 新协议与执行边界

新增 `minicpm47-sdk-defaults-m4-off-v2`，`acceptance_policy=interface-only-v2`、`adapter_version=modelbest-json-v2`。Thinking、temperature、max_tokens、stream仍未指定。新适配只合并连续开头system；不移动或删除中途system，不删减历史证据。仅转换服务端不支持的uniqueItems，本地仍严格检查完整schema、去重、工具参数、图片ID与资产绑定。

能力答错会记录并进入原Doubao质量/行为评分，不作为接口验收阻断；结构错误、非法引用、缺usage、未知调用、预算和基础设施错误仍受原限制。新版30项接口验收及3题真实WeMM链路必须通过后才开始100题。原v1配置和冻结结果保持，重试/复现均有独立物理请求身份。

## 私有证据与状态

诊断runtime：`.100:/home/realmagic/camroll-wemm/runtime/minicpm47-sdk-defaults-m4-off-20261014-160439-diagnose-279d5b/`。

新v2 runtime：`.100:/home/realmagic/camroll-wemm/runtime/minicpm47-sdk-defaults-m4-off-20261014-161142-v2-93f384/`。

所有私人证据保存于runtime及本地runtime_refs，不进入Git。以下为本轮最终观测，前述原始失败不改写。

## 16:32 最终状态：原兼容问题修复，新六图请求被服务端拒绝

冻结 release 为 `.100:/home/realmagic/camroll-wemm/releases/minicpm-v2-20261014-161912-06c689`，425个源码文件在Mac和Linux逐一一致。两端全量回归均 **1152通过、1跳过**；另有1项warning，完整日志保留。实际云端3个完整“工具决策→工具结果→最终答案”会话共6次HTTP200，严格JSON、参数、图片ID、型号和usage均通过；随后固定集成探针也通过。

正式30项验收已执行13项：**12项完成、1项失败、17项未执行**。前12项首次HTTP及结构化输出均成功；第13项 `six_images-1` 返回HTTP400。包含失败的整体首次HTTP成功率为12/13（92.31%），30项计划的已完成比例为12/30（40%），不能称为验收100%通过。可观察事实自动检查为32/42（76.19%），仅代表这些已执行样例；按用户最新授权，事实答错保留，不作为此次接口验收的停止原因。

新400正文为：

```json
{"error":{"code":"content_policy_violation","message":"这条内容暂时无法继续处理，换个话题试试吧。","type":"invalid_request_error"}}
```

按用户要求，原六图请求又独立执行两次：**原始请求加两次复现共3/3均为同一400正文**。完整payload SHA256均为 `4131000b545882e901757156591db6b67f676c350a649cdbd6f22be7d52d4e6a`；图片、提示、schema和参数完全相同。每次都有独立物理请求身份；原验收文件、claim和失败均未改写，不将诊断结果填回验收。

HTTP400表示服务器拒绝了请求，必须结合正文判断原因。本次原system布局和uniqueItems属于已证实的接口兼容问题，已修复；新的 `content_policy_violation` 是供应商返回的内容策略拒绝，不能算成模型答题错误，也不能据此判定模型不具备六图理解能力。现有正文未指出是哪张图片、输入还是生成内容触发，具体原因仍需供应商核验，不能推断为参数错、图片违规或模型能力差。

独立离线复核确认6个图片输入均可严格base64解码且JPEG完整，尺寸符合2048边界；实际请求仅含model/messages/response_format，system位于开头。未发现图片损坏、遗漏或本地请求注入，但这些检查不能定位供应商内部审核规则。

供供应商查询的三个request-id（尚未发送给任何人）：

- 原始：`req_5f9a78f6cd254dc9a4d03d65bf462d56`
- 复现1：`req_714573f39925425bbd1107be75d1a77a`
- 复现2：`req_36d0f6d3182b44db8738bac8818b7d7d`

**3题真实WeMM端到端与100题／111轮均未启动**。30项未通过接口门槛，不能仅把明确API拒绝记成答错后进入正式评测。后续应先查清并解决供应商拒绝，再建立独立的验收记录；不得重放或覆盖已失败claim，也不得改图片/提示规避拦截后冒充同请求通过。

## 费用、服务与历史保护

2026-10-14 16:31:53最终只读审计：本轮继续阶段（7次原接口定位＋24次v2调用）共 **31次物理请求，24次HTTP200、7次明确HTTP400，68877个已知tokens，新增费用0元**。成功24条为settled，明确拒绝7条为released；不把无usage的明确拒绝伪造为有token结算。此前独立3144-token探针仍按原证据单列，不并入本轮共享账本结算。

验收前16:22:49重新读取供应商目录：输入价0、输出价0、`is_free=false`；保留原始配置与时间，不推断积分到人民币换算。共享账本仍为上限600元、已结算375.308192元、旧未知预留1.268122元、可用223.423686元。唯一旧未知请求保留精确例外，无新增unknown或在途请求。

GPU1为18MiB、0%利用率，无计算进程，仅原Xorg。本轮未加载WeMM，8765/8770/8772/8773/18778保持停止，无需恢复模型服务；GPU0未操作。原355图快照三文件、F37-Off、5个WeMM保护文件、1167个索引文件和模型fingerprint全部一致。

[8774只读报告](http://192.168.0.100:8774/)已部署v2入口，显示停止原因、Thinking未指定、接口与能力错误的区分、原F37-Off绑定及全部历史失败。13条历史运行保留，比较入口尚未开放。报告代码部署不改变8773或历史结果；回滚仅按 `report-deployment/deployment-result.json` 的旧报告release/runtime恢复报告进程，不回退账本、相册或运行数据。

## 证据索引与接手限制

以下路径相对新v2 runtime；本地镜像位于 `runtime_refs/minicpm47-sdk-defaults-m4-off-20261014-161142-v2-93f384/`：

- `validation-local.json`、`validation.json`：两端全量离线结果与源码绑定。
- `diagnostics/three-complete-roundtrips/result.json`：3次完整工具会话。
- `acceptance/protocol.json`、`acceptance/status.json`、`acceptance/cases/`及journals：固定30项计划与原失败。
- `diagnostics/six-images-identical-reproduction/result.json`：三次相同请求拒绝；SHA256 `c4d8a974f2533c16f00e943511b37e2efbf20125838e9ca710716485b3fd8bed`。
- `diagnostics/six-images-identical-reproduction/ledger-audit.json`：重复调用及本轮账本合计。
- `completion-audit-final.json`：费用、GPU、服务和保护文件回执；SHA256 `be961860d486f3a8997fcc88b3550f94185e018b52453d05b642564e26c28be9`。
- `report-deployment/deployment-result.json`：只读报告部署、历史哈希保护及回滚源。
- `finalization/report-state-update.json`：最终报告状态更新，原验收哈希不变。

新协议指纹为 `1a63cff0a97874dafc774528216f5e9a673f278e76388fc14441db0203e1a9fc`。本次尚未提交或推送源码；测试/部署依据已冻结的文件清单，不以Git HEAD冒充新实现提交。此交接的最终补记晚于release冻结，因此不回写远端冻结源码。下一位Agent先读本页及最终回执，保留当前失败状态，切勿直接执行旧claim、旧四组实验或100题入口。

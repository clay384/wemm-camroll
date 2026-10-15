# MiniCPM六图HTTP400：逐张与组合诊断（2026-10-14）

承接[接口与JSON修复记录](MINICPM47_JSON_FIX_20261014.md)。用户最新授权：六张原图逐张测试，再逐步组合，保留所有失败；接口验收通过后继续原100题。Thinking仍未指定，原相册、已失败验收和历史Qwen结果不变。

## 16:45 结论

**拒绝稳定关联原第2张图片的请求；多图请求本身可以成功。** 原第2张单图三次均返回HTTP400 `content_policy_violation`；其他五张单独请求成功，不含第2张的2至5张组合也全部成功。包含第2张的所有已测组合均被拒绝，包括把第2张移到请求首位。

这定位了可复现的输入关联，但不能从黑盒响应确定是图片像素、图片ID绑定、输入审核、生成审核或供应商的哪条规则触发。图片是室内仪式舞台，能看到粉红色背景、白色长裙、蓝色裙装及持设备的人；不能据API错误认定图片违规，也不能据此判断模型不具备多图理解能力。

## 固定实验与结果

原六图请求的提示及两份schema都固定6个ID、6条输出。对子集只同步修改图片选择、顺序标记、ID顺序和schema的枚举及min/maxItems；system、任务语义、原data URL字节、模型及SDK默认参数保持一致。每个请求保存精确diff和完整实际wire body；原六图重新构造后的payload SHA256仍为 `4131000b545882e901757156591db6b67f676c350a649cdbd6f22be7d52d4e6a`。

图片编号均按原六图顺序，从1开始：

| 阶段 | 图片集合 | 结果 |
|---|---|---|
| 首轮单图 | 1 | HTTP200，严格JSON与ID通过 |
| 首轮单图 | 2 | HTTP400，content_policy_violation |
| 首轮单图 | 3 | HTTP200，严格JSON与ID通过 |
| 首轮单图 | 4 | HTTP200，严格JSON与ID通过 |
| 首轮单图 | 5 | HTTP200，严格JSON与ID通过 |
| 首轮单图 | 6 | HTTP200，严格JSON与ID通过 |
| 原顺序递增 | 1,2 | HTTP400，content_policy_violation |
| 原顺序递增 | 1,2,3 | HTTP400，content_policy_violation |
| 原顺序递增 | 1,2,3,4 | HTTP400，content_policy_violation |
| 原顺序递增 | 1,2,3,4,5 | HTTP400，content_policy_violation |
| 原六图对照 | 1,2,3,4,5,6 | HTTP400，content_policy_violation |
| 相同单图复现1 | 2 | HTTP400，content_policy_violation |
| 相同单图复现2 | 2 | HTTP400，content_policy_violation |
| 排除第2张的递增对照 | 1,3 | HTTP200，严格JSON与ID通过 |
| 排除第2张的递增对照 | 1,3,4 | HTTP200，严格JSON与ID通过 |
| 排除第2张的递增对照 | 1,3,4,5 | HTTP200，严格JSON与ID通过 |
| 排除第2张的递增对照 | 1,3,4,5,6 | HTTP200，严格JSON与ID通过 |
| 第2张移至首位 | 2,3 | HTTP400，content_policy_violation |

共18个独立物理请求：9个HTTP200、9个明确HTTP400。所有成功响应均有正确型号、完整usage、有效完整JSON及正确ID顺序；没有通过正则截取或删除解释文字修补输出。图2单图三次完整payload哈希相同：`2607e808d79cbab35169718f55b198a89b18a031f4f90c19e9b9b87469000fd8`。

所有明确拒绝的正文相同：

```json
{"error":{"code":"content_policy_violation","message":"这条内容暂时无法继续处理，换个话题试试吧。","type":"invalid_request_error"}}
```

## 为什么100题仍未启动

这些子集用于定位问题，没有从正式相册删除第2张，也没有把“排除第2张成功”写成“原六图通过”。原30项验收仍为12完成、1失败、17未执行，原状态SHA256保持 `3632369c3b49a285a734cefc1478e24c475ff11be6a1b6557329ba3040d6f72a`。原六图本轮再次失败，因此接口门槛仍未通过；3题真实WeMM端到端和100题均未启动。

下一步需要供应商核验此图相关请求被拒绝的具体原因，或得到能保留原输入的合法接口修复。修复后使用新runtime、claim和journal完整验收，保留本次全部失败；30项及3题E2E通过后，才能运行原100题／111轮。事实答错继续只作为能力结果记录，不作为接口门槛。

## 费用与证据

调用前16:40:16读取ModelBest模型目录，HTTP200，request-id `req_5a4d66c312ea4e64ba94cd3e4da26db1`；原始输入价和输出价均0、`is_free=false`，保留时间及完整模型字段。

本次18请求有 **55320个已知tokens**（输入52821、输出2499），新增费用0元，无新增unknown。错误请求完整保留并按明确拒绝记账，不伪造usage。共享600元账本沿用原余额与唯一旧未知预留；本轮未运行Doubao评分或加载WeMM。

16:47:01最终只读审计：累计账本已结算375.308192元、旧未知预留1.268122元、可用223.423686元，无在途请求。GPU1仍18MiB且无计算进程；仅8774监听，8765/8770/8772/8773/18778保持停止。原验收、相册三文件、F37-Off、5个WeMM保护文件、1167个索引文件及冻结release全部不变。

本次继续复用冻结release `minicpm-v2-20261014-161912-06c689`，未修改接入源码或协议；原Mac/Linux各1152通过的证明保持与冻结源码绑定。诊断脚本及私人图片/请求仅保存在runtime，不进入Git。

远端runtime：`.100:/home/realmagic/camroll-wemm/runtime/minicpm47-sdk-defaults-m4-off-20261014-161142-v2-93f384/`；本地镜像：`runtime_refs/minicpm47-sdk-defaults-m4-off-20261014-161142-v2-93f384/`。

- 联合18请求摘要：`diagnostics/image-isolation-18-summary.json`，SHA256 `ae7137ca291a67d23c562255e286df39ed57d7f14bbd5c269b4754a74577e6e8`；可读表为同目录 `image-isolation-18-table.md`。
- 最终保护与账本审计：`completion-audit-after-isolation.json`，SHA256 `773fd1b81ebb8462eb679c3299c41190c5bb25d46367909bfa77b1c0595ea902`。
- 首轮11项：`diagnostics/six-image-isolation-20261014-163916-078718/`。result SHA256 `7618b6a0db85156a6ca49336a9b23d9ed8e6a4d6a17ac11cea48cdfe53307186`。
- 后续7项：`diagnostics/six-image-confirmation-20261014-164350-7cb9a1/`。result SHA256 `1548e21f7704354c7c6aae2d74fea1e57c149ed457bd4c4cb8e9c8e02850ff30`。
- 每批保留plan、claim、完整payload、实际wire、逐次结果、journal、physical-calls、预算前后记录和原验收哈希。
- 首轮独立审核：首轮目录的 `independent-audit-11.json`，SHA256 `1e7c3ee71d0aa25ed3203f8a5d773e1735d91f915f8010cc7185ff65f70beb34`。后续7项独立审核另存后续目录的 `independent-audit-7.json`，SHA256 `e63213eb1c0df8006015c49014734858d42b0a0f28cd10cd5b4b94851a75b357`；不改原文件。
- 第2张原图ID为 `img_77b9b408e1b6a86c87d518841430397507d5ec233d30d61118acd47431f7a93a`；原文件SHA256为 `6de56ff583ab1d30cb8c3dbfd601de735328e95b5bc0469aa8f90cc78cb65a40`。私人查看副本为本地runtime的 `visual-review/fixture-1.jpg`，编号从0起，故对应报告第2张。

[8774只读报告](http://192.168.0.100:8774/)追加本次定位结果，保留此前全部失败和原F37-Off。不改变8773、旧四组结果、原355图快照或费用事实；未向供应商客服发送工单或额外披露私人图片。

# MiniCPM 4.7：SDK默认调用与独立接入验收

观测日期：2026-10-14，Asia/Shanghai。承接[首次Thinking On接入](MINICPM47_THINKING_20261014.md)。用户随后明确授权：以官方最小SDK为基准，先显式指定Thinking，不行则省略该参数。

## 当前结论

**省略Thinking参数后，文本和真实图片调用可以返回有效答案与JSON；现有WeMM流程的能力验收仍未通过。30次稳定性验收、3题真实WeMM端到端、100题／111轮均未启动。**

显式Thinking的最小请求仅比官方示例多`chat_template_kwargs.enable_thinking=true`，HTTP200但无最终content。省略该参数后返回`Hello! How can I help you today?`。因此新配置标为**Thinking未指定／供应商默认**，不标为On或Off，不推断服务端默认采样值。

真实相册石柱图多次返回完整JSON，正确识别“三峡壩址基石”及红色文字；但后续不同模式的一次请求把“壩址”误读为“螺蛳”。原生工具有一次完整往返成功，集成探针又出现最终回答复述Schema定义；JSON决策配合json_schema则收到明确HTTP400。接口可达与一次图片理解成功，不等于整个工作流已稳定可用。

## 本轮全部调用

| 阶段 | 物理请求 | 结果 |
|---|---:|---|
| 最小SDK显式Thinking，然后省略参数 | 2 | 前者无content；后者正常文本 |
| 默认SDK真实图：schema/object/prompt | 3 | 三次完整JSON及OCR均正确；初始预标注错误已追加纠正 |
| 默认SDK原生工具直接往返 | 2 | get及FINAL_SCHEMA答案通过 |
| 新客户端原生工具集成 | 3 | 图片及get通过；final返回Schema定义，严格校验失败 |
| JSON决策＋json_schema集成 | 2 | 图片通过；决策HTTP400明确拒绝 |
| JSON决策＋json_object集成 | 1 | 结构有效，但OCR错误；未进入工具往返 |

本节统计SDK配置选择与集成阶段13次物理请求，不包含更早单列的2次健康诊断（49＋56＝105 tokens，费用0元）。这13次中12次HTTP200、1次明确HTTP400；各配置的失败和未执行步骤均保留。没有重跑相同配置挑选一次全绿，没有把reasoning替代content，也没有从解释文字中截取JSON。最后一个配置在事实验收失败后停止，未继续尝试prompt决策模式；不能声称所有可能适配均已穷尽。

### 标注纠错

旧准备稿误把原图繁体“壩”写成“坝”，还会误拒颜色“紅色”。两个独立目视复核确认原图字形，追加带哈希的review，未覆盖旧脚本输出。该错误导致最初图片发现阶段多执行了后两种模式，三次均完整计入调用记录。

30次验收启动前另外独立复核了七张原图，明确人数只计现场真人、不计墙上照片中人物，并记录粉红／玫红等合理颜色表述。新协议使用独立标注版本；旧协议和冻结结果不改。标注文件SHA256：`0f6c3984ba453b67b7fbd5b4fb1227245e63d8c118a6c5bc52d71896083548cc`。

## 已交付代码与验证

新增协议`minicpm47-sdk-defaults-m4-off-v1`，独立`request_profile=sdk-defaults-v1`、`thinking_mode=unspecified`。请求中省略thinking、temperature、max_tokens、stream字段，配置里的null只表示未指定，不发送null占位。实际SDK按非流式响应处理，TTFT不可用；实际输出上限未知。模型、256000上下文、六工具、600秒／25步／5次看图／50次请求、单请求90秒、M4、Gate-Off及原Doubao评分约束保持。

原固定Thinking On协议和Qwen默认行为保留；旧MiniCPM配置指纹不变。真实OpenAI SDK的MockTransport验证了字段确实缺席。错误／缺失型号及缺失usage即使零价仍记未知、停止后续请求；保留有限429／空404补发及明确上下文超限按轮失败继续的边界。

Mac与Linux全量回归均为**1107 passed，1 skipped**。报告针对SDK默认配置披露参数差异，3题非正式记录不使用100题分母。新旧协议只允许与哈希锁定的原F37-Off作限定比较，不放开任意跨协议比较。

冻结源码423个文件；源码包SHA256：`03978c3957ebb97e1eb28f86be2bbae27dd551d22b5d0fdf873e46ba50dc3fdc`。

独立执行release：`.100:/home/realmagic/camroll-wemm/releases/minicpm-sdk-defaults-20261014-152533-2d3b3b/`。后续文档更新不改写该冻结release。

## 独立证据与费用

以下目录均位于`.100:/home/realmagic/camroll-wemm/runtime/`，对应Mac私有回执位于`/Users/zhanyiqi/Documents/camwwem/runtime_refs/`。凭据、原始图片、完整响应和数据库不进入Git。

- 原生集成：`minicpm47-sdk-defaults-m4-off-20261014-152533-2d3b3b/`。
- JSON决策／schema：`minicpm47-sdk-defaults-m4-off-20261014-154023-json-75317b/`。
- 最终停止记录：`minicpm47-sdk-defaults-m4-off-20261014-154309-object-05b9d1/`。
- 最小调用及初探仍在原Thinking runtime的`diagnostics/`，并复制到新目录的`capability-discovery/`供审计；以call_id去重统计，不能重复记账。

各目录保存完整profile、模式选择依据、独立claim、probe-results、request-journal、原始输出、失败、价格和旧未知请求精确例外。最终目录的`completion-audit.json`及`report-deployment/`记录费用与恢复结果。

私有证据归档已同时保存到最终runtime及Mac回执目录：`evidence-archive.tar.gz`，216个文件、12798562字节，SHA256 `facedc1241a2fe809238f27279cf15839b5576e7b0d93667cfdc005637531bd4`。SQLite均通过一致性backup快照归档，包含共享账本只读快照，排除凭据；未修改共享账本或发起新调用。

15:27重新核验官方模型目录，输入／输出原始价格均0、`is_free=false`。按真实call_id关联账本，12笔已结算零价、1笔明确拒绝释放，新增未知／待结算均0。本轮已知用量28711 tokens（输入28122、输出589；reasoning9已含输出，image15270已含输入），新增费用0元；原600元账本仍已结算375.308192元，旧未知预留1.268122元，可用223.423686元。原未知请求不释放、不重发；任何新未知仍阻断。

15:47官方控制台显示20次API请求、40027 tokens（输入38896、输出1131）、消耗0.00积分、余额110.00积分。平台请求数不是成功次数；此轮含一笔HTTP400。用量对账为更早累计11211＋前置健康诊断105＋本节SDK配置测试28711＝40027 tokens。原独立3144-token探针仍仅关联账号证据，未伪造成共享账本已结算，也未猜测积分与人民币换算。

## 服务与后续边界

GPU1已按用户15:05指令释放；15:48审计GPU1仍18MiB、无计算进程，仅有既有图形进程；此轮云端验收没有启动WeMM或占用GPU。模型服务保持停止，不切GPU0或8773。相册快照仍为`bcc1cd1176ab39697c789a4f44c7426e`；原F37-Off仅只读引用，不重跑、不重评分。五份WeMM保护文件、1167份索引文件、模型指纹及相册三份核心哈希均与原交接一致。

8774继续只读，显示默认调用配置的失败与正式未启动，保留原Thinking失败及全部历史结果。首次报告切换曾因端口TIME_WAIT失败，已恢复旧报告；部署失败和恢复回执保留。15:48最终切换成功，Mac浏览器与两端HTTP均核验只读正常，保留13条历史运行、19份历史冻结文件和原失败记录。最终目录`report-deployment/deployment-result.json`状态为passed。报告回滚只更换报告进程和代码，不回退账本或结果。

后续若修正具体适配问题，必须用新的版本／runtime，并保留这些失败依据。先完成能力验收，再执行一次冻结30用例及独立逐项复核，通过后才允许真实WeMM三题及100题正式运行。当前没有允许绕过这些门槛的结果。

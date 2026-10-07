# 仅执行尚未开始的 F38-On

此入口只执行 `F38-On` 的 100 题、111 轮。前三组保留 g 批次的真实 run ID、状态和文件 SHA，按只读引用展示。F38-Off 的首轮 `context_limit`、其余未执行轮次及既有费用不补跑、不替换。

新目录固定建议使用 `/home/realmagic/camroll-wemm/runtime/photobench-flash-gate-20261007h`。新 release 必须独立发布，不覆盖 `gate-b433657`，也不改 g 或 f 目录。此脚本未自带启动计划；默认执行只显示待执行范围，`--execute-paid` 才进入监督器。

启动前门槛：

- g 的 freeze、终态 suite 和前三组 run SHA 必须匹配已独立核验的值；g 已阻断且恢复，F38-Off 恰好首轮本地 `context_limit`，F38-On 没有目录或执行 claim。
- 新 release 中 b433657 的既有文件逐字一致，仅允许已授权的报告文件 `camroll_eval/repository.py`、`camroll_eval/static/app.js` 变化；Agent、工具、Gate、Runner、模型、提示、限制及指标全部不变。
- 引用 g 已通过且恢复的真实小样及其源码/协议哈希，不新增付费小样。新 release 必须完成 Mac 与 Linux 离线回归，h 的 `validation.json` 源码清单必须精确匹配。
- 继续使用原 ¥600 账本及 g 保存的唯一旧 unknown acknowledgement；原 ¥1.268122 预留和未知请求原行保持不变，新增未知或待处理请求阻止启动。远端至少有 2 GiB 可用空间。

以下命令供发布审核通过后的主任务执行，将 `NEW_RELEASE` 替换为实际冻结 release 绝对路径；本说明本身不表示已经执行：

```bash
CAMROLL_GATE_VALIDATION_ROOT=/home/realmagic/camroll-wemm/runtime/photobench-flash-gate-20261007h \
PYTHONDONTWRITEBYTECODE=1 \
/home/realmagic/camroll-wemm/.venv-photobench/bin/python \
NEW_RELEASE/deployment/gate_validate.py

/home/realmagic/camroll-wemm/.venv-photobench/bin/python \
NEW_RELEASE/deployment/gate_remaining.py --execute-paid \
--runtime /home/realmagic/camroll-wemm/runtime/photobench-flash-gate-20261007h \
--parent-runtime /home/realmagic/camroll-wemm/runtime/photobench-flash-gate-20261007g \
--acknowledgement /home/realmagic/camroll-wemm/runtime/photobench-flash-gate-20261007g/acknowledgement.json
```

使用现有共享 GPU1 registry、新 window ID、600 秒租约和 30 秒续租；不控制 GPU0。仅启动一个 F38-On child，新的未知请求或账本上限变化写取消标记，等待已发请求结算并清理本轮进程。任何出口均尝试恢复进入前服务状态；进入前停止则结束仍停止。结束复核原模型、索引、全部输入、父记录和 WeMM 全量权重哈希。恢复未完成、清理失败或新预留未处理时，不能返回成功。

`remaining_group_complete` 只表示本次 F38-On 轮次与必要评分完整；h 的 `status=passed` 还要求恢复及账本检查合格。四组比较仍因原 F38-Off 未完成而显示 `comparison_status=incomplete`。报告导出由独立报告入口处理，不能把前三组复制成新回答。运行与归档数据不提交仓库；本地空间不足时采用独立、无损压缩流归档，禁止覆盖既有证据。

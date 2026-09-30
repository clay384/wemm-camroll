# 四组部署与恢复

仅 `.100:/home/realmagic/camroll-wemm` 独立目录运行。通用runtime：`runtime/photobench-qwen-20260930`；只读输入来自 `runtime/photobench-v2/source/album3-max`。源码release为 `releases/four-<git SHA>`，避免覆盖旧current链接及8773。

在新release根目录使用 `.venv-photobench/bin/python`：

```bash
python deployment/four_cloud_smoke.py --execute-paid
python deployment/four_suite.py
python deployment/four_suite.py --formal
python deployment/four_report.py
```

首次预算升额另加 `--raise-budget`；只能将原300改为600，后续禁止再次升额。云小样通过请求日志回放，不自动重发未知请求。四组全链路小样运行合成3图/3题/4轮；正式门槛读取 `cloud-smoke.json`、`gpu-acceptance.json`、`smoke-suite.json` 和 `validation.json` 的passed状态。validation由执行测试的操作员记录命令、版本和结果。

suite拒绝复用已有状态。正式顺序Q37→Q38→W37→W38Max，每组单独run_id和相册，期间租约自动每30秒续约。数据及全部逐题中间状态保留在runtime；`formal-suite.json`、`freeze.json`、`evaluation/comparison-profile.json`记录套件状态、冻结和报告选择。正式中断不自动补跑。小样修复后的新尝试必须先归档失败套件（含恢复收据），明确使用新run_id，不覆盖旧记录。

`windows/<window_id>/state.json`和`receipt.json`记录进程身份、GPU初始/结束状态、旧文件校验、模型40文件哈希、600秒租约与恢复结果。独立watchdog在owner退出或租约到期时恢复；无法验证进程身份时拒绝发信号。人工恢复命令须使用刚读取的window_id和匹配release：

```bash
python deployment/four_gpu_window.py status --window-id WINDOW_ID
python deployment/four_gpu_window.py restore --window-id WINDOW_ID
```

不复用历史PID，不向GPU0发送控制指令。`four_report.py`直接绑定8774；端口占用立即退出。报告不加载凭据、不挂Agent或评测写接口，可独立于GPU长期运行。预留/未知请求须查原账本与request-journal.sqlite；不得删库、清预留或降低已消费金额来“恢复”。

报告可使用独立 `releases/report-<git SHA>`，不得覆盖正在执行的冻结 `four-<git SHA>`。`python deployment/four_export.py` 只读读取结果和原共享账本，生成私有 `reports/four-models.json`、`.md` 及 `evaluation/cost-audit.json`；`--smoke` 导出合成小样。费用按本run独立相册的全部建库轨迹、问答轨迹、Judge轨迹的唯一请求ID归属，另列共享账本同期增量，避免其他项目并发消费混算。未完整轨迹、缺失ID或不可用账本保留未知总额和已知小计。报告页读取费用审计覆盖层，不改写冻结run.json。实际套餐账单不可用时不将等价估算标为实付。

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

## 充值恢复后的独立64题补测

2026-09-30用户确认账户欠费导致W38Max的HTTP403，充值后明确选择沿用本组快照，只补测受影响64题并单独报告。原四组套件、报告和记录不改写；补测为64题/71轮/52道可回答题，不作独立冷建库，也不合并替换原100题得分。

补测前使用新请求namespace完成qwen3.8-max工具/图片/结构化输出小样，保留独立probe JSON；共用原600元账本，不再次抬高上限。使用包含独立adapter的新release，付费Agent/Judge核心文件必须与原8874328冻结清单一致。监督器仍使用GPU1、600秒租约和30秒续租，进入前后状态均核验。

```bash
python deployment/four_supplement.py \
  --parent-id w38max-formal-20260930-122300-813923 \
  --run-id NEW_UNIQUE_SUPPLEMENT_ID \
  --probe /home/realmagic/camroll-wemm/runtime/photobench-qwen-20260930/supplements/RECOVERY_PROBE.json
python deployment/four_supplement_export.py --run-id NEW_UNIQUE_SUPPLEMENT_ID
```

这是显式授权的一次新补测，不能把命令用作自动重试。已存在run或同组选题claim时拒绝重复。输入仍核验原355图/100题，选题仅从父run结构化http_403调用确定；64题各从第一轮开始，保留本题多轮上下文，不继承首跑答案。原快照复制到独立pinned_memory，旧state.sqlite不复制，父run和快照前后哈希核对。Qwen/Judge在新run目录中使用新journal；再遇403停止新题，已投评分收尾，未知请求保留预留并停止。

新监督器证据在`runtime/supplements/<run_id>/supervisor.json`，窗口ID与run_id一致；补测记录在`evaluation/runs/<run_id>/`。独立费用覆盖层为`evaluation/supplement-cost-audit.json`，报告为`reports/<run_id>.json/.md`，不改原`cost-audit.json`与`four-models.*`。费用仅按补测请求ID归属，原建库/首跑/充值小样费用分列。8774继续只读展示原四组和独立补测记录。

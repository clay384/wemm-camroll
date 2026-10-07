# WeMM + CamRoll 工作约定

本仓库是两个既有项目的独立源码备份与后续开发仓库。先读 README.md、docs/ARCHITECTURE_CURRENT.md、docs/CODE_MAP.md 和 docs/releases/2026-10-07-architecture-v1.md。RESTORE.md 与 baseline/source_manifest.json 只针对对应的2026-09-16基线；`docs/project/` 是原样保存的历史交接及方案，当前用户授权优先于历史记录。

- 云端初次阅读见 docs/CLOUD_READING.md；读取架构不需要权重或私人运行数据。不要为理解项目而直接运行付费、建库或部署入口。
- 当前Flash Gate仅含3.7/3.8 Flash，原600元累计账本不自动提额。2026-10-07最新实验未完整交付；架构版本发布不改变冻结模型、提示、评分、停止规则或任何旧结果。

- `camroll/` 对应 `.100:/home/realmagic/camroll-wemm`；`wemm/` 对应 `.100:/home/realmagic/wemm-embedding-lab`。不要混淆两个根目录。
- 基线标签 `baseline/2026-09-16-pre-photobench` 不移动、不覆盖。后续修改使用新的提交、实验相册和 run_id。
- 绝不提交模型权重、私人照片/身份参考图、QA/GT、运行报告、原始向量、数据库、凭据或原 Git 历史。
- `.153` 的 Sentrix 原始资料、业务目录及 `http://192.168.0.153:8771/` 页面保持只读。评测改动与结果页面使用独立项目。
- WeMM 仅使用既有物理 GPU 1，UUID `GPU-4ef9200c-d8f3-638a-3eeb-26fe4fa3d7b0`；不控制 GPU 0，不终止其他任务，不启动第二份模型。
- 保留 WeMM encoder、worker、预处理、模型配置、旧数据集与冻结结果。部署前后核对指纹及旧向量/报告哈希。
- 继续使用已有累计云费用账本；回滚代码或相册不回滚费用事实。真实配置、密钥和运行数据由服务器本地独立保管。
- 状态注明观测日期。源码备份、历史验收和实时健康状态分别记录，不相互替代。

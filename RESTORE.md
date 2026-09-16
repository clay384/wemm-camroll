# 当前基线恢复说明

基线：`baseline/2026-09-16-pre-photobench`。本说明是恢复路径，未执行恢复操作；服务状态、数据快照与校验收据由独立运行备份保管。

## 恢复前必须具备的材料

1. 此标签的源码及 `baseline/source_manifest.json`，并校验文件 SHA256。
2. `.100` 上改动前独立保留的 CamRoll 相册 SQLite、阶段缓存、资产目录、全部已发布快照及 `CURRENT` 指针。SQLite 必须使用一致性备份，不能仅在写入期间复制主文件而遗漏 WAL。
3. 当前服务配置、受限凭据、进程参数以及已有 Python 环境或对应环境锁。凭据由部署主机本地保存，不通过 GitHub 恢复。
4. 既有 WeMM 数据、向量、模型 revision 和指纹的校验记录，以及授权 GPU UUID 的实时核验。现有权重从原部署路径复用，或按固定 revision 重新取得，不在 Git 仓库中。
5. 当前累计费用账本。费用账本不可回退到旧额度、删除或换路径重置；恢复代码和相册时仍应保留最新累计费用事实。

## 恢复顺序

1. 停止新实验提交请求，保留失败版本、日志及新 run 结果。核对当前部署是否有其他人的后续修改，禁止无条件覆盖。
2. 从此标签检出到独立恢复目录，按 source manifest 比较 `camroll/` 与 `wemm/` 文件。仓库的两个子目录分别对应原服务器的两个项目根，不可把整个单体仓库覆盖到其中一个根。
3. CamRoll 新功能仅使用独立相册和数据库。回退时恢复原 CamRoll 代码与原配置，接回改动前的相册路径及经过验证的 `CURRENT`；不把新 schema 数据库直接交给旧代码。根据原部署管理器的进程身份校验受控切换 CamRoll 服务。
4. 只有 WeMM API 源码本轮确实改过且需要回退时，才恢复本标签的 `wemm/backend/app.py` 与 `wemm/backend/camroll_internal.py`，并核对其他文件仍为预期版本。原 encoder、worker、preprocess、config、权重和旧索引不应随 CamRoll 功能回滚而改变。
5. WeMM 若无需改动，保持既有 worker 运行。确需受控重启时，只在授权 GPU 1 可用且既有服务身份、队列状态已核对后进行；不得控制 GPU 0、终止其他任务或启动第二份 WeMM。
6. 对比源码、WeMM 指纹、旧资产/向量及冻结报告哈希，检查原相册快照可读取、原端口和服务参数正确，再执行获授权的必要功能验证。将观测时间与验证结果保存到新的恢复 run，不能用历史健康记录替代实时验证。

## 源码校验

在仓库根执行下列命令仅校验备份源码，不连接服务、不修改索引，也不调用模型：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('.')
manifest = json.loads((root / 'baseline/source_manifest.json').read_text())
for item in manifest['files']:
    path = root / item['path']
    assert path.is_file(), item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['path']
print(f"Verified {len(manifest['files'])} source files")
PY
```

`camroll/docs/DEPLOYMENT.md` 保留原服务运维细节。其中早期安装脚本和旧 before 快照是历史变更记录，不是此次当前基线恢复的快捷方式；已有 API 扩展不能重新盲装。

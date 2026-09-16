# 调用、复跑与运维

**以下对服务器的操作均是未来获得相应授权后的参考命令。本次交接不执行，因为用户禁止修改 `.100`。** 查询会写日志/数据库，启停脚本可能写运行状态，不能视为无文件副作用。

## Python 登录与检索

在可访问局域网的 Mac，使用已有 requests 环境。密钥从本机受限文件读取，不写入代码，不打印 cookie。

```python
from pathlib import Path
import requests

base = "http://192.168.0.100:8765"
s = requests.Session()
s.trust_env = False  # 局域网请求不走环境代理
key = Path("/Users/zhanyiqi/Documents/nx/wemm-embedding-lab/run/access.key")
r = s.post(base + "/api/login", json={"password": key.read_text().strip()}, timeout=15)
r.raise_for_status()

r = s.post(base + "/api/search", json={
    "dataset": "ad8f153d865a4d0caa5d1704b0874426",
    "text": "在停车场的照片", "kind": "image",
    "k": 50, "dimension": 4096, "aggregate": True,
}, timeout=180)
r.raise_for_status()
result = r.json()
for hit in result["results"]:
    print(hit["rank"], hit["asset"], hit["name"], hit["score"])
```

混合候选测试删除 kind 字段；不根据答案选择 kind。图搜图先上传查询图片到独立查询数据集并等待 indexed，再传 image_asset。图文联合查询同时传 text 和 image_asset。当前没有直接上传视频作为查询的 API。

## 创建与上传（写操作）

沿用上面的登录 Session：

```python
r = s.post(base + "/api/datasets", json={"name": "新的独立实验"}, timeout=15)
r.raise_for_status()
dataset_id = r.json()["id"]
with open("/absolute/path/photo.jpg", "rb") as f:
    r = s.post(base + "/api/upload",
               params={"dataset": dataset_id, "name": "photo.jpg"},
               data=f, timeout=180)
r.raise_for_status()
print(r.json())  # 注册完成不等于编码完成
r = s.get(base + "/api/assets", params={"dataset": dataset_id}, timeout=15)
r.raise_for_status()
# 核对每个资产 status == indexed；保留 failed/error，不跳过宣称成功。
```

目录导入通过 `/api/import-directory` 提交 dataset 和相对 import 根目录的 path。PhotoBench 分别导入 `photobench-20260914/album3/photos`、`photobench-20260914/album3/videos`，不导入 QA 父目录。网页流程：登录 → 数据管理 → 创建数据集 → 选择文件夹 → 等待入库 → 检索选择数据集。

## 运维命令（参考，不在本轮执行）

```bash
ssh realmagic@192.168.0.100
cd /home/realmagic/wemm-embedding-lab
./scripts/start.sh
./scripts/status.sh
./scripts/logs.sh
# 仅在需要停止/恢复本服务时：
./scripts/stop.sh
./scripts/recover.sh
```

这些命令逐条按需要使用，不要整段顺序执行。recover.sh 实际调用 manage.py restart。管理器校验本项目进程身份；仍应先核对 PID、工作目录、GPU UUID，不使用广泛 pkill，也不控制 GPU 0 任务。

启动前检查 GPU 1 空闲显存、模型路径、配置备份。启动可能在 8765 冲突时寻找 8766–8785 并更新配置，必须核对实际地址。服务没有 systemd/开机自启保证。恢复会处理被中断资产，指纹改变可能使旧索引重新编码；不要为了查看状态随意重启。

## PhotoBench 固定实验复跑

实现入口：`scripts/photobench/run.py`；原运行 ID 为 `20260914-115425`。阶段包含 prepare、verify、selftest、ingest、evaluate、export。新实验必须使用新 ID，源目录应在所有阶段保持一致。流程是为 album3 编写，内有固定计数与选择规则，不是任意 QA 数据集的通用框架。

```bash
# 在 .100 项目目录；未来授权执行时，先指定新的 run_id
.venv/bin/python scripts/photobench/run.py prepare --run-id NEW_RUN_ID
.venv/bin/python scripts/photobench/run.py verify --run-id NEW_RUN_ID
.venv/bin/python scripts/photobench/run.py selftest --run-id NEW_RUN_ID
# 完成下面的离线改写并复制、冻结输出后，再继续：
.venv/bin/python scripts/photobench/run.py ingest --run-id NEW_RUN_ID
.venv/bin/python scripts/photobench/run.py evaluate --run-id NEW_RUN_ID
.venv/bin/python scripts/photobench/run.py export --run-id NEW_RUN_ID
```

prepare 不覆盖已有 freeze。可用 `--source /absolute/path` 指定快照；各阶段保持同一参数。ingest 会等现有 worker 就绪。evaluate 逐行落盘，以题目/轨道/组别跳过已完成项，检查冻结信息后续跑；不要手工删除失败行来提高指标。export 也会写报告，不是只读动作。

离线改写在 `.153` 独立目录，以其现有 Python 执行复制过去的 rewrite.py：

```bash
/home/asus/miniconda3/envs/sentrix-vllm/bin/python rewrite.py \
  --input questions_only.jsonl --out output \
  --model /home/asus/models/Qwen3-4B-Instruct-2507 \
  --wemm-tokenizer tokenizer.json \
  --gpu GPU-930c71f5-9fed-cfdd-1977-1b4b8e4e1cc5
```

只提供 qa_id/question 和验证用 tokenizer；不提供答案、证据、事件或检索结果。沿用固定提示词、BF16、greedy、batch=1、最多 512 新 token、最多重试一次。把 output 中 rewrites.jsonl、rewrite_lock.json、rewrite_runtime.json、rewrite_complete.json 复制至新 run 的报告目录并按脚本冻结校验，然后才评测。完整原流程参见项目 `scripts/photobench/README.md`。

`acceptance.py --restart` 会实际重启服务，并沿用原 bench 默认 run；不能假设传入新 run.py ID 就会自动切换此脚本。旧 `/api/benchmark` 和 run_benchmark.sh 属于 demo，不用于 PhotoBench。

## 查看报告

直接打开 Mac 文件：`/Users/zhanyiqi/Documents/nx/wemm-embedding-lab/reports/photobench/20260914-115425/report.html`。报告依赖同目录 report-data.js、report.js、report.css、media/，交付时须一起保留。

若浏览器本地文件策略影响预览，可仅在 Mac 启动静态服务：

```bash
python3 -m http.server 8877 --bind 127.0.0.1 --directory /Users/zhanyiqi/Documents/nx/wemm-embedding-lab/reports/photobench/20260914-115425
```

然后打开 http://127.0.0.1:8877/report.html 。此方式不调用 `.100`，结束时 Ctrl-C。不要把历史本地预览端口当成当前已启动服务。

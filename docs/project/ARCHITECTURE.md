# 架构与文件结构

## 主机职责

| 位置 | 职责与路径 |
|---|---|
| Mac | 本次交接 `/Users/zhanyiqi/Documents/camwwem`；源码及报告副本 `/Users/zhanyiqi/Documents/nx/wemm-embedding-lab` |
| `.100` / realmagic | `/home/realmagic/wemm-embedding-lab`：FastAPI、前端、预处理、唯一 WeMM worker、SQLite 和向量文件 |
| `.153` / asus | Sentrix PhotoBench 原始媒体与 QA；独立 Qwen 离线改写任务，任务已于 9 月 14 日完成退出 |

`.100` 物理 GPU 1 UUID 为 `GPU-4ef9200c-d8f3-638a-3eeb-26fe4fa3d7b0`。GPU 0 留给原有任务。通过 CUDA_VISIBLE_DEVICES 绑定后，进程内部显示 cuda:0 正常。

## 固定模型与限制

- 模型：`tencent/WeMM-Embedding-9B`，revision `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482`。
- 模型目录：`models/WeMM-Embedding-9B`；BF16、4096 维、batch=1，无量化和 CPU offload。
- 9 月 14 日环境记录：PyTorch 2.10.0+cu128、Transformers 5.2.0、qwen-vl-utils 0.0.14；实际环境以项目锁定记录为准。
- 图片模型预算 262,144 pixels；视频帧预算 65,536 pixels，每 8 秒切片采样 4 帧；文本最多 512 token。
- 上传单文件 128 MiB；源图解码上限 45,000,000 pixels；视频最多 300 秒；PDF 最多 100 页。
- 文本读取上限 2 MiB / 200,000 字符；预处理超时 60 秒；导入队列限制 500。
- 45M 是源图解码限制，不是模型像素预算。API 的 1024/256 维选项截断后重新归一化，不减少模型加载显存。

## 数据流

```text
浏览器 / Python 客户端 → 登录 cookie → FastAPI
  上传或受限目录导入 → SHA256 去重 → dataset/asset 映射
  → CPU 媒体预处理 → 单 WeMM worker → 归一化向量 .npy + SQLite units

文本 / 已入库查询图片 → WeMM 查询编码
  → 按数据集、有效指纹、可选 kind 选择候选
  → 精确点积排序 → 同一源资产取最高单元分数 → Top-K
```

图片执行 EXIF 方向处理和 RGB 转换，生成派生图与缩略图。PDF 按页渲染后做视觉编码，不使用 OCR。视频转码后原生视频编码，忽略音频；短尾片段不足 4 帧时重复末帧至 4 帧。文本按 token 预算分块。后台单 worker 处理查询和入库，避免多个模型副本抢占显存。

## 项目目录

```text
wemm-embedding-lab/
├── config.yaml                 # 端口、GPU UUID、模型、预算
├── SOURCE_LOCK.json            # 来源锁定记录
├── backend/
│   ├── app.py                  # 路由、认证、静态前端
│   ├── common.py               # 配置、SQLite、目录、事件
│   ├── worker.py               # 队列、编码调度与恢复
│   ├── encoder.py              # 模型加载、编码、指纹
│   ├── preprocess.py           # 图片/PDF/文本/视频预处理
│   ├── retrieval.py            # 精确向量检索与源资产聚合
│   └── benchmark.py            # 旧 demo 固定评测
├── frontend/{src,dist}/        # 前端源码与构建结果
├── models/WeMM-Embedding-9B/   # 权重与模型文件
├── official-locked/            # 官方来源代码
├── data/
│   ├── lab.sqlite              # 数据集、资产、单元、事件
│   ├── originals/<sha256>      # 原始字节，无扩展名
│   ├── derived/<sha256>/       # 页面、视频片段、图片缩略图
│   ├── indexes/<unit_id>.npy   # 单元向量
│   └── tmp/
├── import/                     # API 目录导入的允许根目录
├── run/{access.key,service.json,model.lock}
├── logs/service.log
├── scripts/                    # 启停、验收、打包、模型下载
│   └── photobench/             # 独立 CLI 与报告生成
├── reports/                    # 旧 demo 报告（不可覆盖）
│   └── photobench/20260914-115425/
├── deliveries/                 # 旧源码包与公开 demo 包
├── CURRENT_STATE.md
└── WORKLOG.md
```

`common.py` 导入时会创建目录；数据库连接使用 WAL。不要为只读检查在远端直接 import backend。运行目录中的 PID、锁、访问密钥不能当成可移植配置照搬。

## 数据标识与状态

| 表 | 主要含义 |
|---|---|
| datasets | id、名称、paused |
| assets | id=原始文件 SHA256；扩展名、大小、状态、错误与预处理耗时 |
| members | dataset/asset/name 关联；同一字节内容可以有多个名字或属于多个数据集 |
| units | 单元 ID、asset、kind、路径、缩略图、meta、向量、fingerprint、stats |
| annotations | query/unit/label 人工标注 |
| events | 查询等事件日志 |

资产状态：pending → preprocessing → encoding → indexed；异常为 failed，取消为 cancelled。暂停属于数据集标记。资产状态与向量由共享资产持有，操作一个数据集时要考虑同 SHA 资产在其他数据集中的复用。

模型及预处理指纹用于防止不同管线向量混用。9 月 14 日有效指纹：
`cb0b50fe0c768df7ec200dcc15160643ce11660e0ffea39e48eafc451e6a4cf7`。

检索读取匹配指纹的向量，计算归一化点积；并列时按 asset、unit ID 稳定排序。aggregate=true 时每个源资产取最高单元分数，其他片段放在 children。不是 FAISS/ANN 索引，目前为精确全量比较。

## 原始 QA 与改写目录

`.153` 数据源：`/home/asus/Github/Sentrix-Home-Web/services/photobench/data/album3-max-video10/`。

- photos/：334 JPG；videos/：10 MP4。
- qa/：image-related-1115q.jsonl、single-video-52q.jsonl、mixed-image-video-1167q.jsonl、summary.json。1167 是合并集合，不要把三个文件行数相加。
- metadata/：图片元数据及视频源图映射；identity/、faces/：身份参考资料。本轮均未进入向量索引。
- 顶层 dataset_audit.json 存在旧计数，QA 计数应核对 qa/summary.json 和冻结选题文件。

本轮所需快照位于 `.100` 的 `import/photobench-20260914/album3/`，仅分别导入 photos/、videos/。视频为基于照片构造的短基准素材，不能替代真实长视频动作质量评测。

独立改写目录：`.153:/home/asus/wemm-photobench-rewrite/20260914-115425`；模型 `/home/asus/models/Qwen3-4B-Instruct-2507`；Python `/home/asus/miniconda3/envs/sentrix-vllm/bin/python`。它是离线程序，不是常驻 router API。

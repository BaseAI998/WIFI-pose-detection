# 基于关键点的单帧姿态分类

Version: 2.0

将 COCO17 二维关键点转换为逐人、逐帧的 `sitting`、`standing` 或 `unknown`。
项目提供 NumPy 推理、人工标注、CPU 训练评估和视频可视化。

## 安装

使用 Python 3.11，在项目根目录执行：

~~~powershell
conda env create -f posture_mlp_v2/environment.yml
conda activate posture-mlp-v2
python -m pip install -e "./posture_mlp_v2[render]"
~~~

按用途安装依赖：

| 用途 | 依赖 |
|---|---|
| 关键点分类 API | NumPy |
| 标注台、视频处理 | NumPy、OpenCV |
| 模型训练、完整 Python 测试 | NumPy、OpenCV、PyTorch CPU |
| 前端回归测试 | Node.js |

训练环境追加安装：

~~~powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
~~~

## 项目结构

~~~text
posture_mlp_v2/
  inputs/
    README.md                输入格式
    example/                 原视频＋关键点JSONL
  src/posture_mlp_v2/         算法源码与cli命令入口
  configs/                   训练与推理配置
  pyproject.toml             包配置和分组依赖
  runs/                      实验配置、模型、指标
  outputs/
    README.md                输出格式
    example/                 与inputs/example一一对应的结果
  tests/                     测试
posture_labeler/
  server.py                  标注服务
  static/                    网页界面
  data/annotations.sqlite3   标注数据库
  exports/                   冻结标注快照
  tests/                     通用测试与历史批次核验
  hard_samples/              队列清单、active.json批次选择与审计
~~~

文档：[输入格式](posture_mlp_v2/inputs/README.md) · [算法模块](posture_mlp_v2/README.md) ·
[标注台](posture_labeler/README.md) · [模型评估报告](posture_mlp_v2/runs/hard_v2_complete/REPORT.md)。

## 分类 API

以下示例读取一个人物并执行分类：

~~~python
import json
from pathlib import Path
from posture_mlp_v2.model import PostureClassifier

model = PostureClassifier("posture_mlp_v2/runs/hard_v2_complete/model")

with Path("posture_mlp_v2/inputs/example/VID_20260921_182630_keypoints.jsonl").open(
    encoding="utf-8"
) as stream:
    frame = next(row for line in stream if (row := json.loads(line))["people"])

result = model.predict(frame["people"][0]["keypoints"])
print(result["label"])
print(result["accepted"])
print(result["rejection_reasons"])
~~~

`predict(keypoints)` 接受 COCO 顺序的 `(17, 3)` 数组，每行为 `[x, y, confidence]`；
也接受由17个 `{index, x, y, confidence}` 字典组成的列表。

坐标使用原图像素，置信度范围为 `[0, 1]`。缺失关节保留索引，以低置信度表示。
批量调用 `model.predict_many([person1_kps, person2_kps, ...])`，结果与输入顺序一致。

| 输出字段 | 含义 |
|---|---|
| `label` | 最终姿态：sitting、standing、unknown |
| `candidate_label` | 二分类候选姿态 |
| `probabilities` | sitting、standing 的分类分数 |
| `confidence` | 二分类最高分 |
| `accepted` | 是否通过拒识检查 |
| `rejection_reasons` | 触发拒识的原因 |

业务判断使用 `label` 和 `accepted`；分类分数用于分析候选姿态。
部署时携带模块源码及模型目录中的 `model.json`、`weights.npz`。
模型适用范围和指标见评估报告；新场景上线前使用独立视频评估。

## 绘制视频

每个输入目录包含一个原视频和一份关键点 JSONL：

~~~powershell
python -m posture_mlp_v2.cli.render --input-dir "posture_mlp_v2/inputs/my_video" --model-dir "posture_mlp_v2/runs/hard_v2_complete/model"
~~~

程序读取视频尺寸、FPS和总帧数，按关键点中的原视频帧号绘制骨架与姿态。
默认输出到 `posture_mlp_v2/outputs/my_video/`，与输入文件夹同名。
先创建inputs/my_video并放入视频和JSONL；现有example输入和结果无需重新运行。
已有结果不会自动覆盖。`--output-dir`可显式指定其他结果目录；
加 `--skip-video` 可仅导出逐帧预测 JSONL。

## 人工标注

~~~powershell
python posture_labeler/server.py
~~~

打开 http://127.0.0.1:8765 。默认输入为 `posture_mlp_v2/inputs/example/`，
标签保存到 `posture_labeler/data/annotations.sqlite3`。

| 标签 | 标注标准 | 用途 |
|---|---|---|
| `sitting` | 确认坐在支撑物上 | 二分类训练 |
| `standing` | 确认站立，包括弯腰站立 | 二分类训练 |
| `uncertain` | 单帧支撑关系有歧义、起坐过渡等情况 | 拒识参考及评估 |
| `ignore` | 误检、严重错框、重复框 | 标注复核 |

姿态和遮挡程度分别标注。每个数据集使用独立数据库。
操作、导出及备份见[标注台说明](posture_labeler/README.md)。

## 训练与评估

~~~powershell
python -m posture_mlp_v2.cli.train --input-dir "posture_mlp_v2/inputs/example" --old-labels "posture_mlp_v2/inputs/example/labels/baseline.jsonl" --hard-labels "posture_mlp_v2/inputs/example/labels/hard_v1.jsonl" --output-dir "posture_mlp_v2/runs/my_experiment"
~~~

`--old-labels` 为基础标注，`--hard-labels` 为包含基础标注和新增样本的完整导出。
两份标注均与输入视频、关键点的内容哈希绑定。

当前训练协议面向60秒片段：A=`[0,17)`、B=`[23,37)`、C=`[43,60]`，
块间保留6秒缓冲。三轮轮换训练、校准及留块开发验证；最终模型使用A+C训练、B校准。
其他片段长度在 `data.py` 中配置分割协议。

模型选择依据开发评估，最终验收使用单独保留的视频。训练参数、指标和产物说明见
[算法模块文档](posture_mlp_v2/README.md)。

## 测试与核验

~~~powershell
python -m unittest discover -s posture_mlp_v2/tests -v
python -B -m unittest discover -s posture_labeler/tests -p "test_*.py" -v
node posture_labeler/tests/test_hard_queue.cjs
~~~

核验随项目提供的结果：

~~~powershell
python -m posture_mlp_v2.cli.verify --run-dir "posture_mlp_v2/runs/hard_v2_complete" --output-metadata "posture_mlp_v2/outputs/example/VID_20260921_182630_run_metadata.json"
~~~

核验包括输入哈希、训练参考库隔离、帧及关键点一致性、批量预测一致性和完整视频解码。

# 单帧姿态分类 · MLP v2

复用COCO17关键点，逐人逐帧输出 sitting / standing / unknown。
只训练/运行姿态分类，不重新执行YOLO。现有模型仍是实验模型，没有新增独立测试集。

## 目录

~~~text
posture_mlp_v2/
├── README.md                    安装、目录与使用说明（唯一主导航）
├── pyproject.toml               包配置、命令入口与分组依赖
├── environment.yml              Conda环境定义
├── configs/
│   ├── train.yaml               输入、标注、输出位置和随机种子
│   └── inference.yaml           输入、模型、输出和视频开关
├── src/posture_mlp_v2/
│   ├── features.py              关键点特征与标准化
│   ├── model.py                 NumPy分类与拒识API
│   ├── data.py                  输入校验、时间分块、历史路径解析
│   ├── training.py              三种轻量模型的训练
│   ├── calibration.py           保守校准与拒识策略选择
│   ├── evaluation.py            分类、覆盖率和分组评估
│   ├── paths.py                 数据项目定位，不依赖源码安装位置
│   └── cli/                    train/render/report/review/verify入口
├── tests/                       单元与集成测试
├── inputs/
│   ├── README.md                输入格式
│   └── example/                 原视频、关键点JSONL、labels/标注快照
├── outputs/
│   ├── README.md                输出格式
│   └── example/                 预测JSONL、叠加视频、metadata和QA
└── runs/
    └── hard_v2_complete/
        ├── model/               model.json、weights.npz、training_info.json
        ├── folds/               训练/校准/留块开发验证记录
        ├── provenance/          历史来源、迁移映射与哈希清单
        ├── label_review/        标签复核图片
        ├── REPORT.md            实验报告
        └── *.json / *.jsonl     审计、分块、评估与预测记录
~~~

inputs/<名称>/ → outputs/<名称>/ 是数据配对；runs/<实验名>/ 是训练实验。
同一模型可以处理多份输入，不必逐视频重训。模型只保存在runs/<实验名>/model/，不另维护重复权重。

## 安装

下列命令均从本目录posture_mlp_v2/执行。依赖统一定义在pyproject.toml。

~~~powershell
conda env create -f environment.yml
conda activate posture-mlp-v2
python -m pip install -e ".[render]"
~~~

环境已存在则直接激活并安装项目。-e为可编辑安装，源码修改即时生效。

| 用途 | 安装项 |
|---|---|
| 纯关键点分类API | python -m pip install -e .，仅NumPy |
| 视频处理、YAML配置 | 安装.[render]，增加OpenCV和PyYAML |
| CPU训练、全部测试 | 先安装CPU版PyTorch，再安装.[train] |

~~~powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[train]"
~~~

推理不需要PyTorch。自动测试使用Python自带的unittest，无额外测试框架。

## 使用已有模型

~~~python
from posture_mlp_v2.model import PostureClassifier

model = PostureClassifier("runs/hard_v2_complete/model")
# keypoints: COCO顺序(17, 3)数组，或17个关键点字典
result = model.predict(keypoints)
results = model.predict_many([keypoints])
~~~

业务读取label、accepted、rejection_reasons；probabilities和confidence是二分类分数，不是可靠性保证。
部署需要安装包及model.json、weights.npz，两者必须匹配；training_info.json用于核验，不参与普通推理。
251维特征包含归一化坐标、置信度、缺失mask、几何及可见程度，不使用绝对座位位置。
拒识检查包括骨架质量、镜像一致性、分数、训练支持、uncertain邻域及类别证据冲突。

## 处理新输入

先在inputs/my_video/放入一个原视频和一份关键点JSONL，再运行：

~~~powershell
python -m posture_mlp_v2.cli.render --input-dir inputs/my_video --model-dir runs/hard_v2_complete/model
~~~

默认生成outputs/my_video/。位置由输入所在的inputs/推导，不从源码安装位置推导。
非标准输入位置可指定--project-root或--output-dir；非空输出目录拒绝覆盖。
--skip-video只生成预测和metadata，不创建空QA目录。

也可以复制并编辑[推理配置](configs/inference.yaml)：

~~~powershell
python -m posture_mlp_v2.cli.render --config configs/inference.yaml
~~~

示例配置指向已有outputs/example/，直接运行会触发覆盖保护。应改用新数据名称或显式指定新输出目录。
配置中的project_root相对于YAML文件，其余配置路径相对于该根目录；命令行参数优先，命令行路径相对于当前工作目录。
--no-skip-video可覆盖YAML中的skip_video: true。未知配置字段会报错，不会静默忽略。

输入/输出字段详见[输入格式](inputs/README.md)和[输出格式](outputs/README.md)。

## 重新训练

~~~powershell
python -m posture_mlp_v2.cli.train --config configs/train.yaml
~~~

默认使用inputs/example/labels/baseline.jsonl和hard_v1.jsonl，输出到新的runs/my_experiment/。
这两份标注是原标注台导出的字节级快照；标注台的原工作文件未移动或修改。
hard_v1必须包含基础标签及新增样本；训练前会校验标签与视频、关键点的内容身份。
后续标注按版本导出到inputs/<名称>/labels/，避免覆盖已用于训练的快照。

本次重构没有改变训练算法：配置仅暴露运行参数与随机种子。
模型族和训练超参数仍在training.py，时间块在data.py，校准与拒识规则在calibration.py。
程序将实际训练协议保存在每次实验的protocol.json和训练记录中。

当前模型族：32/16 MLP、强正则16/8 MLP及线性分类器；校准比较T=1和T≥1。
60秒片段分为A [0,17)、B [23,37)、C [43,60]，块间保留6秒缓冲。
三轮轮换训练、校准、留块开发验证，最终用A+C训练、B校准；这不是独立测试成绩。
起坐过渡或支撑关系不明标uncertain，确认弯腰站立仍标standing，误检/重复框标ignore。

## 报告、复核与核验

~~~powershell
python -m posture_mlp_v2.cli.report --run-dir runs/hard_v2_complete
python -m posture_mlp_v2.cli.review --run-dir runs/hard_v2_complete
python -m posture_mlp_v2.cli.verify --run-dir runs/hard_v2_complete --output-metadata outputs/example/VID_20260921_182630_run_metadata.json
python -B -m unittest discover -s tests -v
~~~

安装后也可使用posture-train、posture-render、posture-report、posture-review、posture-verify。
旧命令python -m posture_mlp_v2.render等已改为python -m posture_mlp_v2.cli.render；分类API导入不变。

核验包括来源哈希、训练参考库无泄漏、检测框/关键点和预测一致性、完整视频解码。
历史审计与输出metadata保留原字节，通过runs/hard_v2_complete/provenance/input_example_legacy/relocations.json解析迁移路径。
原archive/已归入实验的provenance/，不是删除历史来源；现有实验与视频没有重训或重绘。
模型表现与适用范围见[实验报告](runs/hard_v2_complete/REPORT.md)。

## 清理边界

- __pycache__/、.pyc、测试/构建缓存可删除，之后运行或安装可能重新生成。
- 空目录可以清理，但不删除仍包含数据的inputs、outputs或runs。
- 原STRUCTURE.md合并到本文，依赖清单统一到pyproject.toml。
- provenance/、folds/、模型和评估记录不是缓存，不作为垃圾删除。
- 大视频和JSONL保留，目录整理不丢弃训练证据或交付结果。

2026-10-22整理验证：49项算法模块测试、29项标注台测试通过；83个既有数据/模型/实验文件哈希不变，5394帧视频完整核验通过。
缓存删除命令被执行策略拦截，仍留有根目录及tests/下的__pycache__/（共15个.pyc，约182 KiB），以及迁移后空的archive/。
这三处可以手动移入回收站。本轮没有绕过策略删除它们，详见[迁移记录](runs/hard_v2_complete/provenance/layout_migration.json)。

## 设计参考

- [LAViTSPose](https://pmc.ncbi.nlm.nih.gov/articles/PMC12732004/)：遮挡感知思路。
- [SitPose](https://arxiv.org/abs/2412.12216)：轻量分类思路；不直接套用三维骨架的可分性结论。

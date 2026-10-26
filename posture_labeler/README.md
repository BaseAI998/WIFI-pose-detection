# 姿态标注台

本地网页标注工具，读取原视频和COCO17关键点JSONL。
后端采用Python HTTP、SQLite和OpenCV，默认输入位于 `posture_mlp_v2/inputs/example/`。

环境安装见[项目README](../README.md)。以下命令从项目根目录执行。

## 目录与数据边界

~~~text
posture_labeler/
├── README.md / .gitignore
├── server.py                 网页后端、标注保存与队列接口
├── mine_hard_samples.py       困难样本筛选与批次激活
├── start.cmd / start.ps1      Windows启动入口
├── static/                   HTML、JavaScript、CSS
├── data/
│   └── annotations.sqlite3   正式数据库，保留标注和操作历史
├── exports/                  冻结标注快照，不与数据库自动同步
│   ├── baseline.jsonl        原255条标注
│   ├── hard_v1.jsonl         原255条＋第一批96条困难标注
│   └── baseline.csv          便于人工查看的基础快照
├── logs/                     历史运行日志，可在停止服务后清理
├── tests/                    通用测试；历史批次核验单独运行
└── hard_samples/
    ├── active.json           只记录当前批次名，不复制目标清单
    ├── batch_20261019_v1/    queue.json、审计、安装前备份
    └── _ui_snapshots/        历史截图归档，不参与运行
~~~

2026-10-22整理时，数据库含352条标注，hard_v1快照为351条；数据库没有被导出快照覆盖。
两份JSONL和CSV只移动/改名，内容与SHA256保持不变。历史实验旧路径已加入
项目既有relocations.json映射，无需改写实验审计文件。
后续导出请用新文件名保存在exports/或自己的备份目录，不覆盖实验快照。
网页下载位置由浏览器决定，不自动写入exports/。
原preview.jpg归档到hard_samples/_ui_snapshots/，旧Python字节码归档到
logs/cache_before_layout_20261022/，均不参与程序运行；本轮没有永久删除标注、备份、截图或日志。

## 启动

全部入口统一使用Conda环境 **posture-mlp-v2**，不再回退到yolo-pose。
首次安装：

~~~powershell
conda env create -f posture_mlp_v2/environment.yml
conda activate posture-mlp-v2
python -m pip install -e "./posture_mlp_v2[render]"
~~~

~~~powershell
conda activate posture-mlp-v2
python posture_labeler/server.py
~~~

Windows可直接双击start.cmd，也可执行：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File posture_labeler/start.ps1 -Port 8765
powershell -NoProfile -ExecutionPolicy Bypass -File posture_labeler/start.ps1 -CheckEnvironment
~~~

后一条只检查Python、NumPy和OpenCV，不启动服务。标注无需安装PyTorch；
模型训练额外安装PyTorch CPU，见项目README。

看到 `Ready` 后打开 http://127.0.0.1:8765 。
服务绑定本机地址127.0.0.1；终端保持运行，Ctrl+C停止服务。

为其他数据集指定视频、关键点和数据库：

~~~powershell
python posture_labeler/server.py --video "D:/data/video.mp4" --keypoints "D:/data/video_keypoints.jsonl" --database "D:/data/labels.sqlite3" --port 8765
~~~

| 参数 | 默认值 |
|---|---|
| `--video` | posture_mlp_v2/inputs/example/VID_20260921_182630.mp4 |
| `--keypoints` | posture_mlp_v2/inputs/example/VID_20260921_182630_keypoints.jsonl |
| `--database` | posture_labeler/data/annotations.sqlite3 |
| `--port` | 8765 |
| `--hard-samples-dir` | posture_labeler/hard_samples |

每个数据集使用独立数据库。数据库身份由视频和关键点内容哈希组成。
输入字段约定见[输入格式](../posture_mlp_v2/inputs/README.md)。

## 标注流程

1. 左侧选择帧，在画面或P按钮中选择人物；同帧重叠框可反复点击切换。
2. 根据原图和人物放大图判断姿态，填写遮挡程度及备注。
3. 点击姿态按钮保存。开启“保存后自动进入”可继续下一个目标。
4. 使用撤销恢复最后一次保存，使用清除标签重新标注。
5. 导出全部已保存标签，得到JSONL或CSV。

默认按每3秒抽样。修改抽样间隔时，已标帧和手动添加的帧继续保留。
姿态按钮立即保存；已标样本的备注在停止输入约1.2秒后保存。
正式数据存储在SQLite，浏览器保存光标、偏好和备注草稿。
多标签页编辑采用revision检查，发生冲突时刷新当前样本供核对。

## 标注标准

| 标签 | 判定依据 |
|---|---|
| `sitting` | 能确认坐在支撑物上 |
| `standing` | 能确认站立，包括弯腰站立 |
| `uncertain` | 当前帧的支撑关系有歧义，或处于起坐过渡等状态 |
| `ignore` | 误检、严重错框、重复框 |

依据当前帧独立判定；姿态与遮挡程度分别记录。
同一人的重复框保留一个适合的目标，其余标ignore。
覆盖多人物、朝向、位置和姿态，并按时间段规划训练、校准及评估样本。

| 遮挡标签 | 含义 |
|---|---|
| `unmarked` | 待填写 |
| `clear` | 清晰可见 |
| `partial` | 部分遮挡 |
| `severe` | 严重遮挡或下半身出画 |

## 快捷键

| 按键 | 操作 |
|---|---|
| 1 / 2 / 3 / 4 | sitting / standing / uncertain / ignore |
| ← / → | 上一个 / 下一个人物 |
| N | 下一个待标人物 |
| Enter | 保存补充信息并进入下一目标 |
| Ctrl+Z / Cmd+Z | 撤销最后一次保存 |
| K / B | 切换骨架 / 检测框显示 |
| ? | 使用指南 |

## 困难样本队列

打开 http://127.0.0.1:8765/#hard ，启用“困难样本”筛选。
带★的P按钮表示本批补标目标。填写腿部可见程度后保存姿态；ignore目标可直接保存。
该筛选控制队列进度和自动进入，导出覆盖数据库中的全部已保存标签。

运行服务后生成新队列：

~~~powershell
python posture_labeler/mine_hard_samples.py --batch-dir "posture_labeler/hard_samples/my_batch" --count 96 --install
~~~

`--run-dir` 指定实验目录，`--predictions` 指定模型输出的逐帧JSONL。
默认使用 `posture_mlp_v2/runs/hard_v2_complete` 与 `posture_mlp_v2/outputs/example/` 中的结果。
批次目录使用新目录或空目录。

`--install` 先备份SQLite，再通过带本机令牌保护的接口激活批次。
批次目录必须是服务端hard-samples目录下的直接子目录，名称只使用英文字母、数字、下划线或连字符。
服务从该批次queue.json读取目标，并在抽样结果中自动包含目标帧。
安装不会写标注、修改操作历史或增加数据库pin；切换后刷新网页或点击“重新连接”即可。

**queue.json是唯一目标清单来源。** active.json只包含：

~~~json
{"version": 1, "batch": "batch_20261019_v1"}
~~~

后端每次读取当前批次，网页使用/api/queue中的hard_queue信息；
GET /api/hard-queue也可单独查看当前清单。不再往HTML复制JSON。
批次与视频/关键点dataset_id不匹配时禁用困难筛选，不把旧目标套到新数据集。
批次损坏、重复目标、无效帧/人员和越界路径会报错，不静默忽略。
未配置active.json时只提供普通标注。

| 批次文件 | 用途 |
|---|---|
| `queue.json` | 网页目标清单 |
| `../active.json` | 当前批次名，切换批次时原子更新 |
| `selection_audit.json` | 候选筛选依据 |
| `installation.json` | 队列安装核验 |
| `annotations_before_queue.sqlite3` | 安装前数据库备份 |

主动筛选的样本用于开发与补标，最终评估另留独立视频或时间段。

## 导出格式

JSONL每行一个人工标注的人体实例：

| 字段 | 内容 |
|---|---|
| `dataset_id` | 视频与关键点内容哈希的组合标识 |
| `output_frame`、`source_frame` | 导出帧号、原视频帧号，均从0开始 |
| `source_time_sec`、`segment`、`person_id` | 原视频时间、片段编号、帧内人员编号 |
| `label`、`occlusion`、`note` | 姿态、遮挡程度、备注 |
| `revision`、`updated_at` | 标注版本和UTC保存时间 |
| `bbox_xyxy`、`keypoints`、`detection_confidence` | 检测框、17个关键点和人体检测置信度 |
| `image_width`、`image_height` | 原图尺寸 |
| `training_eligible` | sitting/standing为true，其余为false |

CSV使用UTF-8 BOM，数组编码为JSON字符串；可能被解释为公式的备注加单引号前缀。
训练使用JSONL导出，标注复核可使用CSV。

## 备份与恢复

定期导出JSONL。完整数据库备份时，停止服务后复制 `data/annotations.sqlite3`。
恢复时配套使用相同视频、关键点和数据库文件。

保存异常时先核对当前样本和服务连接；服务重启后刷新页面。
服务运行期间保持输入视频和关键点内容稳定。

## 测试

~~~powershell
python -B -m unittest discover -s posture_labeler/tests -p "test_*.py" -v
node posture_labeler/tests/test_hard_queue.cjs
~~~

Python测试使用临时视频和数据库，覆盖抽样、保存、撤销、冲突、导出及HTTP访问。
队列与前端测试使用合成清单，不依赖真实批次或正式数据库。
从项目根目录执行，不要直接运行tests/中的单个Python文件。

历史batch_20261019_v1的只读核验单独运行，不混入通用测试：

~~~powershell
python -B -m unittest discover -s posture_labeler/tests -p "check_legacy_batch.py" -v
~~~

该核验依赖历史queue.json、审计记录和安装前SQLite备份，因此整个历史批次应一起归档。

# 输入格式

每个输入子目录包含一个视频和一份关键点 JSONL：

~~~text
posture_mlp_v2/inputs/example/
  VID_20260921_182630.mp4
  VID_20260921_182630_keypoints.jsonl
  labels/                         可选训练标注快照
    baseline.jsonl
    hard_v1.jsonl
~~~

视频支持 MP4、MOV、AVI、MKV。JSONL 使用 UTF-8 编码，每行保存一个帧对象。

## 使用

安装包后，从WIFI工作区目录执行（在算法项目内则去掉路径前的posture_mlp_v2/）：

~~~powershell
python -m posture_mlp_v2.cli.render --input-dir "posture_mlp_v2/inputs/my_video" --model-dir "posture_mlp_v2/runs/hard_v2_complete/model"
~~~

`--input-dir` 用于指定数据目录。训练命令使用相同参数，并额外提供人工标注文件。
输入目录顶层的视频和JSONL各为一个；labels/子目录不参与关键点文件扫描；上述my_video需先按此规范创建。
默认输出到同名的 `posture_mlp_v2/outputs/my_video/`，不必重复填写输出路径。
现有示例为inputs/example与outputs/example。程序拒绝覆盖非空结果目录；
可用 `--output-dir`显式指定其他目录。参见[输出格式](../outputs/README.md)。

## 帧对象

~~~json
{"output_frame":0,"source_frame":1798,"source_time_sec":60.003045,"source_time":"00:01:00.003","segment":1,"people":[]}
~~~

| 字段 | 类型 | 约定 |
|---|---|---|
| `output_frame` | integer | 导出帧号，从0开始连续递增 |
| `source_frame` | integer | 原视频帧号，从0开始，按时间递增 |
| `source_time_sec` | number | 原视频秒数，与source_frame / FPS对应 |
| `source_time` | string | 可读时间，格式HH:MM:SS.mmm |
| `segment` | integer | 处理片段编号，从1开始按时间连续编号 |
| `people` | array | 当前帧的人物列表；空列表表示无人 |

## 人物对象

`people` 中每个人物包含：

| 字段 | 类型 | 约定 |
|---|---|---|
| `person_id` | integer | 当前帧内唯一编号 |
| `bbox_xyxy` | array | 原图检测框：[左上x, 左上y, 右下x, 右下y] |
| `detection_confidence` | number | 人体检测置信度，范围[0,1] |
| `keypoints` | array | 17个COCO关键点对象 |

`person_id` 的作用范围为当前帧。跨帧身份匹配由接入方的跟踪模块处理。

## 关键点对象

17点列表中单个元素的格式：

~~~json
{"index":13,"name":"left_knee","x":530.2,"y":410.8,"confidence":0.12}
~~~

- `index`：COCO17关节索引，每个人完整包含0–16。
- `name`：可选的关节名称，方便阅读。
- `x`、`y`：原图像素坐标，左上角为原点，向右为x正方向、向下为y正方向。
- `confidence`：关节置信度，范围 `[0,1]`；缺失关节保留索引并将置信度设为0。

索引顺序如下，左右以人物自身为准：

~~~text
0 鼻子
1/2 左眼/右眼       3/4 左耳/右耳
5/6 左肩/右肩       7/8 左肘/右肘
9/10 左腕/右腕      11/12 左髋/右髋
13/14 左膝/右膝     15/16 左踝/右踝
~~~

坐标和置信度使用有限数值。姿态分类内部按身体中心和躯干尺度进行归一化，
并为低置信度关节生成缺失掩码。

## 视频属性与处理区间

程序从视频读取尺寸、FPS及总帧数，从JSONL统计处理帧数和人体实例数。
每个segment的区间由首个source_frame至最后一个source_frame后一帧推导。
距整数秒半帧以内的边界对齐到整数秒。

示例数据包含原视频01:00–02:00、06:00–07:00、10:00–11:00，共5394帧。
视频绘制按实际导出的帧处理；训练分割协议配置见[算法模块](../README.md)。

## 添加数据集

1. 在 `posture_mlp_v2/inputs/` 下创建子目录。
2. 放入原视频及对应的关键点JSONL。
3. 使用 `--input-dir` 指定该目录。
4. 程序在 `posture_mlp_v2/outputs/` 下生成同名子目录。
5. 为人工标注指定独立数据库，导出与该数据集匹配的标签。

## 训练标注快照

`labels/` 为可选子目录，推理不读取它。训练通过 `--old-labels` 和 `--hard-labels` 显式指定快照，或使用 `configs/train.yaml`。
示例快照复制自标注台的baseline.jsonl与hard_v1.jsonl，字节内容和SHA256保持一致；原标注工作文件未移动。
后续新标注导出为新的版本文件，不要无意覆盖已用于训练的快照。


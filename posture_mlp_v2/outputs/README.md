# 输出格式与输入对应关系

每个数据集使用一个同名输入/输出文件夹：

~~~text
posture_mlp_v2/
  inputs/
    example/                原视频＋关键点
    meeting_02/             另一份原视频＋关键点
  outputs/
    example/                example的预测结果
    meeting_02/             meeting_02的预测结果
~~~

从WIFI项目根目录运行：

~~~powershell
python -m posture_mlp_v2.cli.render --input-dir "posture_mlp_v2/inputs/meeting_02" --model-dir "posture_mlp_v2/runs/hard_v2_complete/model"
~~~

前提是已创建inputs/meeting_02并放入一个视频和一份关键点JSONL。
程序默认输出到inputs/的同级outputs/meeting_02，与安装的源码位置无关。非标准输入位置可指定--project-root或--output-dir。
输出文件名取自视频名，不要求视频名也等于文件夹名。

## 每个结果目录包含

~~~text
outputs/<数据集名>/
  <视频名>_keypoints.jsonl               每帧关键点＋新姿态
  <视频名>_hard_v2_pose_posture.mp4       骨架与姿态叠加视频
  <视频名>_run_metadata.json             输入、模型、参数与产物路径
  qa/                                  抽查帧/可选标注复查图
  verification.json                    执行verify后生成，不是render必有文件
~~~

加--skip-video不生成视频和抽查帧，仍导出逐帧JSONL和运行记录，不创建空的qa/目录。
输出是新文件；原视频、检测框和关键点保持不变。

## 结果JSONL

沿用[输入格式](../inputs/README.md)的帧号、时间、片段、人物编号、框、关键点。
每个people元素增加/更新posture：

| 字段 | 含义 |
|---|---|
| label | 最终姿态：sitting / standing / unknown |
| candidate_label | 拒识前的二分类候选 |
| accepted | 是否通过拒识检查 |
| rejection_reasons | 拒识原因列表 |
| probabilities、confidence | 二分类分数，不是可靠性保证 |
| visibility_group | 关键点可见度代理分组 |
| support_distance等 | 训练邻域拒识的诊断信息 |

若输入已有posture，原值保存在previous_posture；旧预测不作为模型输入。
person_id仅在当前帧内有效。

## 覆盖保护与历史结果

- 默认按输入文件夹名配对；非空输出目录会报错，不覆盖已有结果。
- 多模型比较可用--output-dir显式指定新目录；训练实验仍放在runs/。
- 当前outputs/example由原outputs/hard_v2完整迁入，视频和JSONL没有重跑或修改。
- 历史run_metadata保留旧路径以维持原始记录，通过data.resolve_recorded_path兼容映射到新位置。
  新运行记录会直接保存实际的新路径及dataset_name/input_directory/output_directory。
- 现有example结果来自历史v2关键点缓存，历史来源核验仍使用runs/hard_v2_complete/provenance/中的对应字节版本；
  不能把inputs/example的原始YOLO导出哈希当作历史v2缓存哈希。

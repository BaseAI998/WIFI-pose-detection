# 历史来源证据

本目录由原archive/input_example_legacy/迁入，归属于hard_v2_complete实验。

- *_keypoints_v2.jsonl：历史关键点及旧姿态，保持原字节不变。
- *_source_metadata.json、input.json：原来源记录，保持原字节不变。
- relocations.json：历史路径到当前文件的映射，目标路径相对于posture_mlp_v2项目根目录。

历史标注路径映射到inputs/example/labels/中的同字节快照；标注台原工作文件仍保留。
现有审计记录、模型权重、视频和预测JSONL不改写；迁移哈希清单位于上一级layout_migration.json。
普通--input-dir推理不读取本目录，历史核验会读取，不能当作运行缓存删除。

在算法项目根目录运行：

~~~powershell
python -m posture_mlp_v2.cli.verify --run-dir runs/hard_v2_complete --output-metadata outputs/example/VID_20260921_182630_run_metadata.json
~~~

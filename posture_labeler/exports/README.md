# 标注快照

这里保存实验使用的冻结导出，不是标注台的实时数据库。

| 文件 | 标注数量 | 用途 |
|---|---:|---|
| baseline.jsonl | 255 | 基础标注快照 |
| hard_v1.jsonl | 351 | 基础标注＋第一轮96个困难样本 |
| baseline.csv | 255 | 基础快照的人读版本 |

2026-10-22整理时仅移动/改名，没有更改文件内容。
原始名称分别为posture_labels_4678427a (1).jsonl、posture_labels_4678427a (1)-hard.jsonl、
posture_labels_4678427a (1).csv。历史实验中的旧路径通过项目已有的relocations.json兼容。

当前正式数据库仍在 ../data/annotations.sqlite3，整理时有352条标注，因此上述351条快照并不是最新数据库。
后续导出请选择新的文件名，例如labels_20261022.jsonl，不覆盖实验快照。
网页导出由浏览器选择下载位置；它不会自动覆盖此目录。

# 工作区格式

每个项目位于 `workspace/projects/<project-id>/`：

- `project.json`：项目类别、裁剪策略、图片索引和状态。
- `originals/`：导入的原图副本，不被覆盖。
- `images/`：统一为正方形的工作图。
- `labels/`：与工作图同名的 YOLO bbox 文本。
- `exports/<id>/`：导出的标准数据集。

标签每行是 `class_id center_x center_y width height`，坐标均为相对于正方形工作图的 0 到 1 归一化值。导出前应检查审核状态和类别统计。

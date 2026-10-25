# YOLO Label Studio Local

Windows 本地离线 YOLO 检测框数据集整理与审核工具。当前版本支持：

- 图片导入并统一为 1:1（默认中心裁剪）；已有 YOLO bbox 会同步变换。
- 浏览图片、读取 YOLO 标签并在 Canvas 回显。
- Canvas 新增、拖动、删除和保存 bbox。
- 通过本地 `.pt`（Ultralytics）或常见 `.onnx` 检测模型批量生成候选框。
- 导出标准 `images/{train,val,test}`、`labels/{train,val,test}` 和 `dataset.yaml`。

## 启动

```powershell
cd D:\Github\yolo-label-studio-local
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,pt,onnx]"
$env:YOLO_LABEL_WORKSPACE = "D:\Github\yolo-label-studio-local\workspace"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

浏览器打开 <http://127.0.0.1:8765>。模型路径必须是本机文件路径；输入图片通过浏览器上传到工作区，不会发送到外部服务。

## 目前限制

第一版仅支持 YOLO 检测框，不支持 polygon/segmentation。ONNX 适配器覆盖常见 `xywh + class scores` 输出，特殊导出布局需要新增适配器。批量预测写入草稿标签；人工保存后再作为已编辑结果导出。训练、持久化任务队列、复杂数据增强和实例分割属于后续版本。

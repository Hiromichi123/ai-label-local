# ai-label-local

运行在本机或自有服务器上的 YOLO 检测框标注工具。浏览器负责图片与标注交互，Python 后端管理项目、模型推理和数据导出。

## 功能

- 按项目管理图片、类别和 YOLO bbox 标签。
- 导入图片目录，自动匹配同目录同名 `.txt` 检测框标签；原图另存副本。
- 非正方形图片先选区：裁剪边长固定为短边，用滑块或拖动移动正方形，确认后永久重写工作图并变换标签，再进入标注。
- Canvas 新增、选择、拖动、删除框，支持撤销、缩放和当前类别选择。
- 图片切换时自动保存；支持 `↑` / `↓` 和鼠标侧键翻图（输入框中不拦截方向键）。
- 新增类别、修改类别名称，保留已有类别 ID。
- 使用本地 Ultralytics `.pt` 或常见 YOLO `.onnx` 检测模型批量生成候选框。
- 导出 `images/{train,val,test}`、`labels/{train,val,test}`、`dataset.yaml` 和拆分清单。

## 环境

- Python **3.11 或 3.12**。
- 浏览器支持 Canvas 与目录选择；建议使用 Chrome 或 Edge。
- 基础功能无需 GPU；`.pt` 推理通过可选的 Ultralytics/PyTorch 依赖提供，GPU 使用取决于所安装的 PyTorch。

## Windows 启动

```powershell
git clone https://github.com/Hiromichi123/ai-label-local.git
cd ai-label-local
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,pt,onnx]"
$env:YOLO_LABEL_WORKSPACE = Join-Path $PWD "workspace"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765>。已有 Python 3.11/3.12 时，也可用对应解释器创建虚拟环境。

仅使用人工标注时可安装 `-e .`；只需要 ONNX 或 PyTorch 推理时，分别安装 `-e ".[onnx]"` 或 `-e ".[pt]"`。

## Linux / Gentoo 部署

```bash
cd ~
git clone https://github.com/Hiromichi123/ai-label-local.git
cd ai-label-local
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test,pt,onnx]'
export YOLO_LABEL_WORKSPACE="$HOME/ai-label-local/workspace"
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

也可以使用 Python 3.12。部署时保留源码中的 `frontend/` 目录；当前采用源码目录的 editable 安装，前端静态资源由 FastAPI 直接提供，无需 Node.js 构建。

服务器模式下，图片和模型路径均属于**服务器**。在本机建立 SSH 转发后访问网页：

```bash
ssh -N -L 8765:127.0.0.1:8765 user@server
```

SSH 使用自定义端口时加 `-p 端口`。浏览器访问 <http://127.0.0.1:8765>。应用目前没有账户认证，默认保持回环监听；需要其他接入方式时另配认证和反向代理。

### systemd 用户服务

将以下内容保存为 `~/.config/systemd/user/ai-label-local.service`，把 `/home/USER` 替换为实际家目录：

```ini
[Unit]
Description=AI Label Local
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/USER/ai-label-local
Environment=YOLO_LABEL_WORKSPACE=/home/USER/ai-label-local/workspace
ExecStart=/home/USER/ai-label-local/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now ai-label-local.service
systemctl --user status ai-label-local.service
journalctl --user -u ai-label-local.service -n 50
```

用户服务在退出 SSH 后是否保持运行取决于主机的 linger 设置，可用 `loginctl show-user "$USER" -p Linger` 查询。

## 标注流程

1. 新建项目，按顺序填写类别名，类别 ID 从 0 开始。
2. 选择图片目录并导入。**导入会替换当前项目已有图片和标注**；需要保留旧数据时先导出或另建项目。
3. 手工编辑框，或填写服务器/本机上实际存在的模型文件绝对路径进行批量预测。
4. 检查预测框，人工保存后导出。预测产生的候选标签不等于已人工验收的标签。
5. 导出结果保存在项目的 `exports/` 下，页面会显示服务器端导出路径。

支持输入 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。同名标签每行应为：

```text
class_id center_x center_y width height
```

坐标归一化到 `[0, 1]`，以对应图片宽高为基准。类别 ID 必须属于项目类别列表。

## 工作区与备份

使用环境变量 `YOLO_LABEL_WORKSPACE` 指定工作区，默认是启动目录下的 `workspace/`。

```text
workspace/projects/<project-id>/
  project.json     # 类别、图片索引及状态
  originals/       # 导入时保留的原图副本
  images/          # 正方形工作图
  labels/          # YOLO 检测框标签
  exports/         # 导出数据集
```

备份时保存整个工作区；更新源码时保留工作区和另存的数据集。工作区、数据集、模型权重及虚拟环境均不进入 Git 仓库。详见 [工作区格式](docs/format.md)。

## 开发与测试

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests
```

前端为原生 HTML / CSS / JavaScript，后端使用 FastAPI。接口文档位于运行服务的 `/docs`。

## 当前边界

- 只支持 YOLO **检测框**，不支持多边形、实例分割或堆叠深度标签。
- `.pt` 适配器加载原生 Ultralytics 模型；自定义网络 checkpoint 需要额外适配。
- ONNX 适配器支持常见 YOLO 检测输出，不保证兼容所有导出布局。
- 预测任务在当前进程内执行，重启后任务队列不保留。
- 数据导出按图片随机拆分；同一场景、视频相邻帧等需要在后续训练前另行做组隔离。
- 当前不是多人并发标注服务，建议单用户操作同一项目。


## 本地标注、完成后同步

双击 `start_local.cmd`，打开 http://127.0.0.1:8765 。直接使用本机工作区，不需要 SSH 转发。

导入新目录时保留完整画面，非 1:1 图先显示裁剪界面。长图水平移动、竖图垂直移动；正方形边长固定为短边，不放大。点击“确认并永久裁剪”会重写工作数据集里的 JPEG、裁切和平移 YOLO 标签、更新尺寸；裁剪框外的标签移除。之后导出的数据和同步到远端的图片都是裁剪后的图片，并非仅改变显示视口。

原始上传副本 `originals/` 和裁剪前备份 `crop_backups/` 保留；源照片目录不被修改。旧版已经自动裁成方形的项目不会被自动重置；若需重新选完整原图区域，另建项目重新导入原图，避免覆盖现有标注。

全部打标完成并保存后，双击 `sync_labels.cmd` 选择本地项目，或运行：

```powershell
.\.venv\Scripts\python.exe scripts/sync_to_remote.py --project 项目ID
```

默认同步到 `hiro@jinjiao.net:4022` 的 `/home/hiro/ai-label-local/workspace`，可通过 `--host`、`--port`、`--remote-workspace` 修改。通过本机 OpenSSH 的 scp/ssh 交互认证，不保存密码。同步工作图、标签和 project.json；永久裁剪改变了图像坐标，不能只单独传 TXT。原始副本及裁剪备份留在本地，不随包重复上传。远端已有同 ID 项目会备份到 `workspace/sync_backups/`，其他项目保留。

同步期间暂停编辑，完成后刷新远端项目列表。仍有待裁剪图片时拒绝同步。

裁剪界面可按 Enter 确认永久裁剪；在滑块上同样有效，输入文本及中文输入法组合期间不会触发。服务可用 `YOLO_LABEL_DEFAULT_MODEL` 指定默认检测模型；网页自动填入该路径。批量预标注默认仅处理空标签且未人工保存确认的图片，保留现有人工标注；API 的 `overwrite_existing=true` 仅用于明确要求覆盖的情况。

预标注的候选框在仅浏览／翻图时不会自动成为人工审核结果；实际修改会自动保存，或点击“保存标注”明确确认。


智能打标有两个入口：“标注当前图片（覆盖已有）”会重做当前图，包括已完成标注；“标注未完成图片”包含未标注和模型待审核候选，保留已确认标注，跳过待裁剪图片。预测结果仍是候选；原标签备份在项目 `prediction_backups/`。进度条显示模型加载、已处理数量、百分比和失败原因。

回车按当前阶段完成操作：裁剪时确认永久裁剪；标注／审核时保存并标记审核完成，自动进入下一张（最后一张只保存）。空标签也可用回车明确确认。文本输入、中文输入法组合、长按重复及模型运行期间不会触发完成操作。

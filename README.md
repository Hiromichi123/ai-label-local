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

智能打标置信度使用滑动条，并在当前浏览器记忆（本地与远程地址分别保存）。点击“编辑标签文件”直接修改当前 YOLO 文本，校验保存后更新画布；空文本保存为无目标。画布左键拖动新增框（包括从已有框内起笔），右键选择已有框（重叠时选最上层），Delete 删除选中框，撤销可恢复删除；空白处右键取消选择。单击不产生零面积框，Esc 可取消绘制。文本输入、标签编辑器、训练页和裁剪模式不会误触 Delete 删除。坐标也可用标签文本编辑器修改。编辑器中回车用于换行，取消不保存。

### 服务器目录浏览与导入
网页点击“浏览服务器目录并导入”，逐级浏览或输入服务器绝对路径，填写新项目名称与按编号排列的类别，选择是否包含子目录后导入。支持 jpg/jpeg/png/bmp/webp 及图片旁同名 YOLO txt。导入复制到新项目，后续修改不回写源目录。项目下拉框切换已导入项目。
默认允许浏览服务账号的 home 和标注 workspace；可用环境变量 `YOLO_LABEL_IMPORT_ROOTS` 指定目录（Linux 用冒号分隔，Windows 用分号分隔）。符号链接不能逃出允许范围，隐藏目录不递归扫描，单次最多 20000 张。此服务应继续通过本机端口或 SSH 隧道访问。

### 网页模型训练
切换“模型训练”页签，选择当前项目，设置服务器上已有的 YOLO 检测 `.pt` 初始权重、设备、轮数、分辨率、批大小、学习率、早停、验证比例、种子和色彩/亮度/翻转/Mosaic 增广后开始。仅使用已审核且完成裁剪的图片（包括明确审核过的空标签），至少两张；未审核候选不参与。按种子划分互不重叠的训练/验证集，任务目录 `workspace/training/<任务ID>` 保存快照、manifest、参数、日志和 best.pt。小样本验证分数只供参考。
训练在独立进程运行，同一服务一次运行一个训练任务；刷新浏览器可继续查看历史、轮次、指标和日志，也可停止。服务重启会中断任务并标记失败。安装训练依赖：`pip install -e ".[pt]"`。训练完成后在打标页“已训练模型”下拉框选择，或点击“在打标页使用此模型”；自动沿用训练分辨率。不自动覆盖当前模型或重新打标。
进度通过 [Ultralytics 训练回调](https://docs.ultralytics.com/usage/callbacks/) 在每轮验证完成后更新。

远程数据统一放在项目内：`/home/hiro/ai-label-local/datasets/` 存放源数据和早期训练数据，`workspace/projects/` 存放网页标注数据，`workspace/training/` 存放网页训练快照与权重。这些目录均由 `.gitignore` 排除，不上传图片、标签或训练产物。

类别选择：当前标签下拉框在未选框时设置下一次绘制的类别，选中框时修改该框类别；数字 0–9 直接选择同编号类别，支持主键盘和数字小键盘（Num Lock 开启）。输入文本、裁剪、绘制和智能打标期间不触发类别快捷键。下拉框只列出当前项目已有类别，可在左侧新增类别。

类别支持自定义非连续编号（如 5、12、30）：左侧新增或编辑“编号 + 类别名”，当前标签旁可手动输入已有编号并点击应用或按回车。修改编号会备份到 `class_backups/` 并迁移现有标签。工作标签保留自定义编号；训练/导出数据使用连续编号，并保存 `class_mapping.json`，推理结果映射回项目编号。
“模型管理”列出网页训练产物及项目 `models/` 内的权重，可选择、设为默认和删除。默认权重和训练正使用的权重不能删除。删除训练模型时移除 best.pt/last.pt，保留数据和日志。“下载当前图片”通过浏览器下载当前工作图片（包含已确认的裁剪），文件格式为 JPEG。

训练参数按项目自动保存到 `workspace/projects/<项目ID>/training_settings.json`，也可点击“保存训练参数”。模型名、初始权重、设备及所有训练/增广参数均持久化；刷新、切换项目和服务重启后恢复。未设置过的项目会恢复最近一次训练配置，否则使用默认值。浏览器暂存尚未同步的修改，断线重连后重试；不完整或无效参数保留草稿，不覆盖服务端有效配置。同步图片和标签时保留远程主机的训练参数。

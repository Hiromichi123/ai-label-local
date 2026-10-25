const state = { project: null, images: [], current: null, annotations: [], version: 0, selected: -1, history: [] };
const $ = (id) => document.getElementById(id);
const message = (text) => { $('message').textContent = text; };
async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail?.message || body.detail || '请求失败');
  return body;
}
function jsonOptions(value) { return { headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value) }; }
async function loadProjects() {
  const projects = await api('/api/projects');
  $('projectSelect').replaceChildren(...projects.map(p => { const option = new Option(p.name, p.id); return option; }));
  if (projects.length) await selectProject(projects[0].id);
}
async function selectProject(id) {
  state.project = await api(`/api/projects/${id}`);
  state.images = Object.values(state.project.images);
  renderClasses();
  renderImages();
  if (state.images.length) await selectImage(state.images[0].id);
}
function renderClasses() {
  $('classes').replaceChildren(...state.project.class_names.map((name, i) => {
    const row = document.createElement('div'); row.className = 'class-row';
    const index = document.createElement('span'); index.textContent = i;
    const input = document.createElement('input'); input.value = name; input.disabled = true;
    row.append(index, input); return row;
  }));
}
function renderImages() {
  $('images').replaceChildren(...state.images.map(image => { const item = document.createElement('div'); item.className = `image-item ${state.current === image.id ? 'active' : ''}`; item.textContent = `${image.filename} (${image.status})`; item.onclick = () => selectImage(image.id); return item; }));
}
async function selectImage(id) {
  state.current = id; state.selected = -1; state.history = [];
  const result = await api(`/api/projects/${state.project.id}/images/${id}/annotations`);
  state.annotations = result.boxes; state.version = result.version;
  renderImages(); await draw(); await checkSaveStatus();
}
async function checkSaveStatus() {
  const indicator = $('saveStatus');
  if (!state.current) { indicator.textContent = '未检查'; indicator.className = 'save-status'; return; }
  try {
    const status = await api(`/api/projects/${state.project.id}/images/${state.current}/annotation-status`);
    indicator.className = `save-status ${status.valid ? 'ok' : status.saved ? 'warn' : 'error'}`;
    indicator.textContent = status.valid ? `已保存 · ${status.box_count} 个框` : status.status === 'missing' ? '标签未保存' : `标签${status.status}`;
    indicator.title = status.error || '本地标签文件校验通过';
  } catch (error) {
    indicator.className = 'save-status error'; indicator.textContent = '无法检查标签'; indicator.title = error.message;
  }
}
function canvasPoint(event) {
  const canvas = $('canvas'), rect = canvas.getBoundingClientRect();
  return { x: (event.clientX - rect.left) * canvas.width / rect.width, y: (event.clientY - rect.top) * canvas.height / rect.height };
}
async function draw() {
  if (!state.current) return;
  const image = new Image(); image.src = `/api/projects/${state.project.id}/images/${state.current}/asset`; await image.decode();
  const canvas = $('canvas'); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d'); ctx.drawImage(image, 0, 0);
  state.annotations.forEach((box, index) => { ctx.strokeStyle = index === state.selected ? '#ffdf5d' : '#4cff91'; ctx.lineWidth = Math.max(2, canvas.width / 400); ctx.strokeRect(box.x1, box.y1, box.x2-box.x1, box.y2-box.y1); ctx.fillStyle = ctx.strokeStyle; ctx.font = `${Math.max(14, canvas.width / 45)}px sans-serif`; ctx.fillText(`${box.class_id}${box.confidence == null ? '' : ` ${(box.confidence * 100).toFixed(0)}%`}`, box.x1 + 3, Math.max(16, box.y1 - 4)); });
  $('selection').textContent = state.selected >= 0 ? `框 ${state.selected + 1}` : '未选中';
}
function snapshot() { state.history.push(JSON.stringify(state.annotations)); if (state.history.length > 30) state.history.shift(); }
$('canvas').addEventListener('pointerdown', event => {
  if (!state.current) return;
  const p = canvasPoint(event);
  snapshot();
  // Existing boxes are intentionally read-only. Every pointer-down starts a new box.
  const box = { class_id: 0, x1: p.x, y1: p.y, x2: p.x, y2: p.y, source: 'human', status: 'edited' };
  state.annotations.push(box);
  state.selected = state.annotations.length - 1;
  const start = { ...p };
  const canvas = $('canvas');
  canvas.setPointerCapture?.(event.pointerId);

  const move = e => {
    const q = canvasPoint(e);
    // Normalize both corners, so drawing works from any starting corner.
    box.x1 = Math.max(0, Math.min(start.x, q.x));
    box.y1 = Math.max(0, Math.min(start.y, q.y));
    box.x2 = Math.min(canvas.width, Math.max(start.x, q.x));
    box.y2 = Math.min(canvas.height, Math.max(start.y, q.y));
    draw();
  };
  const up = () => {
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
    canvas.releasePointerCapture?.(event.pointerId);
    // Do not leave a zero-area annotation when the pointer only clicked.
    if (box.x2 - box.x1 < 2 || box.y2 - box.y1 < 2) {
      state.annotations.splice(state.selected, 1);
      state.selected = -1;
    }
    draw();
  };
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
});
$('addBox').onclick = () => { if (!state.current) return; snapshot(); state.annotations.push({class_id: 0, x1: 20, y1: 20, x2: 120, y2: 120, source: 'human', status: 'edited'}); state.selected = state.annotations.length - 1; draw(); };
$('deleteBox').onclick = () => { if (state.selected < 0) return; snapshot(); state.annotations.splice(state.selected, 1); state.selected = -1; draw(); };
$('undo').onclick = () => { const previous = state.history.pop(); if (previous) { state.annotations = JSON.parse(previous); state.selected = -1; draw(); } };
$('save').onclick = async () => { try { const result = await api(`/api/projects/${state.project.id}/images/${state.current}/annotations`, {method:'PUT', ...jsonOptions({version: state.version, boxes: state.annotations})}); state.version = result.version; message('标注已保存'); await selectProject(state.project.id); } catch (e) { message(e.message); } };
$('newProject').onclick = async () => { const name = prompt('项目名称', '我的数据集'); const classes = prompt('类别名（逗号分隔）', 'object'); if (!name || !classes) return; try { await api('/api/projects', {method:'POST', ...jsonOptions({name, class_names: classes.split(',').map(x => x.trim()), crop_mode: 'center_crop'})}); await loadProjects(); message('项目已创建'); } catch(e) { message(e.message); } };
$('projectSelect').onchange = e => selectProject(e.target.value);
$('addClass').onclick = async () => {
  if (!state.project) return;
  const input = $('newClass'); const name = input.value.trim();
  if (!name) { message('请输入类别名称'); return; }
  try {
    const result = await api(`/api/projects/${state.project.id}/classes`, {method: 'POST', ...jsonOptions({name})});
    state.project.class_names = result.class_names; renderClasses(); input.value = '';
    message(`已新增类别 ${result.class_id}: ${name}`);
  } catch (e) { message(e.message); }
};
$('import').onclick = async () => {
  const selected = [...$('directory').files];
  if (!selected.length || !state.project) { message('请先选择一个目录'); return; }
  const data = new FormData(); selected.forEach(file => data.append('files', file, file.webkitRelativePath || file.name));
  try {
    message('正在清除旧图片并导入新目录…');
    const result = await api(`/api/projects/${state.project.id}/images/import`, {method:'POST', body:data});
    state.current = null; await selectProject(state.project.id); message(`目录导入完成：${result.count} 张图片`);
    $('directory').value = '';
  } catch(e) { message(e.message); }
};
$('predict').onclick = async () => { if (!state.project) return; try { const job = await api(`/api/projects/${state.project.id}/predict`, {method:'POST', ...jsonOptions({model_path: $('model').value, confidence: Number($('confidence').value)})}); $('job').textContent = '任务运行中'; const timer = setInterval(async () => { const current = await api(`/api/jobs/${job.id}`); $('job').textContent = `${current.status}: ${current.completed}/${current.total}`; if (['completed','failed','cancelled'].includes(current.status)) { clearInterval(timer); await selectProject(state.project.id); } }, 500); } catch(e) { message(e.message); } };
$('export').onclick = async () => { if (!state.project) return; try { const result = await api(`/api/projects/${state.project.id}/export`, {method:'POST', ...jsonOptions({})}); message(`导出完成：${result.counts.train}/${result.counts.val}/${result.counts.test}`); } catch(e) { message(e.message); } };
loadProjects().catch(e => message(e.message));

const state = { project: null, images: [], current: null, annotations: [], version: 0, selected: -1, history: [], zoom: 0.35, crosshair: null, crop: null, drawId: 0 };
const imageCache = new Map();
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
  const config = await api('/api/config');
  if (!$('model').value) $('model').value = config.default_model_path || '';
  state.defaultImageSize = config.default_image_size || 640;
  const projects = await api('/api/projects');
  $('projectSelect').replaceChildren(...projects.map(p => { const option = new Option(p.name, p.id); return option; }));
  if (projects.length) await selectProject(projects[0].id);
}
async function selectProject(id) {
  if (state.project && state.project.id !== id && state.current) await saveCurrentImage();
  state.current = null; state.crop = null; state.annotations = []; state.selected = -1;
  imageCache.clear();
  state.project = await api(`/api/projects/${id}`);
  state.images = Object.values(state.project.images);
  renderClasses();
  renderImages();
  if (state.images.length) await selectImage(state.images[0].id);
  else { prepareCrop(); $('canvas').getContext('2d').clearRect(0,0,$('canvas').width,$('canvas').height); }
}
function renderClasses() {
  $('classes').replaceChildren(...state.project.class_names.map((name, i) => {
    const row = document.createElement('div'); row.className = 'class-row';
    const index = document.createElement('span'); index.textContent = i;
    const input = document.createElement('input'); input.value = name; input.title = '修改类别名称';
    const save = document.createElement('button'); save.textContent = '保存';
    save.onclick = async () => {
      try {
        const result = await api(`/api/projects/${state.project.id}/classes/${i}`, {method: 'PUT', ...jsonOptions({name: input.value})});
        state.project.class_names = result.class_names; renderClasses(); renderCurrentClass(); message(`类别 ${i} 已更新`);
      } catch (error) { message(error.message); }
    };
    row.append(index, input, save); return row;
  }));
  renderCurrentClass();
}
function renderCurrentClass() {
  const select = $('currentClass');
  select.replaceChildren(...state.project.class_names.map((name, i) => new Option(`${i}: ${name}`, i)));
  select.value = state.selected >= 0 ? String(state.annotations[state.selected]?.class_id ?? 0) : '0';
  select.disabled = state.selected < 0;
}
function applyCurrentClass(value) {
  if (state.selected < 0) return;
  snapshot(); state.annotations[state.selected].class_id = Number(value); renderBoxList(); renderCurrentClass(); draw();
}
function renderBoxList() {
  const list = $('boxList');
  list.replaceChildren(...state.annotations.map((box, index) => {
    const row = document.createElement('div'); row.className = `box-row ${state.selected === index ? 'selected' : ''}`;
    const title = document.createElement('button'); title.className = 'box-select'; title.textContent = `框 ${index + 1} · ${state.project.class_names[box.class_id] ?? `类别 ${box.class_id}`}`;
    title.onclick = () => { state.selected = index; renderBoxList(); renderCurrentClass(); draw(); };
    const remove = document.createElement('button'); remove.className = 'box-remove'; remove.textContent = '×'; remove.title = '删除此框';
    remove.onclick = event => { event.stopPropagation(); if (state.selected !== index) state.selected = index; snapshot(); state.annotations.splice(index, 1); state.selected = -1; renderBoxList(); renderCurrentClass(); draw(); };
    row.onclick = event => { if (event.target !== remove) { state.selected = index; renderBoxList(); renderCurrentClass(); draw(); } };
    row.append(title, remove); return row;
  }));
}
function renderImages() {
  $('images').replaceChildren(...state.images.map(image => {
    const item = document.createElement('div'); item.className = `image-item ${state.current === image.id ? 'active' : ''}`;
    const name = document.createElement('span'); name.textContent = image.filename;
    const reviewed = image.box_count > 0 && ['labeled','edited','approved'].includes(image.status);
    const status = document.createElement('span'); status.className = reviewed ? 'image-check' : 'image-state';
    status.textContent = image.needs_crop ? '待裁剪' : image.status === 'predicted' ? '候选·待审核' : reviewed ? '✓' : image.status === 'invalid' ? '标签错误' : '未标注';
    item.append(name, status); item.onclick = () => selectImage(image.id); return item;
  }));
}
async function saveCurrentImage(force = false) {
  if (!state.current || !state.project || state.crop) return true;
  if (!force && state.history.length === 0) return true;
  const result = await api(`/api/projects/${state.project.id}/images/${state.current}/annotations`, {
    method: 'PUT', ...jsonOptions({version: state.version, boxes: state.annotations})
  });
  state.version = result.version; state.history = [];
  const current = state.images.find(image => image.id === state.current);
  if (current) { current.status = 'labeled'; current.box_count = state.annotations.length; }
  renderImages();
  return true;
}
async function selectImage(id, saveBefore = true) {
  if (state.current === id) return;
  if (saveBefore && state.current) {
    try { await saveCurrentImage(); } catch (error) { message(`自动保存失败：${error.message}`); return; }
  }
  state.current = id; state.selected = -1; state.history = [];
  const result = await api(`/api/projects/${state.project.id}/images/${id}/annotations`);
  state.annotations = result.boxes; state.version = result.version;
  prepareCrop();
  renderImages(); renderBoxList(); await draw(); await checkSaveStatus();
}
function navigateImage(delta) {
  if (!state.images.length || !state.current) return;
  const index = state.images.findIndex(image => image.id === state.current);
  const next = Math.max(0, Math.min(state.images.length - 1, index + delta));
  if (next !== index) selectImage(state.images[next].id).catch(error => message(error.message));
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
  const drawId = ++state.drawId;
  const src = `/api/projects/${state.project.id}/images/${state.current}/asset?v=${state.version}`;
  if (!imageCache.has(src)) { if (imageCache.size >= 3) imageCache.delete(imageCache.keys().next().value); const img = new Image(); img.src = src; imageCache.set(src, img.decode().then(() => img).catch(error => { imageCache.delete(src); throw error; })); }
  const image = await imageCache.get(src); if (drawId !== state.drawId) return;
  const canvas = $('canvas'); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  canvas.style.width = `${image.naturalWidth * state.zoom}px`;
  canvas.style.height = `${image.naturalHeight * state.zoom}px`;
  const ctx = canvas.getContext('2d'); ctx.drawImage(image, 0, 0);
  if (state.crop) { drawCrop(ctx, canvas); return; }
  state.annotations.forEach((box, index) => { ctx.strokeStyle = index === state.selected ? '#ffdf5d' : '#4cff91'; ctx.lineWidth = Math.max(2, canvas.width / 400); ctx.strokeRect(box.x1, box.y1, box.x2-box.x1, box.y2-box.y1); ctx.fillStyle = ctx.strokeStyle; ctx.font = `${Math.max(14, canvas.width / 45)}px sans-serif`; ctx.fillText(`${box.class_id}${box.confidence == null ? '' : ` ${(box.confidence * 100).toFixed(0)}%`}`, box.x1 + 3, Math.max(16, box.y1 - 4)); });
  if (state.crosshair) {
    ctx.save(); ctx.strokeStyle = '#ffffff'; ctx.lineWidth = Math.max(3, canvas.width / 260); ctx.setLineDash([Math.max(10, canvas.width / 90), Math.max(8, canvas.width / 110)]);
    ctx.beginPath(); ctx.moveTo(state.crosshair.x, 0); ctx.lineTo(state.crosshair.x, canvas.height); ctx.moveTo(0, state.crosshair.y); ctx.lineTo(canvas.width, state.crosshair.y); ctx.stroke(); ctx.restore();
  }
  $('selection').textContent = state.selected >= 0 ? `框 ${state.selected + 1}` : '未选中';
  renderCurrentClass();
}
function snapshot() { state.history.push(JSON.stringify(state.annotations)); if (state.history.length > 30) state.history.shift(); }
$('canvas').addEventListener('pointermove', event => {
  if (!state.current) return;
  state.crosshair = canvasPoint(event);
  draw();
});
$('canvas').addEventListener('pointerleave', () => { state.crosshair = null; draw(); });
$('canvas').addEventListener('pointerdown', event => {
  if (!state.current || event.button !== 0) return;
  if (state.crop) { dragCrop(event); return; }
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
    renderBoxList(); draw();
  };
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
});
$('addBox').onclick = () => { if (!state.current || state.crop) return; snapshot(); state.annotations.push({class_id: 0, x1: 20, y1: 20, x2: 120, y2: 120, source: 'human', status: 'edited'}); state.selected = -1; renderBoxList(); draw(); };
$('deleteBox').onclick = () => { if (state.selected < 0) return; snapshot(); state.annotations.splice(state.selected, 1); state.selected = -1; renderBoxList(); renderCurrentClass(); draw(); };
$('currentClass').onchange = event => applyCurrentClass(event.target.value);
$('undo').onclick = () => { const previous = state.history.pop(); if (previous) { state.annotations = JSON.parse(previous); state.selected = -1; renderBoxList(); draw(); } };
$('save').onclick = async () => { try { await saveCurrentImage(true); await checkSaveStatus(); message('标注已保存'); } catch (e) { message(e.message); } };
$('newProject').onclick = async () => { const name = prompt('项目名称', '我的数据集'); const classes = prompt('类别名（逗号分隔）', 'object'); if (!name || !classes) return; try { await api('/api/projects', {method:'POST', ...jsonOptions({name, class_names: classes.split(',').map(x => x.trim()), crop_mode: 'manual_crop'})}); await loadProjects(); message('项目已创建'); } catch(e) { message(e.message); } };
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
$('predict').onclick = async () => { if (!state.project) return; try { await saveCurrentImage(); const job = await api(`/api/projects/${state.project.id}/predict`, {method:'POST', ...jsonOptions({model_path: $('model').value, confidence: Number($('confidence').value), image_size: state.defaultImageSize || 640})}); $('job').textContent = '任务运行中'; const timer = setInterval(async () => { const current = await api(`/api/jobs/${job.id}`); $('job').textContent = `${current.status}: ${current.completed}/${current.total}`; if (['completed','failed','cancelled'].includes(current.status)) { clearInterval(timer); await selectProject(state.project.id); } }, 500); } catch(e) { message(e.message); } };
$('export').onclick = async () => { if (!state.project) return; try { await saveCurrentImage(); const result = await api(`/api/projects/${state.project.id}/export`, {method:'POST', ...jsonOptions({})}); message(`导出完成：${result.counts.train}/${result.counts.val}/${result.counts.test}`); } catch(e) { message(e.message); } };
document.addEventListener('keydown', event => {
  const target = event.target;
  if (event.key === 'Enter' && state.crop && !event.isComposing && !event.repeat && !event.ctrlKey && !event.altKey && !event.metaKey && !(target instanceof HTMLTextAreaElement) && !target.isContentEditable && (!(target instanceof HTMLInputElement) || target.type === 'range') && !(target instanceof HTMLSelectElement)) {
    event.preventDefault();
    if (!$('confirmCrop').disabled) $('confirmCrop').click();
    return;
  }
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement || target.isContentEditable) return;
  if (event.key === 'ArrowUp') { event.preventDefault(); navigateImage(-1); }
  if (event.key === 'ArrowDown') { event.preventDefault(); navigateImage(1); }
});
$('canvas').addEventListener('mousedown', event => {
  if (event.button === 3 || event.button === 4) { event.preventDefault(); navigateImage(event.button === 3 ? -1 : 1); }
});
$('canvas').addEventListener('contextmenu', event => event.preventDefault());
$('zoomRange').oninput = event => {
  state.zoom = Number(event.target.value) / 100;
  $('zoomValue').textContent = `${event.target.value}%`;
  draw();
};
loadProjects().catch(e => message(e.message));

function prepareCrop() {
 const r=state.images.find(x=>x.id===state.current), side=r?Math.min(r.width,r.height):0;
 state.crop=r?.needs_crop?{left:Math.floor((r.width-side)/2),top:Math.floor((r.height-side)/2),side,width:r.width,height:r.height}:null;
 $('cropPanel').hidden=!state.crop; ['addBox','deleteBox','save','undo','currentClass'].forEach(id=>$(id).disabled=Boolean(state.crop)); $('boxList').hidden=Boolean(state.crop);
 if(state.crop){const c=state.crop,h=c.width>c.height; $('cropAxis').textContent=h?'水平位置':'垂直位置'; $('cropPosition').max=Math.max(c.width,c.height)-side; $('cropPosition').value=h?c.left:c.top; $('cropInfo').textContent=`${c.width} × ${c.height} → ${side} × ${side}`; $('selection').textContent='待确认裁剪区域';}
}
function setCropPosition(value){if(!state.crop)return; const c=state.crop,p=Math.round(Math.max(0,Math.min(Number(value),Math.max(c.width,c.height)-c.side))); c.left=c.width>c.height?p:0; c.top=c.height>c.width?p:0; $('cropPosition').value=p; draw();}
function drawCrop(ctx,canvas){const c=state.crop; ctx.fillStyle='rgba(0,0,0,.62)'; ctx.beginPath();ctx.rect(0,0,canvas.width,canvas.height);ctx.rect(c.left,c.top,c.side,c.side);ctx.fill('evenodd');ctx.strokeStyle='#4ee7ed';ctx.lineWidth=Math.max(3,canvas.width/350);ctx.strokeRect(c.left,c.top,c.side,c.side);ctx.save();ctx.lineWidth=Math.max(1,canvas.width/900);ctx.setLineDash([12,8]);for(const n of [1,2]){ctx.beginPath();ctx.moveTo(c.left+c.side*n/3,c.top);ctx.lineTo(c.left+c.side*n/3,c.top+c.side);ctx.moveTo(c.left,c.top+c.side*n/3);ctx.lineTo(c.left+c.side,c.top+c.side*n/3);ctx.stroke();}ctx.restore();}
function dragCrop(event){const start=canvasPoint(event),horizontal=state.crop.width>state.crop.height,initial=horizontal?state.crop.left:state.crop.top;const move=e=>{if(!state.crop)return;const q=canvasPoint(e);setCropPosition(initial+(horizontal?q.x-start.x:q.y-start.y));};const up=()=>{window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);window.removeEventListener('pointercancel',up);};window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);window.addEventListener('pointercancel',up);}
$('cropPosition').oninput=e=>setCropPosition(e.target.value);
$('confirmCrop').onclick=async()=>{if(!state.crop)return;$('confirmCrop').disabled=true;try{const result=await api(`/api/projects/${state.project.id}/images/${state.current}/crop`,{method:'POST',...jsonOptions({version:state.version,left:state.crop.left,top:state.crop.top})});const i=state.images.findIndex(x=>x.id===state.current);state.images[i]=result.image;state.project.images[state.current]=result.image;state.annotations=result.boxes;state.version=result.version;state.history=[];state.selected=-1;state.crosshair=null;prepareCrop();renderImages();renderBoxList();await draw();await checkSaveStatus();message('数据集图片和标签已永久裁剪，可以开始标注');}catch(e){message(e.message);}finally{$('confirmCrop').disabled=false;}};

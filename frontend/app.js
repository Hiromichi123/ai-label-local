const state = { project: null, images: [], current: null, annotations: [], version: 0, selected: -1, history: [], zoom: 0.35, crosshair: null, crop: null, drawId: 0 };
const imageCache = new Map();
function projectClassIds(p = state.project) { return p?.class_ids || (p?.class_names || []).map((_,i) => i); }
function className(id) { return state.project?.class_names[projectClassIds().indexOf(id)] ?? `类别 ${id}`; }
function parseClassSpec(text) { const items = text.split(/[,，]/).map(s => s.trim()).filter(Boolean); const parsed = items.map((s,i) => { const m = s.match(/^(\d+)\s*[:：]\s*(.+)$/); return {id:m?Number(m[1]):i,name:m?m[2].trim():s}; }); return {class_names:parsed.map(x=>x.name),class_ids:parsed.map(x=>x.id)}; }
const $ = (id) => document.getElementById(id);
const message = (text) => { $('message').textContent = text; };
async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail?.message || (Array.isArray(body.detail) ? body.detail.map(e => `${e.loc?.slice(1).join('.')}: ${e.msg}`).join('；') : body.detail) || '请求失败');
  return body;
}
function jsonOptions(value) { return { headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value) }; }
async function loadProjects() {
  const config = await api('/api/config');
  if (!$('model').value) $('model').value = config.default_model_path || '';
  state.defaultModelPath = config.default_model_path || '';
  state.defaultImageSize = config.default_image_size || 640;
  const projects = await api('/api/projects');
  $('projectSelect').replaceChildren(...projects.map(p => { const option = new Option(p.name, p.id); return option; }));
  $('trainingProject').replaceChildren(...projects.map(p => new Option(p.name, p.id)));
  if (projects.length) await selectProject(projects[0].id);
}
async function selectProject(id) {
  if (state.project && state.project.id !== id) await flushTrainingParameters(state.project.id).catch(() => {});
  if (state.project && state.project.id !== id && state.current) await saveCurrentImage();
  if (state.project?.id !== id) { state.activeClass = 0; trainingRuns = []; selectedTrainingRun = ''; state.trainingModelProject = null; state.selectedModelImageSize = null; $('model').value = state.defaultModelPath || ''; $('trainModel').value = ''; }
  state.current = null; state.crop = null; state.annotations = []; state.selected = -1;
  imageCache.clear();
  state.project = await api(`/api/projects/${id}`);
  state.images = Object.values(state.project.images);
  await restoreTrainingParameters(id);
  renderClasses();
  renderImages();
  $('projectSelect').value = id; $('trainingProject').value = id;
  refreshTraining().catch(e => { $('trainingError').textContent = e.message; });
  if (state.images.length) await selectImage(state.images[0].id);
  else { prepareCrop(); $('canvas').getContext('2d').clearRect(0,0,$('canvas').width,$('canvas').height); }
}
function renderClasses() {
  $('classes').replaceChildren(...state.project.class_names.map((name,i) => {
    const classId = projectClassIds()[i];
    const row = document.createElement('div'); row.className = 'class-row';
    const number = document.createElement('input'); number.type = 'number'; number.min = '0'; number.max = '999999'; number.value = classId; number.setAttribute('aria-label',`类别 ${name} 的编号`);
    const input = document.createElement('input'); input.value = name; input.title = '修改类别名称';
    const save = document.createElement('button'); save.textContent = '保存';
    save.onclick = async () => {
      try {
        if (state.predictBusy || state.enterBusy || state.drawing || state.classBusy) return;
        state.classBusy = true; save.disabled = true;
        if (number.value === '' || !number.checkValidity()) throw new Error('请输入有效的类别编号');
        await saveCurrentImage();
        const result = await api(`/api/projects/${state.project.id}/classes/${classId}`,{method:'PUT',...jsonOptions({name:input.value,class_id:Number(number.value)})});
        state.project.class_names = result.class_names; state.project.class_ids = result.class_ids;
        state.activeClass = result.class_id;
        const current = state.current; state.current = null; state.history = [];
        if (current) await selectImage(current,false);
        renderClasses(); await refreshTraining(); message(`类别已更新为 ${result.class_id}: ${result.name}`);
      } catch(e) { message(e.message); }
      finally { state.classBusy = false; save.disabled = false; renderCurrentClass(); }
    };
    row.append(number,input,save); return row;
  }));
  renderCurrentClass();
}
function renderCurrentClass() {
  const select = $('currentClass');
  const names = state.project?.class_names || [];
  const ids = projectClassIds();
  const signature = JSON.stringify([names,ids]);
  if (select.dataset.classes !== signature) {
    select.replaceChildren(...names.map((name,i) => new Option(`${ids[i]}: ${name}`,ids[i])));
    select.dataset.classes = signature;
  }
  const selected = state.annotations[state.selected];
  if (selected) state.activeClass = selected.class_id;
  if (!ids.includes(state.activeClass)) state.activeClass = ids[0] ?? 0;
  select.value = String(state.activeClass);
  if (document.activeElement !== $('currentClassId')) $('currentClassId').value = String(state.activeClass);
  select.disabled = !names.length || Boolean(state.crop || state.predictBusy || state.enterBusy || state.drawing || state.classBusy);
  $('currentClassId').disabled = select.disabled; $('applyClassId').disabled = select.disabled;
}
function applyCurrentClass(value) {
  if (!state.project || state.crop || state.predictBusy || state.enterBusy || state.drawing) return;
  const classId = Number(value);
  if (!Number.isInteger(classId) || !projectClassIds().includes(classId)) { message('该编号尚未定义，请先在左侧新增或修改类别编号'); return; }
  state.activeClass = classId;
  const box = state.annotations[state.selected];
  if (box && box.class_id !== classId) {
    snapshot(); box.class_id = classId; box.source = 'human'; box.status = 'edited';
    $('saveStatus').textContent = '有未保存修改';
  }
  renderBoxList(); renderCurrentClass(); draw();
  message(`当前类别 ${classId}: ${className(classId)}${box ? '（已修改选中框）' : '（用于新框）'}`);
}
function renderBoxList() {
  const list = $('boxList');
  list.replaceChildren(...state.annotations.map((box, index) => {
    const row = document.createElement('div'); row.className = `box-row ${state.selected === index ? 'selected' : ''}`;
    const title = document.createElement('button'); title.className = 'box-select'; title.textContent = `框 ${index + 1} · ${className(box.class_id)}`;
    title.onclick = () => { state.selected = index; renderBoxList(); renderCurrentClass(); draw(); };
    const remove = document.createElement('button'); remove.className = 'box-remove'; remove.textContent = '×'; remove.title = '删除此框';
    remove.onclick = event => { event.stopPropagation(); if (state.predictBusy) return; if (state.selected !== index) state.selected = index; snapshot(); state.annotations.splice(index, 1); state.selected = -1; renderBoxList(); renderCurrentClass(); draw(); };
    row.onclick = event => { if (event.target !== remove) { state.selected = index; renderBoxList(); renderCurrentClass(); draw(); } };
    row.append(title, remove); return row;
  }));
}
function renderImages() {
  $('images').replaceChildren(...state.images.map(image => {
    const item = document.createElement('div'); item.className = `image-item ${state.current === image.id ? 'active' : ''}`;
    const name = document.createElement('span'); name.textContent = image.filename;
    const reviewed = !image.needs_crop && (['edited','approved'].includes(image.status) || (image.status === 'labeled' && image.box_count > 0));
    const status = document.createElement('span'); status.className = reviewed ? 'image-check' : 'image-state';
    status.textContent = image.needs_crop ? '待裁剪' : image.status === 'predicted' ? '候选·待审核' : reviewed ? '✓' : image.status === 'invalid' ? '标签错误' : '未标注';
    item.append(name, status); item.onclick = () => selectImage(image.id); return item;
  }));
}
async function saveCurrentImage(force = false) {
  if (state.drawing) throw new Error('请先完成当前框的绘制');
  if (!state.current || !state.project || state.crop) return true;
  if (!force && state.history.length === 0) return true;
  const result = await api(`/api/projects/${state.project.id}/images/${state.current}/annotations`, {
    method: 'PUT', ...jsonOptions({version: state.version, boxes: state.annotations})
  });
  state.version = result.version; state.history = [];
  const current = state.images.find(image => image.id === state.current);
  if (current) { current.status = 'edited'; current.box_count = state.annotations.length; }
  renderImages();
  return true;
}
async function selectImage(id, saveBefore = true) {
  if (state.drawing || ((state.predictBusy || state.enterBusy) && saveBefore)) return;
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
  if (state.predictBusy || state.enterBusy || !state.images.length || !state.current) return;
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
  if (state.drawing) { const b = state.drawing; ctx.save(); ctx.strokeStyle = '#ffdf5d'; ctx.lineWidth = Math.max(2,canvas.width/400); ctx.setLineDash([10,6]); ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1); ctx.restore(); }
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
  if (!state.current || state.predictBusy || state.enterBusy || state.drawing) return;
  if (state.crop) { if (event.button === 0) dragCrop(event); return; }
  if (![0,2].includes(event.button)) return;
  event.preventDefault(); $('canvas').focus({preventScroll:true});
  const p = canvasPoint(event);
  if (event.button === 2) {
    state.selected = state.annotations.findLastIndex(b => p.x >= b.x1 && p.x <= b.x2 && p.y >= b.y1 && p.y <= b.y2);
    renderBoxList(); draw(); return;
  }
  const canvas = $('canvas'), imageId = state.current, projectId = state.project.id;
  const box = {class_id:Number($('currentClass').value) || 0, x1:p.x,y1:p.y,x2:p.x,y2:p.y,source:'human',status:'edited'};
  state.drawing = box; state.selected = -1;
  const move = e => {
    if (e.pointerId !== event.pointerId) return;
    const q = canvasPoint(e);
    box.x1 = Math.max(0,Math.min(canvas.width,p.x,q.x)); box.x2 = Math.max(0,Math.min(canvas.width,Math.max(p.x,q.x)));
    box.y1 = Math.max(0,Math.min(canvas.height,p.y,q.y)); box.y2 = Math.max(0,Math.min(canvas.height,Math.max(p.y,q.y)));
    draw();
  };
  const finish = (e, cancelled = false) => {
    if (e && e.pointerId !== event.pointerId) return;
    if (!cancelled && e) move(e);
    window.removeEventListener('pointermove',move); window.removeEventListener('pointerup',up); window.removeEventListener('pointercancel',cancel);
    window.removeEventListener('blur',cancel); canvas.removeEventListener('lostpointercapture',cancel);
    state.drawing = null; state.cancelDrawing = null;
    if (canvas.hasPointerCapture?.(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    if (!cancelled && state.current === imageId && state.project.id === projectId && box.x2-box.x1 >= 2 && box.y2-box.y1 >= 2) {
      snapshot(); state.annotations.push(box); state.selected = state.annotations.length-1;
      $('saveStatus').textContent = '有未保存修改';
    }
    renderBoxList(); draw();
  };
  const up = e => finish(e);
  const cancel = e => finish(e?.pointerId == null ? null : e,true);
  state.cancelDrawing = () => finish(null,true);
  window.addEventListener('pointermove',move); window.addEventListener('pointerup',up); window.addEventListener('pointercancel',cancel);
  window.addEventListener('blur',cancel); canvas.addEventListener('lostpointercapture',cancel);
  canvas.setPointerCapture?.(event.pointerId);
  renderBoxList(); draw();
});
$('editLabelFile').onclick = async () => {
  if (!state.current || state.crop || state.predictBusy || state.enterBusy || state.drawing) return;
  state.enterBusy = true;
  try {
    await saveCurrentImage();
    const data = await api(`/api/projects/${state.project.id}/images/${state.current}/label-file`);
    state.labelFileVersion = data.version;
    $('labelFilename').textContent = data.filename;
    $('labelText').value = data.text; $('labelError').textContent = '';
    $('labelEditor').showModal(); $('labelText').focus();
  } catch (e) { message(e.message); }
  finally { state.enterBusy = false; }
};
$('cancelLabelText').onclick = () => $('labelEditor').close();
$('labelEditor').addEventListener('cancel', e => { if ($('saveLabelText').disabled) e.preventDefault(); });
$('saveLabelText').onclick = async () => {
  $('saveLabelText').disabled = true; $('cancelLabelText').disabled = true;
  try {
    const result = await api(`/api/projects/${state.project.id}/images/${state.current}/label-file`, {method:'PUT', ...jsonOptions({version:state.labelFileVersion,text:$('labelText').value})});
    state.version = result.version; state.annotations = result.boxes; state.history = []; state.selected = -1;
    const record = state.project.images[state.current]; record.version = result.version; record.status = 'edited'; record.box_count = result.boxes.length;
    renderImages(); renderBoxList(); await draw(); await checkSaveStatus();
    $('labelEditor').close(); message('标签文件已保存');
  } catch(e) { $('labelError').textContent = e.message; }
  finally { $('saveLabelText').disabled = false; $('cancelLabelText').disabled = false; }
};
const confidenceKey = 'ai-label-local.confidence';
try { const value = localStorage.getItem(confidenceKey); if (value !== null && Number.isFinite(Number(value)) && Number(value) >= 0 && Number(value) <= 1) $('confidence').value = value; } catch (_) {}
function updateConfidence() { $('confidenceValue').textContent = Number($('confidence').value).toFixed(2); try { localStorage.setItem(confidenceKey, $('confidence').value); } catch (_) {} }
$('confidence').oninput = updateConfidence; updateConfidence();
function deleteSelectedBox() {
  if (!state.current || state.crop || state.predictBusy || state.enterBusy || state.drawing || state.selected < 0 || state.selected >= state.annotations.length) return;
  snapshot(); state.annotations.splice(state.selected,1); state.selected = -1;
  $('saveStatus').textContent = '有未保存修改'; renderBoxList(); renderCurrentClass(); draw();
}
$('deleteBox').onclick = deleteSelectedBox;
$('currentClass').onchange = event => applyCurrentClass(event.target.value);
function applyTypedClass() { const value = $('currentClassId').value; if (!/^\d+$/.test(value)) { message('请输入非负整数类别编号'); return; } applyCurrentClass(value); }
$('applyClassId').onclick = applyTypedClass;
$('currentClassId').onkeydown = e => { if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); e.stopPropagation(); applyTypedClass(); } };
$('undo').onclick = () => { const previous = state.history.pop(); if (previous) { state.annotations = JSON.parse(previous); state.selected = -1; renderBoxList(); draw(); } };
$('save').onclick = async () => { try { await saveCurrentImage(true); await checkSaveStatus(); message('标注已保存'); } catch (e) { message(e.message); } };
$('newProject').onclick = async () => { const name = prompt('项目名称', '我的数据集'); const classes = prompt('类别名（逗号分隔）', 'object'); if (!name || !classes) return; try { await api('/api/projects', {method:'POST', ...jsonOptions({name, ...parseClassSpec(classes), crop_mode: 'manual_crop'})}); await loadProjects(); message('项目已创建'); } catch(e) { message(e.message); } };
$('projectSelect').onchange = e => selectProject(e.target.value);
$('addClass').onclick = async () => {
  if (!state.project) return;
  const input = $('newClass'); const name = input.value.trim();
  if (!name) { message('请输入类别名称'); return; }
  try {
    const result = await api(`/api/projects/${state.project.id}/classes`, {method: 'POST', ...jsonOptions({name,class_id:$('newClassId').value === '' ? null : Number($('newClassId').value)})});
    state.project.class_names = result.class_names; state.project.class_ids = result.class_ids; state.activeClass = result.class_id; renderClasses(); input.value = ''; $('newClassId').value = '';
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
function setPredictionBusy(busy) {
  state.predictBusy = busy;
  ['renameProject','downloadImage','manageModels','currentClassId','applyClassId','trainedModel','trainingTab','browseServer','predict','predictCurrent','model','confidence','projectSelect','newProject','import','export','save','completeImage','editLabelFile','deleteBox','undo','currentClass','addClass','confirmCrop'].forEach(id => $(id).disabled = busy);
  if (!busy) { prepareCrop(); renderCurrentClass(); }
}
function showPredictionProgress(job) {
  const total = job.total || 0, completed = job.completed || 0;
  const percent = total ? Math.round(completed / total * 100) : 0;
  $('predictProgress').value = percent;
  const names = {queued:'排队中',running:'正在标注',completed:'已完成',failed:'处理结束（有失败）',cancelled:'已取消'};
  const errors = job.errors || [];
  $('job').className = errors.length ? 'job-error' : '';
  $('job').textContent = `${names[job.status] || job.status} · ${completed}/${total} 张 · ${percent}%${errors.length ? `\n失败 ${errors.length} 张：${errors[0]}` : ''}`;
}
async function runPrediction(scope) {
  if (!state.project || state.predictBusy || state.drawing) return;
  if (scope === 'current' && (!state.current || state.crop)) { message('请先选择图片并确认裁剪'); return; }
  const projectId = state.project.id, imageId = state.current;
  try {
    if (scope !== 'current') await saveCurrentImage();
    setPredictionBusy(true); $('job').className = ''; $('job').textContent = '正在加载模型…'; $('predictProgress').removeAttribute('value');
    let job = await api(`/api/projects/${projectId}/predict`, {method:'POST', ...jsonOptions({scope,image_id:imageId,model_path:$('model').value,confidence:Number($('confidence').value),image_size:state.selectedModelImageSize || state.defaultImageSize || 640})});
    showPredictionProgress(job);
    while (!['completed','failed','cancelled'].includes(job.status)) {
      await new Promise(resolve => setTimeout(resolve,400));
      job = await api(`/api/jobs/${job.id}`); showPredictionProgress(job);
    }
    state.project = await api(`/api/projects/${projectId}`); state.images = Object.values(state.project.images);
    state.current = null; state.history = []; state.annotations = []; state.selected = -1; imageCache.clear();
    renderClasses(); renderImages();
    if (imageId && state.project.images[imageId]) await selectImage(imageId,false);
    message(job.status === 'completed' ? '智能打标完成，结果为待审核候选' : '智能打标结束，请查看失败信息');
  } catch(e) {
    $('predictProgress').value = 0; $('job').className = 'job-error'; $('job').textContent = `任务失败：${e.message}`; message(e.message);
  } finally { setPredictionBusy(false); }
}
$('completeImage').onclick = completeCurrentImage;
$('predictCurrent').onclick = () => runPrediction('current');
$('predict').onclick = () => runPrediction('unfinished');
$('export').onclick = async () => { if (!state.project) return; try { await saveCurrentImage(); const result = await api(`/api/projects/${state.project.id}/export`, {method:'POST', ...jsonOptions({})}); message(`导出完成：${result.counts.train}/${result.counts.val}/${result.counts.test}`); } catch(e) { message(e.message); } };
async function completeCurrentImage() {
  if (!state.current || state.crop || state.predictBusy || state.enterBusy || state.drawing) return;
  state.enterBusy = true;
  try {
    await saveCurrentImage(true); await checkSaveStatus();
    const index = state.images.findIndex(image => image.id === state.current);
    if (index >= 0 && index + 1 < state.images.length) {
      await selectImage(state.images[index+1].id,false);
      message('上一张已保存并审核完成，已进入下一张');
    } else { message('已保存并审核完成，当前是最后一张'); }
  } catch(e) { message(`保存失败：${e.message}`); }
  finally { state.enterBusy = false; }
}
document.addEventListener('keydown', event => {
  if (state.activeTab === 'training' || $('labelEditor').open || $('serverBrowser').open || $('modelsDialog').open) return;
  const target = event.target;
  if (state.drawing) { if (event.key === 'Escape') { event.preventDefault(); state.cancelDrawing?.(); } return; }
  const typing = target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement || target.isContentEditable || (target instanceof HTMLInputElement && target.type !== 'range');
  if (/^[0-9]$/.test(event.key) && (!typing || target === $('currentClass')) && !event.isComposing && !event.repeat && !event.ctrlKey && !event.altKey && !event.metaKey) {
    event.preventDefault(); applyCurrentClass(Number(event.key)); return;
  }
  if (event.key === 'Delete' && !typing && !event.isComposing && !event.ctrlKey && !event.altKey && !event.metaKey) { event.preventDefault(); deleteSelectedBox(); return; }
  if (event.key === 'Enter' && !event.isComposing && !event.repeat && !event.ctrlKey && !event.altKey && !event.metaKey && !typing) {
    event.preventDefault();
    if (state.predictBusy || state.enterBusy || state.drawing) return;
    if (state.crop) { if (!$('confirmCrop').disabled) $('confirmCrop').click(); }
    else { completeCurrentImage(); }
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
// Initialized after both page controllers are ready.

function prepareCrop() {
 const r=state.images.find(x=>x.id===state.current), side=r?Math.min(r.width,r.height):0;
 state.crop=r?.needs_crop?{left:Math.floor((r.width-side)/2),top:Math.floor((r.height-side)/2),side,width:r.width,height:r.height}:null;
 $('cropPanel').hidden=!state.crop; ['editLabelFile','deleteBox','save','completeImage','undo','currentClass'].forEach(id=>$(id).disabled=Boolean(state.crop)); $('boxList').hidden=Boolean(state.crop);
 if(state.crop){const c=state.crop,h=c.width>c.height; $('cropAxis').textContent=h?'水平位置':'垂直位置'; $('cropPosition').max=Math.max(c.width,c.height)-side; $('cropPosition').value=h?c.left:c.top; $('cropInfo').textContent=`${c.width} × ${c.height} → ${side} × ${side}`; $('selection').textContent='待确认裁剪区域';}
}
function setCropPosition(value){if(!state.crop)return; const c=state.crop,p=Math.round(Math.max(0,Math.min(Number(value),Math.max(c.width,c.height)-c.side))); c.left=c.width>c.height?p:0; c.top=c.height>c.width?p:0; $('cropPosition').value=p; draw();}
function drawCrop(ctx,canvas){const c=state.crop; ctx.fillStyle='rgba(0,0,0,.62)'; ctx.beginPath();ctx.rect(0,0,canvas.width,canvas.height);ctx.rect(c.left,c.top,c.side,c.side);ctx.fill('evenodd');ctx.strokeStyle='#4ee7ed';ctx.lineWidth=Math.max(3,canvas.width/350);ctx.strokeRect(c.left,c.top,c.side,c.side);ctx.save();ctx.lineWidth=Math.max(1,canvas.width/900);ctx.setLineDash([12,8]);for(const n of [1,2]){ctx.beginPath();ctx.moveTo(c.left+c.side*n/3,c.top);ctx.lineTo(c.left+c.side*n/3,c.top+c.side);ctx.moveTo(c.left,c.top+c.side*n/3);ctx.lineTo(c.left+c.side,c.top+c.side*n/3);ctx.stroke();}ctx.restore();}
function dragCrop(event){const start=canvasPoint(event),horizontal=state.crop.width>state.crop.height,initial=horizontal?state.crop.left:state.crop.top;const move=e=>{if(!state.crop)return;const q=canvasPoint(e);setCropPosition(initial+(horizontal?q.x-start.x:q.y-start.y));};const up=()=>{window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);window.removeEventListener('pointercancel',up);};window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);window.addEventListener('pointercancel',up);}
$('cropPosition').oninput=e=>setCropPosition(e.target.value);
$('confirmCrop').onclick=async()=>{if(!state.crop)return;$('confirmCrop').disabled=true;try{const result=await api(`/api/projects/${state.project.id}/images/${state.current}/crop`,{method:'POST',...jsonOptions({version:state.version,left:state.crop.left,top:state.crop.top})});const i=state.images.findIndex(x=>x.id===state.current);state.images[i]=result.image;state.project.images[state.current]=result.image;state.annotations=result.boxes;state.version=result.version;state.history=[];state.selected=-1;state.crosshair=null;prepareCrop();renderImages();renderBoxList();await draw();await checkSaveStatus();message('数据集图片和标签已永久裁剪，可以开始标注');}catch(e){message(e.message);}finally{$('confirmCrop').disabled=false;}};

let serverSelectedPath = '', serverImportBusy = false;
async function browseServer(path = '') {
  $('serverError').textContent = '';
  try {
    const data = await api(`/api/server/directories?path=${encodeURIComponent(path)}`);
    serverSelectedPath = data.path; $('serverPath').value = data.path;
    $('serverUp').onclick = () => browseServer(data.parent || '');
    $('serverDirectoryInfo').textContent = data.path ? `当前目录有 ${data.image_count} 张图片（不含子目录）` : '请选择服务器上的目录';
    $('serverDirectories').replaceChildren(...data.directories.map(d => { const b = document.createElement('button'); b.textContent = `📁 ${d.name}`; b.onclick = () => browseServer(d.path); return b; }));
    $('serverImport').disabled = !data.path;
    if (data.path) $('serverProjectName').value = data.path.split(/[\\/]/).filter(Boolean).pop() || '服务器数据集';
  } catch(e) { $('serverError').textContent = e.message; $('serverImport').disabled = true; }
}
$('browseServer').onclick = async () => {
  if (state.predictBusy || state.enterBusy || state.drawing) return;
  try { await saveCurrentImage(); } catch(e) { message(e.message); return; }
  $('serverClasses').value = state.project?.class_names.map((n,i) => `${projectClassIds()[i]}: ${n}`).join(',') || 'object';
  $('serverBrowser').showModal(); await browseServer();
};
$('serverGo').onclick = () => browseServer($('serverPath').value.trim());
$('serverPath').onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); browseServer($('serverPath').value.trim()); } };
$('serverRoots').onclick = () => browseServer();
$('serverClose').onclick = () => $('serverBrowser').close();
$('serverBrowser').addEventListener('cancel', e => { if (serverImportBusy) e.preventDefault(); });
$('serverImport').onclick = async () => {
  if (!serverSelectedPath || serverImportBusy) return;
  serverImportBusy = true; $('serverError').textContent = '';
  const controls = [...$('serverBrowser').querySelectorAll('button,input')]; controls.forEach(el => el.disabled = true);
  $('serverImportProgress').hidden = false;
  try {
    const result = await api('/api/server/directories/import', {method:'POST', ...jsonOptions({path:serverSelectedPath,name:$('serverProjectName').value.trim(),...parseClassSpec($('serverClasses').value),recursive:$('serverRecursive').checked})});
    await loadProjects(); $('projectSelect').value = result.project_id; await selectProject(result.project_id);
    $('serverBrowser').close(); message(`服务器目录已导入新项目：${result.count} 张图片`);
  } catch(e) { $('serverError').textContent = typeof e.message === 'string' ? e.message : '导入失败'; }
  finally { serverImportBusy = false; controls.forEach(el => el.disabled = false); $('serverImportProgress').hidden = true; }
};
let trainingRuns = [], selectedTrainingRun = '', trainingPoll = null, trainingRefreshing = false, trainingRefreshPending = false;
const trainingNames = {preparing:'准备数据快照',running:'正在训练',completed:'训练完成',failed:'训练失败',cancelled:'已停止',stopping:'正在停止'};
async function showPage(tab) {
  if (state.predictBusy || state.enterBusy || state.drawing) return;
  try { await saveCurrentImage(); } catch(e) { message(e.message); return; }
  state.activeTab = tab;
  $('annotationPage').hidden = tab === 'training'; $('trainingPage').hidden = tab !== 'training';
  $('annotationTab').setAttribute('aria-pressed', String(tab !== 'training')); $('trainingTab').setAttribute('aria-pressed', String(tab === 'training'));
  if (tab === 'training') await refreshTraining();
}
$('annotationTab').onclick = () => showPage('annotation');
$('trainingTab').onclick = () => showPage('training').catch(e => { $('trainingError').textContent = e.message; });
$('trainingProject').onchange = async e => { try { selectedTrainingRun = ''; await selectProject(e.target.value); } catch(err) { $('trainingError').textContent = err.message; } };
function compatibleModel(run) { return run.model_available && JSON.stringify(run.class_names) === JSON.stringify(state.project?.class_names); }
function chooseModel(run) {
  if (!compatibleModel(run)) { message('模型文件不存在或类别与当前项目不一致'); return; }
  $('model').value = run.model_path; $('trainedModel').value = run.id; state.selectedModelImageSize = run.params.imgsz;
  try { localStorage.setItem(`ai-label-local.model.${state.project.id}`,run.id); } catch (_) {}
  message(`已选择 ${run.name} · 推理分辨率 ${run.params.imgsz}`);
}
$('trainedModel').onchange = e => {
  const run = trainingRuns.find(r => r.id === e.target.value);
  if (run) chooseModel(run);
  else { state.selectedModelImageSize = null; $('model').value = state.defaultModelPath || ''; try { localStorage.removeItem(`ai-label-local.model.${state.project.id}`); } catch (_) {} }
};
$('model').addEventListener('input', () => { $('trainedModel').value = ''; state.selectedModelImageSize = null; try { localStorage.removeItem(`ai-label-local.model.${state.project?.id}`); } catch (_) {} });
async function refreshTraining() {
  if (!state.project) return;
  if (trainingRefreshing) { trainingRefreshPending = true; return; }
  trainingRefreshing = true;
  const projectId = state.project.id;
  try {
    const data = await api(`/api/projects/${projectId}/training`);
    if (state.project?.id !== projectId) return;
    trainingRuns = data.runs;
    if (['Failed to fetch','NetworkError when attempting to fetch resource.'].includes($('trainingError').textContent)) $('trainingError').textContent = '';
    const summary = data.summary;
    $('trainingSummary').textContent = `可训练：${summary.reviewed} 张已审核图片 / 共 ${summary.total} 张；排除 ${summary.excluded} 张待审核或待裁剪图片。类别：${summary.class_names.join('、')}`;
    $('trainedModel').replaceChildren(new Option('默认或手动指定模型',''), ...trainingRuns.filter(compatibleModel).map(r => new Option(`${r.name} · ${r.created_at.slice(0,16).replace('T',' ')} UTC`,r.id)));
    let remembered;
    try { remembered = localStorage.getItem(`ai-label-local.model.${projectId}`); } catch (_) {}
    const current = trainingRuns.find(r => compatibleModel(r) && r.model_path === $('model').value);
    if (current) $('trainedModel').value = current.id;
    else if (state.trainingModelProject !== projectId) {
      state.trainingModelProject = projectId; state.selectedModelImageSize = null;
      const saved = trainingRuns.find(r => r.id === remembered && compatibleModel(r));
      if (saved) chooseModel(saved);
      else $('model').value = state.defaultModelPath || $('model').value;
    }
    if (!trainingRuns.some(r => r.id === selectedTrainingRun)) selectedTrainingRun = trainingRuns[0]?.id || '';
    $('trainingRun').replaceChildren(...trainingRuns.map(r => new Option(`${r.name} · ${trainingNames[r.status] || r.status} · ${r.created_at.slice(0,16)} UTC`,r.id)));
    $('trainingRun').value = selectedTrainingRun;
    $('startTraining').disabled = summary.reviewed < 2 || trainingRuns.some(r => ['preparing','running','stopping'].includes(r.status)) || state.trainingAvailable === false;
    const prefs = trainingParameterCache.get(projectId);
    if (prefs?.dirty && !trainingParametersTimer) flushTrainingParameters(projectId).catch(() => {});
    if (selectedTrainingRun) await renderTrainingRun(projectId, selectedTrainingRun);
    else { $('trainingStatus').textContent = '尚无训练任务'; $('trainingLog').textContent = ''; $('trainingProgress').value = 0; $('trainingMetrics').textContent = ''; $('stopTraining').disabled = true; $('useTrainedModel').disabled = true; }
  } finally { trainingRefreshing = false; if (trainingRefreshPending) { trainingRefreshPending = false; refreshTraining().catch(e => { $('trainingError').textContent = e.message; }); } }
}
async function renderTrainingRun(projectId, runId) {
  const run = await api(`/api/projects/${projectId}/training/${runId}`);
  if (state.project?.id !== projectId || selectedTrainingRun !== runId) return;
  const total = run.params.epochs, epoch = run.epoch || 0;
  $('trainingProgress').value = run.status === 'completed' ? 100 : Math.min(100,epoch/total*100);
  const count = run.counts ? ` · 训练 ${run.counts.train} 张 / 验证 ${run.counts.val} 张` : '';
  $('trainingStatus').textContent = `${trainingNames[run.status] || run.status} · ${epoch}/${total} 轮${count}${run.error ? ` · ${run.error}` : ''}${run.status === 'completed' && epoch < total ? ' · 已提前停止' : ''}`;
  $('trainingMetrics').textContent = Object.entries(run.metrics || {}).map(([k,v]) => `${k}: ${Number(v).toFixed(4)}`).join(' · ');
  const log = $('trainingLog'), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight-40;
  log.textContent = run.log || '等待训练日志…'; if (atEnd) log.scrollTop = log.scrollHeight;
  $('stopTraining').disabled = !['preparing','running'].includes(run.status);
  $('useTrainedModel').disabled = !compatibleModel(run);
  $('useTrainedModel').onclick = () => { chooseModel(run); showPage('annotation'); };
}
$('trainingRun').onchange = async e => { selectedTrainingRun = e.target.value; try { await renderTrainingRun(state.project.id,selectedTrainingRun); } catch(err) { $('trainingError').textContent = err.message; } };
$('trainingForm').onsubmit = async e => {
  e.preventDefault(); if (!state.project || $('startTraining').disabled) return;
  $('startTraining').disabled = true; $('trainingError').textContent = '';
  const params = trainingParams(captureTrainingRaw());
  try {
    await flushTrainingParameters(state.project.id);
    await saveCurrentImage();
    const run = await api(`/api/projects/${state.project.id}/training`,{method:'POST',...jsonOptions(params)});
    selectedTrainingRun = run.id; await refreshTraining();
  } catch(err) { $('trainingError').textContent = err.message; $('startTraining').disabled = false; }
};
$('stopTraining').onclick = async () => {
  $('stopTraining').disabled = true;
  try { await api(`/api/projects/${state.project.id}/training/${selectedTrainingRun}/stop`,{method:'POST'}); await refreshTraining(); }
  catch(e) { $('trainingError').textContent = e.message; }
};
async function initializeTraining() {
  try {
    const env = await api('/api/training/environment'); state.trainingAvailable = env.available;
    $('trainDevice').replaceChildren(...env.devices.map(d => new Option(d.name,d.id)));
    $('trainingEnvironment').textContent = env.available ? '训练运行在标注服务所在主机。同一时间支持一个训练任务。优化器 AdamW；显存不足时请降低 Batch 或分辨率。' : '当前主机缺少训练依赖，请安装 pt 扩展。';
  } catch(e) { $('trainingError').textContent = e.message; }
  await loadProjects();
  trainingPoll = setInterval(() => { if (state.project && (state.activeTab === 'training' || trainingRuns.some(r => ['preparing','running'].includes(r.status)))) refreshTraining().catch(e => { $('trainingError').textContent = e.message; }); },2000);
}
// Training parameter controller is initialized below.

$('renameProject').onclick = async () => {
  if (!state.project || state.predictBusy || state.drawing) return;
  const name = prompt('新的项目名称',state.project.name); if (!name?.trim()) return;
  try { await saveCurrentImage(); const id=state.project.id; await api(`/api/projects/${id}/name`,{method:'PUT',...jsonOptions({name:name.trim()})}); await loadProjects(); await selectProject(id); message('项目已改名'); } catch(e) { message(e.message); }
};
$('downloadImage').onclick = () => {
  if (!state.current || !state.project) { message('请先选择图片'); return; }
  const anchor = document.createElement('a'); anchor.href = `/api/projects/${state.project.id}/images/${state.current}/download`;
  anchor.download = ''; document.body.append(anchor); anchor.click(); anchor.remove();
};
let modelsBusy = false;
async function refreshManagedModels() {
  const data = await api('/api/models');
  $('managedModels').replaceChildren(...data.models.map(model => {
    const row = document.createElement('div'); row.className = 'managed-model';
    const title = document.createElement('strong'); title.textContent = `${model.name}${model.default ? '（默认）':''} · ${(model.bytes/1048576).toFixed(1)} MB`;
    const path = document.createElement('p'); path.textContent = model.path;
    const details = document.createElement('p'); details.className='hint'; details.textContent = model.kind==='training' ? `${model.project_name} · ${model.epochs ?? '?'} 轮 · ${model.imgsz} 分辨率 · ${model.created_at ? new Date(model.created_at).toLocaleString() : ''}` : '项目内模型文件';
    const actions = document.createElement('div');
    const use = document.createElement('button'); use.textContent = '用于当前打标';
    use.disabled = Boolean(model.class_names && JSON.stringify(model.class_names)!==JSON.stringify(state.project?.class_names));
    use.onclick = () => { $('model').value = model.path; state.selectedModelImageSize = model.imgsz; $('trainedModel').value = model.run_id || ''; $('modelsDialog').close(); message(`已选择模型 ${model.name}`); };
    const makeDefault = document.createElement('button'); makeDefault.textContent = '设为默认'; makeDefault.disabled = model.default;
    makeDefault.onclick = async () => { if (modelsBusy) return; modelsBusy = true; try { const result = await api(`/api/models/${model.id}/default`,{method:'PUT'}); state.defaultModelPath = result.default_model_path; state.defaultImageSize = result.default_image_size; await refreshManagedModels(); } catch(e) { $('modelManagerError').textContent=e.message; } finally { modelsBusy=false; } };
    const remove = document.createElement('button'); remove.textContent = '删除模型'; remove.disabled = Boolean(model.delete_blocked); remove.title = model.delete_blocked || '删除权重，保留训练记录及数据';
    remove.onclick = async () => {
      if (modelsBusy || !confirm(`删除模型“${model.name}”？${model.kind==='training'?'将删除 best.pt 和 last.pt。':''}图片、标签及训练日志会保留。`)) return;
      modelsBusy=true; $('modelManagerError').textContent='';
      try {
        await api(`/api/models/${model.id}`,{method:'DELETE'});
        const config=await api('/api/config'); state.defaultModelPath=config.default_model_path;
        if ($('model').value===model.path) { $('model').value=config.default_model_path; state.selectedModelImageSize=null; $('trainedModel').value=''; try { localStorage.removeItem(`ai-label-local.model.${state.project.id}`); } catch (_) {} }
        if ($('trainModel').value===model.path) { $('trainModel').value=config.default_model_path; noteTrainingParameterChange(); }
        await refreshTraining(); await refreshManagedModels(); message('模型权重已删除');
      } catch(e) { $('modelManagerError').textContent=e.message; }
      finally { modelsBusy=false; }
    };
    actions.append(use,makeDefault,remove); row.append(title,details,path,actions);
    if (model.delete_blocked) { const reason=document.createElement('p'); reason.className='hint'; reason.textContent=model.delete_blocked; row.append(reason); }
    return row;
  }));
  if (!data.models.length) $('managedModels').textContent='暂无可管理模型';
}
$('manageModels').onclick = async () => { $('modelsDialog').showModal(); $('modelManagerError').textContent=''; try { await refreshManagedModels(); } catch(e) { $('modelManagerError').textContent=e.message; } };
$('closeModels').onclick = () => { if (!modelsBusy) $('modelsDialog').close(); };
$('modelsDialog').addEventListener('cancel', e => { if (modelsBusy) e.preventDefault(); });
const trainingParamFields = {name:'trainName',model_path:'trainModel',device:'trainDevice',epochs:'trainEpochs',imgsz:'trainImgsz',batch:'trainBatch',lr0:'trainLr',patience:'trainPatience',val_ratio:'trainVal',seed:'trainSeed',hsv_s:'trainSaturation',hsv_v:'trainBrightness',fliplr:'trainFlip',mosaic:'trainMosaic'};
const trainingParameterCache = new Map();
const trainingFormDefaults = Object.fromEntries(Object.entries(trainingParamFields).map(([key,id]) => [key,$(id).value]));
let trainingParametersTimer = null;
function captureTrainingRaw() { return Object.fromEntries(Object.entries(trainingParamFields).map(([key,id]) => [key,$(id).value])); }
function trainingParams(raw) { return Object.fromEntries(Object.entries(raw).map(([key,value]) => [key,['name','model_path','device'].includes(key) ? value.trim() : Number(value)])); }
function validTrainingPreferences() { return Object.entries(trainingParamFields).every(([key,id]) => key==='model_path' || $(id).checkValidity()); }
function trainingDraftKey(id) { return `ai-label-local.training-draft.${id}`; }
function setTrainingSaveStatus(id,text) { if (state.trainingFormProject === id) $('trainingSettingsStatus').textContent=text; }
function saveTrainingDraft(id,entry) { try { localStorage.setItem(trainingDraftKey(id),JSON.stringify({raw:entry.raw,dirty:true})); } catch (_) {} }
function applyTrainingRaw(raw) {
  const device = String(raw.device);
  if (![...$('trainDevice').options].some(option => option.value===device)) $('trainDevice').add(new Option(`${device}（当前不可用，请调整设备）`,device));
  for (const [key,id] of Object.entries(trainingParamFields)) $(id).value = raw[key] ?? trainingFormDefaults[key];
}
async function restoreTrainingParameters(projectId) {
  if (state.trainingFormProject === projectId) return;
  clearTimeout(trainingParametersTimer); trainingParametersTimer = null;
  const entry = trainingParameterCache.get(projectId) || {raw:null,dirty:false,revision:0,pending:null};
  trainingParameterCache.set(projectId,entry);
  let saved = {params:trainingFormDefaults,source:'defaults'};
  try { saved = await api(`/api/projects/${projectId}/training-settings`); }
  catch(e) { if (!entry.raw) throw e; }
  if (state.project?.id !== projectId) return;
  try {
    const draft = JSON.parse(localStorage.getItem(trainingDraftKey(projectId)) || 'null');
    if (!entry.dirty && draft?.dirty && draft.raw) { entry.raw = {...saved.params,...draft.raw}; entry.dirty = true; entry.revision++; }
  } catch (_) {}
  if (!entry.dirty) {
    entry.raw = Object.fromEntries(Object.entries(saved.params).map(([key,value]) => [key,String(value)]));
    if (saved.source === 'defaults') entry.raw.device = $('trainDevice').options[0]?.value || 'cpu';
  }
  state.trainingFormProject = projectId;
  applyTrainingRaw(entry.raw);
  setTrainingSaveStatus(projectId,entry.dirty ? '已恢复本机未同步草稿' : saved.source==='last_run' ? '已恢复最近一次训练参数' : saved.source==='saved' ? '已恢复保存的训练参数' : '使用默认训练参数');
  if (entry.dirty && validTrainingPreferences()) flushTrainingParameters(projectId).catch(() => {});
}
function noteTrainingParameterChange() {
  const id = state.trainingFormProject;
  if (!id || state.project?.id !== id) return;
  const entry = trainingParameterCache.get(id);
  if (!entry) return;
  entry.raw = captureTrainingRaw(); entry.dirty = true; entry.revision++;
  saveTrainingDraft(id,entry);
  clearTimeout(trainingParametersTimer); trainingParametersTimer = null;
  if (!validTrainingPreferences()) { setTrainingSaveStatus(id,'参数未完整或无效，已保留本机草稿'); return; }
  setTrainingSaveStatus(id,'参数待保存…');
  trainingParametersTimer = setTimeout(() => { trainingParametersTimer=null; flushTrainingParameters(id).catch(() => {}); },500);
}
async function flushTrainingParameters(id = state.trainingFormProject) {
  const entry = trainingParameterCache.get(id);
  if (!entry?.dirty) return;
  if (entry.pending) { await entry.pending; if (entry.dirty) return flushTrainingParameters(id); return; }
  if (state.trainingFormProject===id && !validTrainingPreferences()) { setTrainingSaveStatus(id,'参数无效，已保留本机草稿'); throw new Error('请补全有效的训练参数'); }
  const revision = entry.revision, params = trainingParams(entry.raw);
  setTrainingSaveStatus(id,'正在保存训练参数…');
  entry.pending = api(`/api/projects/${id}/training-settings`,{method:'PUT',...jsonOptions(params)}).then(() => {
    if (entry.revision===revision) {
      entry.dirty=false;
      try { localStorage.removeItem(trainingDraftKey(id)); } catch (_) {}
      setTrainingSaveStatus(id,'训练参数已保存');
    }
  }).catch(e => { saveTrainingDraft(id,entry); setTrainingSaveStatus(id,`尚未同步，已保留本机草稿：${e.message}`); throw e; }).finally(() => { entry.pending=null; });
  await entry.pending;
  if (entry.dirty) return flushTrainingParameters(id);
}
$('trainingForm').addEventListener('input',noteTrainingParameterChange);
$('trainingForm').addEventListener('change',() => { noteTrainingParameterChange(); clearTimeout(trainingParametersTimer); trainingParametersTimer=null; if (validTrainingPreferences()) flushTrainingParameters().catch(() => {}); });
$('saveTrainingSettings').onclick = async () => {
  noteTrainingParameterChange(); clearTimeout(trainingParametersTimer); trainingParametersTimer=null;
  if (!validTrainingPreferences()) { setTrainingSaveStatus(state.trainingFormProject,'参数无效，已保留本机草稿'); return; }
  try { await flushTrainingParameters(); } catch(e) { $('trainingError').textContent=e.message; }
};
window.addEventListener('online',() => { for (const [id,entry] of trainingParameterCache) if (entry.dirty) flushTrainingParameters(id).catch(() => {}); });
window.addEventListener('pagehide',() => {
  const id=state.trainingFormProject, entry=trainingParameterCache.get(id);
  if (!entry?.dirty) return;
  saveTrainingDraft(id,entry);
  if (validTrainingPreferences()) fetch(`/api/projects/${id}/training-settings`,{method:'PUT',...jsonOptions(trainingParams(entry.raw)),keepalive:true}).catch(() => {});
});
initializeTraining().catch(e => message(e.message));

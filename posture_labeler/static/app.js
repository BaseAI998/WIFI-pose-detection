"use strict";

const $ = (id) => document.getElementById(id);
const names = {sitting: "坐着", standing: "站着", uncertain: "不确定", ignore: "忽略"};
const colors = {sitting: "#43b18a", standing: "#6399eb", uncertain: "#e5b34d", ignore: "#9ca8af"};
const state = {config: null, frames: [], annotations: new Map(), frame: null,
  personIndex: 0, image: null, busy: false, undo: null, step: 3, draftTimer: null,
  hardTargets: new Set(), hardBatch: null};
const keyOf = (frame, person) => `${frame}:${person}`;
const person = () => state.frame?.people[state.personIndex];
const currentKey = () => person() ? keyOf(state.frame.output_frame, person().person_id) : null;
const prefix = () => `posture-labeler:${state.config?.dataset_id}:`;

function storageGet(key, fallback = null) {
  try { const value = localStorage.getItem(prefix() + key); return value === null ? fallback : JSON.parse(value); }
  catch { return fallback; }
}
function storageSet(key, value) {
  try { localStorage.setItem(prefix() + key, JSON.stringify(value)); }
  catch { /* Server annotation writes remain durable if browser storage is unavailable. */ }
}
function storageRemove(key) {
  try { localStorage.removeItem(prefix() + key); } catch { /* no-op */ }
}
function annotation(frame = state.frame?.output_frame, id = person()?.person_id) {
  return state.annotations.get(keyOf(frame, id)) || {output_frame: frame, person_id: id,
    label: null, occlusion: "unmarked", note: "", revision: 0};
}
function showError(message) {
  $("error-text").textContent = message;
  $("error-banner").hidden = false;
  $("save-status").textContent = "操作未完成，请查看提示";
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { $("toast").hidden = true; }, 3000);
}
async function api(path, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 45000);
  try {
    const response = await fetch(path, {signal: controller.signal, cache: "no-store",
      ...(body === undefined ? {} : {method: "POST", headers: {
        "Content-Type": "application/json", "X-Annotation-Token": state.config.csrf_token},
        body: JSON.stringify(body)})});
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(data.error || `请求失败 (${response.status})`);
      error.status = response.status;
      throw error;
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("请求超时。若正在保存，结果可能已写入；请重新连接核对后再操作。");
    if (error instanceof TypeError) throw new Error("无法连接本地服务。请确认启动终端仍在运行，再点「重新连接」。未提交的备注保留在此浏览器。");
    throw error;
  } finally { clearTimeout(timeout); }
}
function lockUI(busy) {
  state.busy = busy;
  document.querySelectorAll("[data-lock], .frame-row, .person-button").forEach((el) => {
    el.disabled = busy || (!state.frame && el.hasAttribute("data-label"));
  });
  $("undo-button").disabled = busy || !state.undo;
  $("hard-only").disabled = busy || !state.hardTargets.size;
  $("scene").setAttribute("aria-busy", String(busy));
}
async function run(task) {
  if (state.busy) return;
  const focus = document.activeElement;
  const focusKey = currentKey();
  const caret = focus === $("note") ? [focus.selectionStart, focus.selectionEnd] : null;
  clearTimeout(state.draftTimer);
  lockUI(true);
  try { await task(); }
  catch (error) {
    showError(error.message);
    if (error.status === 409 || error.status === 403) {
      try {
        state.config = await api("/api/config");
        await refreshQueue();
        if (state.frame) {
          state.frame = await api(`/api/frame?id=${state.frame.output_frame}`);
          mergeAnnotations(state.frame.annotations);
          render();
        }
      } catch { /* Preserve the original failure and the current draft. */ }
    }
  } finally {
    lockUI(false);
    if (caret && currentKey() === focusKey) {
      focus.focus({preventScroll: true});
      focus.setSelectionRange(...caret);
    }
  }
}
function mergeAnnotations(rows) {
  for (const row of rows) state.annotations.set(keyOf(row.output_frame, row.person_id), row);
}
function rememberCursor() {
  if (person()) storageSet("cursor", {frame: state.frame.output_frame, person: person().person_id});
}
function isHardTarget(frame, id) {
  return state.hardTargets.has(keyOf(frame, id));
}
function queueIds(frame) {
  return $("hard-only").checked
    ? frame.person_ids.filter((id) => isHardTarget(frame.output_frame, id))
    : frame.person_ids;
}
function queuePairs() {
  return state.frames.flatMap((frame) => queueIds(frame).map((id) => ({frame: frame.output_frame, person: id})));
}
function frameComplete(frame) {
  return queueIds(frame).every((id) => annotation(frame.output_frame, id).label);
}

function applyHardQueue(batch) {
  const active = batch?.status === "active" && batch.dataset_id === state.config?.dataset_id;
  state.hardBatch = active ? batch.batch : null;
  state.hardTargets = new Set(active ? batch.targets.map(([f, p]) => keyOf(f, p)) : []);
  if (!state.hardTargets.size) $("hard-only").checked = false;
  $("hard-only").disabled = state.busy || !state.hardTargets.size;
}

async function refreshQueue() {
  const result = await api(`/api/queue?step=${state.step}`);
  applyHardQueue(result.hard_queue);
  state.frames = result.frames;
  state.annotations.clear();
  mergeAnnotations(result.annotations);
  renderQueue();
  renderProgress();
}
async function decodeImage(frameId) {
  const image = new Image();
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("视频解码超时，请重新连接后重试。")), 45000);
    image.onload = () => { clearTimeout(timeout); resolve(); };
    image.onerror = () => { clearTimeout(timeout); reject(new Error("无法读取原视频画面，请检查服务终端。")); };
    image.src = `/api/image?id=${frameId}`;
  });
  return image;
}
async function loadTarget(frameId, personId = null) {
  if (personId === null) {
    const queued = state.frames.find((f) => f.output_frame === frameId);
    const ids = queued ? queueIds(queued) : [];
    personId = ids.find((id) => !annotation(frameId, id).label) ?? ids[0] ?? null;
  }
  if (state.frame?.output_frame === frameId && state.image) {
    state.personIndex = Math.max(0, state.frame.people.findIndex((p) => p.person_id === personId));
    render();
    return;
  }
  $("image-loading").hidden = false;
  try {
    const [frame, image] = await Promise.all([api(`/api/frame?id=${frameId}`), decodeImage(frameId)]);
    state.frame = frame;
    state.image = image;
    mergeAnnotations(frame.annotations);
    state.personIndex = Math.max(0, frame.people.findIndex((p) => p.person_id === personId));
    render();
  } finally { $("image-loading").hidden = true; }
}
function renderQueue() {
  const nav = $("frame-list");
  const scroll = nav.scrollTop;
  nav.replaceChildren();
  const filtered = state.frames.filter((f) => queueIds(f).length &&
    (!$("pending-only").checked || !frameComplete(f)));
  $("frame-count").textContent = `${filtered.length} 帧`;
  let segment = null;
  for (const frame of filtered) {
    if (frame.segment !== segment) {
      const heading = document.createElement("div");
      heading.className = "section-eyebrow";
      heading.textContent = `片段 ${frame.segment}`;
      nav.append(heading);
      segment = frame.segment;
    }
    const ids = queueIds(frame);
    const count = ids.filter((id) => annotation(frame.output_frame, id).label).length;
    const button = document.createElement("button");
    button.className = `frame-row${state.frame?.output_frame === frame.output_frame ? " active" : ""}${count === ids.length ? " complete" : ""}`;
    button.dataset.frame = frame.output_frame;
    button.disabled = state.busy;
    button.setAttribute("aria-current", state.frame?.output_frame === frame.output_frame ? "true" : "false");
    button.setAttribute("aria-label", `输出帧 ${frame.output_frame}，原视频 ${frame.source_time}，已标 ${count}/${ids.length}`);
    const icon = document.createElement("span"); icon.className = "frame-icon"; icon.textContent = count === ids.length ? "✓" : "▧";
    const text = document.createElement("span");
    const time = document.createElement("span"); time.className = "frame-time"; time.textContent = frame.source_time.slice(0, 8);
    const sub = document.createElement("span"); sub.className = "frame-sub"; sub.textContent = `输出帧 ${frame.output_frame}`;
    if ($("hard-only").checked) sub.textContent += " · " + ids.map((id) => "P" + id).join("/");
    text.append(time, sub);
    const completion = document.createElement("span"); completion.className = "frame-completion"; completion.textContent = `${count}/${ids.length}`;
    button.append(icon, text, completion);
    button.addEventListener("click", () => run(async () => {
      await flushDraft();
      const id = ids.find((p) => !annotation(frame.output_frame, p).label) || ids[0];
      await loadTarget(frame.output_frame, id);
    }));
    nav.append(button);
  }
  if (!filtered.length) {
    const empty = document.createElement("p"); empty.className = "empty-queue";
    empty.textContent = "当前队列已全部完成。取消筛选可复核，或缩短抽样间隔添加样本。";
    nav.append(empty);
  }
  nav.scrollTop = scroll;
}
function renderProgress() {
  const pairs = queuePairs();
  const counts = {sitting: 0, standing: 0, uncertain: 0, ignore: 0};
  for (const pair of pairs) {
    const label = annotation(pair.frame, pair.person).label;
    if (label) counts[label]++;
  }
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  $("progress-text").textContent = `${total} / ${pairs.length}`;
  $("progress").max = Math.max(1, pairs.length);
  $("progress").value = total;
  for (const label of Object.keys(counts)) $("count-" + label).textContent = counts[label];
  const globalCount = [...state.annotations.values()].filter((a) => a.label).length;
  $("global-count").textContent = `全局已标 ${globalCount} 人 · 导出包含全部标签`;
  const hardDone = [...state.hardTargets].filter((key) => state.annotations.get(key)?.label).length;
  $("hard-summary").textContent = state.hardTargets.size
    ? "本轮 " + hardDone + "/" + state.hardTargets.size + " 人已标。★ 为目标；请补腿部可见程度。仅用于开发，不作独立测试。"
    : "当前数据集没有匹配的困难样本批次。";
}
function renderPeople() {
  $("person-list").replaceChildren();
  state.frame.people.forEach((p, index) => {
    const button = document.createElement("button");
    const a = annotation(state.frame.output_frame, p.person_id);
    button.className = `person-button${index === state.personIndex ? " active" : ""}`;
    button.setAttribute("aria-label", `选择人物 P${p.person_id}，${names[a.label] || "未标注"}`);
    button.setAttribute("aria-pressed", String(index === state.personIndex));
    button.disabled = state.busy;
    const dot = document.createElement("span"); dot.className = "label-dot " + (a.label || "");
    button.append(dot, document.createTextNode(`P${p.person_id}`));
    if (isHardTarget(state.frame.output_frame, p.person_id)) {
      button.append(document.createTextNode(" ★"));
      button.title = "本轮补标目标；星号不表示任何姿态预测";
    }
    button.addEventListener("click", () => run(async () => { await flushDraft(); await loadTarget(state.frame.output_frame, p.person_id); }));
    $("person-list").append(button);
  });
}
function renderEditor() {
  const p = person();
  if (!p) return;
  const a = annotation();
  const draft = storageGet("draft:" + currentKey());
  $("person-title").textContent = `人物 P${p.person_id}`;
  $("person-status").textContent = names[a.label] || "未标注";
  $("person-status").className = "status-pill" + (a.label ? " labeled" : "");
  $("occlusion").value = draft?.occlusion ?? a.occlusion;
  $("note").value = draft?.note ?? a.note;
  document.querySelectorAll("[data-label]").forEach((el) => el.setAttribute("aria-pressed", String(el.dataset.label === a.label)));
  const pairs = queuePairs();
  const index = pairs.findIndex((entry) => entry.frame === state.frame.output_frame && entry.person === p.person_id);
  $("position-text").textContent = index < 0 ? "上下文人物（不计当前队列）" : `${index + 1} / ${pairs.length}`;
  $("occlusion-hint").textContent = isHardTarget(state.frame.output_frame, p.person_id)
    ? "本轮补标必选（忽略除外）" : "可选，建议先选";
  $("save-status").textContent = draft ? (a.label ? "有待保存的修改" : "备注暂存于浏览器 · 选择姿态后提交") : (a.label ? "✓ 当前标签已保存到本机" : "选择姿态即自动保存");
}
function render() {
  if (!state.frame) return;
  $("frame-title").textContent = `输出帧 ${state.frame.output_frame}`;
  $("frame-info").textContent = `原视频 ${state.frame.source_time} · 原始帧 ${state.frame.source_frame} · 片段 ${state.frame.segment} · ${state.frame.people.length} 个检测框`;
  renderQueue(); renderProgress(); renderPeople(); renderEditor(); draw(); rememberCursor();
}

function drawSkeleton(ctx, target, selected) {
  if (!$("show-skeleton").checked) return;
  const points = target.keypoints;
  const valid = (p) => p && p.confidence >= 0.35 && p.x > 0 && p.y > 0;
  ctx.strokeStyle = selected ? "#e7ff73" : "#e8f3ee99";
  ctx.fillStyle = "#f6fff8";
  ctx.lineWidth = selected ? 2.5 : 1.5;
  for (const [a, b] of state.config.skeleton) {
    if (!valid(points[a]) || !valid(points[b])) continue;
    ctx.beginPath(); ctx.moveTo(points[a].x, points[a].y); ctx.lineTo(points[b].x, points[b].y); ctx.stroke();
  }
  for (const point of points) {
    if (!valid(point)) continue;
    ctx.beginPath(); ctx.arc(point.x, point.y, selected ? 3 : 2, 0, Math.PI * 2); ctx.fill();
  }
}
function draw() {
  if (!state.image || !person()) return;
  const canvas = $("scene"), ctx = canvas.getContext("2d");
  canvas.width = state.image.naturalWidth; canvas.height = state.image.naturalHeight;
  ctx.drawImage(state.image, 0, 0);
  const selected = person();
  const [x1, y1, x2, y2] = selected.bbox_xyxy;
  // A subtle outside veil distinguishes the selected person without altering the source image.
  ctx.fillStyle = "rgba(9,24,29,.22)";
  ctx.beginPath(); ctx.rect(0, 0, canvas.width, canvas.height); ctx.rect(x1, y1, x2 - x1, y2 - y1); ctx.fill("evenodd");
  for (const p of state.frame.people) {
    const active = p.person_id === selected.person_id;
    drawSkeleton(ctx, p, active);
    if (!$("show-boxes").checked) continue;
    const [left, top, right, bottom] = p.bbox_xyxy;
    const a = annotation(state.frame.output_frame, p.person_id);
    const color = active ? "#b8f37f" : (colors[a.label] || "#d9e2e5");
    ctx.lineWidth = active ? 4 : 1.5; ctx.strokeStyle = color;
    ctx.strokeRect(left, top, right - left, bottom - top);
    const text = `P${p.person_id}${a.label ? " · " + names[a.label] : ""}`;
    ctx.font = "bold 16px 'Microsoft YaHei',sans-serif";
    const width = ctx.measureText(text).width + 16;
    const labelX = Math.max(0, Math.min(canvas.width - width, left));
    const labelY = Math.max(0, top - 28);
    ctx.fillStyle = color; ctx.fillRect(labelX, labelY, width, 26);
    ctx.fillStyle = "#18352a"; ctx.fillText(text, labelX + 8, labelY + 18);
  }
  drawCrop();
}
function drawCrop() {
  if (!state.image || !person()) return;
  const canvas = $("crop"), ctx = canvas.getContext("2d");
  const bounds = canvas.parentElement.getBoundingClientRect();
  const width = Math.max(1, Math.round(bounds.width * 2));
  const height = Math.max(1, Math.round(bounds.height * 2));
  canvas.width = width; canvas.height = height;
  ctx.fillStyle = "#e7edeb"; ctx.fillRect(0, 0, width, height);
  const [x1, y1, x2, y2] = person().bbox_xyxy;
  const centerX = (x1 + x2) / 2, centerY = (y1 + y2) / 2;
  const zoom = Number($("zoom").value);
  const scale = Math.min(width / Math.max(1, (x2 - x1) * 1.55), height / Math.max(1, (y2 - y1) * 1.3)) * zoom;
  ctx.save();
  ctx.translate(width / 2 - centerX * scale, height / 2 - centerY * scale); ctx.scale(scale, scale);
  ctx.drawImage(state.image, 0, 0);
  drawSkeleton(ctx, person(), true);
  if ($("show-boxes").checked) {ctx.strokeStyle = "#b8f37f"; ctx.lineWidth = 2 / scale; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);}
  ctx.restore();
  $("zoom-value").textContent = zoom.toFixed(1) + "×";
}

function stashDraft() {
  if (!currentKey()) return;
  const draft = {occlusion: $("occlusion").value, note: $("note").value};
  const a = annotation();
  if (draft.occlusion === a.occlusion && draft.note === a.note) storageRemove("draft:" + currentKey());
  else storageSet("draft:" + currentKey(), draft);
  $("save-status").textContent = a.label ? "修改待保存…" : "备注暂存于浏览器 · 选择姿态后提交";
  clearTimeout(state.draftTimer);
  const key = currentKey();
  if (a.label) state.draftTimer = setTimeout(() => {
    if (key === currentKey() && !state.busy) run(async () => { await flushDraft(); });
  }, 1200);
}
async function save(label) {
  const p = person();
  if (!p) return;
  if (label && label !== "ignore" && isHardTarget(state.frame.output_frame, p.person_id)
      && $("occlusion").value === "unmarked") {
    throw new Error("请先选择腿部可见程度，再保存本轮困难样本；误检/重复框可直接选忽略。");
  }
  const a = annotation();
  const key = currentKey();
  $("save-status").textContent = "正在写入本机数据库…";
  const data = await api("/api/annotation", {output_frame: state.frame.output_frame,
    person_id: p.person_id, label, occlusion: $("occlusion").value, note: $("note").value,
    expected_revision: a.revision});
  state.annotations.set(key, data.annotation);
  storageRemove("draft:" + key);
  state.undo = data.event_id;
  storageSet("undo", data.event_id);
  $("error-banner").hidden = true;
  render();
}
async function flushDraft() {
  clearTimeout(state.draftTimer);
  if (!person()) return;
  const a = annotation();
  if (a.label && (a.occlusion !== $("occlusion").value || a.note !== $("note").value)) await save(a.label);
}
async function labelCurrent(label) {
  if (!person()) return;
  await save(label);
  if ($("auto-next").checked) await nextPending();
}
async function nextPending() {
  const pairs = queuePairs();
  const index = pairs.findIndex((p) => p.frame === state.frame?.output_frame && p.person === person()?.person_id);
  for (let offset = 1; offset <= pairs.length; offset++) {
    const target = pairs[(index + offset) % pairs.length];
    if (!annotation(target.frame, target.person).label) {
      await loadTarget(target.frame, target.person); return;
    }
  }
  toast("当前队列全部标完！可以导出，或缩短抽样间隔继续补充。");
}
async function move(delta) {
  await flushDraft();
  const pairs = queuePairs();
  const index = pairs.findIndex((p) => p.frame === state.frame?.output_frame && p.person === person()?.person_id);
  const target = pairs[index + delta];
  if (target) await loadTarget(target.frame, target.person);
  else toast(delta < 0 ? "已经是第一个人物" : "已经是最后一个人物");
}
async function undo() {
  if (!state.undo) return;
  const result = await api("/api/undo", {event_id: state.undo});
  state.undo = null; storageRemove("undo");
  const a = result.annotation;
  storageRemove("draft:" + keyOf(a.output_frame, a.person_id));
  await refreshQueue();
  // Undoing a manually selected frame must keep the frame accessible.
  await loadTarget(a.output_frame, a.person_id);
  toast("已撤销上一次保存；原标签已恢复。");
}
async function jump(value, mode) {
  await flushDraft();
  const result = await api("/api/jump", {value, mode});
  await refreshQueue();
  if ($("hard-only").checked && !state.frames.some((f) => f.output_frame === result.output_frame && queueIds(f).length)) {
    $("hard-only").checked = false; storageSet("hardOnly", false);
  }
  await loadTarget(result.output_frame);
  const list = $("frame-list"), active = list.querySelector(".active");
  if (active) list.scrollTop += active.getBoundingClientRect().top - list.getBoundingClientRect().top - 10;
}
function showHelp() { $("help-dialog").showModal(); }

document.querySelectorAll("[data-label]").forEach((button) => button.addEventListener("click", () => run(() => labelCurrent(button.dataset.label))));
$("prev-button").addEventListener("click", () => run(() => move(-1)));
$("next-button").addEventListener("click", () => run(() => move(1)));
$("next-pending").addEventListener("click", () => run(async () => { await flushDraft(); await nextPending(); }));
$("undo-button").addEventListener("click", () => run(undo));
$("clear-button").addEventListener("click", () => run(async () => {
  if (annotation().label && confirm("清除当前人物的人工标签和备注？可以用「撤销上一步」恢复。")) await save(null);
}));
$("save-next").addEventListener("click", () => run(async () => {
  if (!annotation().label) {toast("请先选坐着、站着、不确定或忽略。"); return;}
  await flushDraft(); await nextPending();
}));
$("note").addEventListener("input", stashDraft);
$("occlusion").addEventListener("change", () => {stashDraft(); $("occlusion").blur();});
$("auto-next").addEventListener("change", () => storageSet("autoNext", $("auto-next").checked));
$("pending-only").addEventListener("change", () => {storageSet("pendingOnly", $("pending-only").checked); renderQueue();});
$("hard-only").addEventListener("change", () => run(async () => {
  const enabled = $("hard-only").checked;
  try { await flushDraft(); }
  catch (error) { $("hard-only").checked = !enabled; throw error; }
  storageSet("hardOnly", enabled);
  const pairs = queuePairs();
  const target = pairs.find((p) => !annotation(p.frame, p.person).label) || pairs[0];
  if (target) await loadTarget(target.frame, target.person);
  else { renderQueue(); renderProgress(); toast("当前困难样本队列为空。"); }
}));
$("step").addEventListener("change", () => run(async () => {
  await flushDraft();
  state.step = Number($("step").value); storageSet("step", state.step);
  await refreshQueue();
  if (!state.frames.some((f) => f.output_frame === state.frame?.output_frame && queueIds(f).length)) {
    const first = state.frames.find((f) => queueIds(f).length && !frameComplete(f))
      || state.frames.find((f) => queueIds(f).length);
    if (first) await loadTarget(first.output_frame);
  } else render();
}));
$("jump-form").addEventListener("submit", (event) => {
  event.preventDefault(); const value = $("jump-value").value;
  run(() => jump(value, $("jump-mode").value));
});
$("jump-mode").addEventListener("change", () => {$("jump-value").placeholder = $("jump-mode").value === "frame" ? "如 2400" : "如 06:20";});
$("show-skeleton").addEventListener("change", draw);
$("show-boxes").addEventListener("change", draw);
$("zoom").addEventListener("input", drawCrop);
window.addEventListener("resize", drawCrop);
$("fullscreen-button").addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else $("stage").requestFullscreen().catch(() => toast("当前浏览器不支持全屏，可用浏览器缩放查看。"));
});
$("scene").addEventListener("click", (event) => {
  if (state.busy || !state.image) return;
  const canvas = $("scene"), rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
  const x = (event.clientX - rect.left - (rect.width - canvas.width * scale) / 2) / scale;
  const y = (event.clientY - rect.top - (rect.height - canvas.height * scale) / 2) / scale;
  const hits = state.frame.people.filter((p) => x >= p.bbox_xyxy[0] && x <= p.bbox_xyxy[2] && y >= p.bbox_xyxy[1] && y <= p.bbox_xyxy[3]);
  if (!hits.length) return;
  // Repeated clicks cycle overlapping detections (including duplicate boxes).
  const current = hits.findIndex((p) => p.person_id === person().person_id);
  const target = hits[(current + 1) % hits.length];
  run(async () => {await flushDraft(); await loadTarget(state.frame.output_frame, target.person_id);});
});
$("help-button").addEventListener("click", showHelp);
$("close-help").addEventListener("click", () => $("help-dialog").close());
$("got-it").addEventListener("click", () => $("help-dialog").close());
$("dismiss-error").addEventListener("click", () => {$("error-banner").hidden = true;});
$("retry-button").addEventListener("click", () => run(async () => {
  state.config = await api("/api/config");
  await refreshQueue();
  if (state.frame) {
    const id = state.frame.output_frame, pid = person()?.person_id;
    state.image = null;
    await loadTarget(id, pid);
  } else if (state.frames.length) await loadTarget(state.frames[0].output_frame);
  $("error-banner").hidden = true;
  toast("已重新连接，标签已从数据库刷新。");
}));
document.querySelectorAll(".export").forEach((link) => link.addEventListener("click", async (event) => {
  event.preventDefault();
  if (state.busy) {toast("请等待当前操作保存完成。"); return;}
  await run(async () => {
    await flushDraft();
    const response = await fetch(link.href);
    if (!response.ok) throw new Error("导出失败，请检查本地服务后重试。");
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a"); anchor.href = url;
    const kind = new URL(link.href).searchParams.get("format");
    anchor.download = `posture_labels_${state.config.dataset_id.slice(0, 8)}.${kind}`;
    document.body.append(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    toast("已导出所有已标样本。未选择姿态的备注草稿不包含在内。");
  });
}));
document.addEventListener("keydown", (event) => {
  if (event.repeat || event.altKey || $("help-dialog").open) return;
  const focus = document.activeElement;
  if (["TEXTAREA", "SELECT"].includes(focus?.tagName) ||
      (focus?.tagName === "INPUT" && !["checkbox", "radio"].includes(focus.type)) || focus?.isContentEditable) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {event.preventDefault(); run(undo); return;}
  if (event.ctrlKey || event.metaKey || state.busy) return;
  const labels = {"1": "sitting", "2": "standing", "3": "uncertain", "4": "ignore"};
  if (labels[event.key]) {event.preventDefault(); run(() => labelCurrent(labels[event.key]));}
  else if (event.key === "ArrowLeft") {event.preventDefault(); run(() => move(-1));}
  else if (event.key === "ArrowRight") {event.preventDefault(); run(() => move(1));}
  else if (event.key.toLowerCase() === "n") {event.preventDefault(); run(async () => {await flushDraft(); await nextPending();});}
  else if (event.key.toLowerCase() === "k") {event.preventDefault(); $("show-skeleton").checked = !$("show-skeleton").checked; draw();}
  else if (event.key.toLowerCase() === "b") {event.preventDefault(); $("show-boxes").checked = !$("show-boxes").checked; draw();}
  else if (event.key === "?") showHelp();
  else if (event.key === "Enter") {event.preventDefault(); $("save-next").click();}
});
window.addEventListener("beforeunload", (event) => {
  if (state.busy) {event.preventDefault(); event.returnValue = "";}
});

async function start() {
  await run(async () => {
    state.config = await api("/api/config");
    // refreshQueue loads the active queue and its target frames atomically from the server.
    $("hard-only").checked = location.hash === "#hard" || storageGet("hardOnly", false);
    $("video-name").textContent = state.config.video;
    $("database-path").textContent = `标注数据库：${state.config.database}`;
    const step = Number(storageGet("step", 3));
    state.step = [1, 3, 5, 10, 30].includes(step) ? step : 3;
    $("step").value = String(state.step);
    $("auto-next").checked = storageGet("autoNext", true);
    $("pending-only").checked = storageGet("pendingOnly", false);
    state.undo = storageGet("undo");
    await refreshQueue();
    const saved = storageGet("cursor");
    const target = location.hash === "#hard" ? null : state.frames.find((f) =>
      f.output_frame === saved?.frame && queueIds(f).includes(saved.person));
    const initial = target || state.frames.find((f) => queueIds(f).length && !frameComplete(f))
      || state.frames.find((f) => queueIds(f).length);
    if (initial) {
      const ids = queueIds(initial);
      const id = target ? saved.person : (ids.find((p) => !annotation(initial.output_frame, p).label) || ids[0]);
      await loadTarget(initial.output_frame, id);
    } else showError("未找到可标注人物，请核对关键点文件。");
  });
}
start();

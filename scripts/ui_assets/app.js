"use strict";

const el = id => document.getElementById(id);
const form = el("task-form");
const defaultDepthModel = "Distill-Any-Depth";
const active = new Set(["queued", "running"]);
const resumable = new Set(["failed", "cancelled", "interrupted"]);
const modeNames = {image: "单图处理", batch: "批处理", demo: "CPU 自检"};
const actionNames = {run: "运行", preview: "预览计划", check: "检查环境"};
const stateNames = {
  queued: "等待启动", running: "运行中", success: "已完成", failed: "失败",
  cancelled: "已取消", interrupted: "已中断", reused: "已复用", skipped: "已跳过",
};
const stageNames = [
  "", "切分输入影像", "准备足迹推理数据", "提取建筑足迹与偏移", "导出建筑检测表",
  "生成建筑足迹图层", "提取 ATL03 光子", "生成稀疏控制高度栅格", "生成基础模型预测",
  "校准并融合高度栅格", "生成逐栋建筑高度",
];
const stageDetails = {
  "Select ATL03 files": "选择 ATL03 文件",
  "Validate shared photon table": "检查共享光子表",
  "Check usable photon controls": "检查可用光子控制点",
  "Extract ICESat-2 ground/building photons (shared table)": "复用共享光子表",
};
const tileStageNames = {
  atl03_download: "准备 ATL03 光子数据", atl03_no_hdf5: "检查 ATL03 文件",
  full_pipeline: "处理瓦片", disk_preflight: "检查磁盘空间", stage_validation: "校验阶段成果",
  stable_result_copy: "保存瓦片成果", retained_result_validation: "校验瓦片成果",
  raw_patch_archive: "保存模型原始窗口", done: "完成",
};
const artifactNames = {
  "building_height_raster.tif": "建筑高度栅格",
  "per_building_heights.geojson": "逐栋建筑高度",
  "output.tif": "建筑高度栅格",
  "footprint_direct_nms_absolute_height.geojson": "逐栋建筑高度",
  "building_heights.geojson": "逐栋建筑高度",
  "city_absolute_height_1m_uint16.tif": "建筑高度栅格（瓦片拼接）",
  "city_building_heights.geojson": "逐栋建筑高度（瓦片拼接）",
  "city_global_patch_height_1m_uint16.tif": "建筑高度栅格（联合优化）",
  "city_global_patch_building_heights.geojson": "逐栋建筑高度（联合优化）",
  "city_synrs3d_raw_1m_uint16.tif": "模型原始高度栅格",
  "city_synrs3d_raw_building_heights.geojson": "模型原始逐栋高度",
  "batch_status.csv": "批处理汇总",
  "global_patch_diagnostics.json": "跨瓦片联合优化诊断",
};
let mode = "image";
let lastRealMode = "image";
let resumeTarget;
let earthdataSaved = false;
let jobId = localStorage.getItem("hab-job");
let selected;
let view;
let submitting = false;
let pickerTarget;
let pickerKind;
let pickerPath;
let displayList = "";
let previewKey;
let previewRequest = 0;
let previewMode = "map";
let lod1Viewer;

function artifactLabel(name) {
  if (Object.hasOwn(artifactNames, name)) return artifactNames[name];
  const tile = name.match(/^(.+?)_(absolute_height\.tif|building_heights\.geojson)$/);
  return tile ? tile[1] + " · " + (tile[2] === "absolute_height.tif" ? "建筑高度栅格" : "逐栋建筑高度") : name;
}

function stageLabel(stage) {
  if (stage.tile) {
    const step = stage.stage?.match(/^\[(\d+)\/\d+\]/)?.[1] || stage.stage?.match(/^stage[_-]?(\d+)$/)?.[1];
    const status = tileStageNames[stage.stage] || stageNames[step];
    return stage.tile + (status || stage.stage ? " · " + (status || stage.stage) : "");
  }
  const title = stageDetails[stage.title] || stageNames[stage.step] || stage.title;
  return stage.step + ". " + title + (stage.elapsed_seconds == null ? "" : " · " + stage.elapsed_seconds + " 秒");
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "本机服务返回错误。");
  return result;
}

function message(id, text) {
  el(id).textContent = text;
  el(id).hidden = !text;
}

function visibility(section, visible) {
  section.hidden = !visible;
  for (const field of section.querySelectorAll("input, select")) field.disabled = !visible;
}

function updateAtl03() {
  const local = el("atl03-source").value === "local";
  visibility(el("online-fields"), mode !== "demo" && !local);
  visibility(el("local-fields"), mode !== "demo" && local);
  el("atl03-dir").required = mode !== "demo" && local;
  if (mode === "demo" || local) el("earthdata-password").value = "";
  updateEarthdataFields();
}

function updateEarthdataFields() {
  const online = mode !== "demo" && el("atl03-source").value === "online";
  visibility(el("earthdata-credentials"), online && !earthdataSaved);
  el("earthdata-note").hidden = !online || earthdataSaved;
  el("earthdata-saved").hidden = !online || !earthdataSaved;
}

function setEarthdataSaved(saved) {
  earthdataSaved = saved;
  if (saved) {
    el("earthdata-username").value = "";
    el("earthdata-password").value = "";
  }
  updateEarthdataFields();
}

async function refreshEarthdata() {
  setEarthdataSaved((await api("/api/earthdata")).saved);
}

function setSidebar(name) {
  for (const button of document.querySelectorAll("[data-sidebar]")) {
    const activePanel = button.dataset.sidebar === name;
    button.setAttribute("aria-pressed", String(activePanel));
    el(button.dataset.sidebar).hidden = !activePanel;
  }
}

function changeMode(next) {
  if (next !== "demo" && next !== lastRealMode) {
    el("input").value = "";
    el("model").value = defaultDepthModel;
  }
  if (next !== "demo") lastRealMode = next;
  mode = next;
  setSidebar("compose");
  const batch = mode === "batch";
  for (const button of document.querySelectorAll(".mode")) {
    button.classList.toggle("active", button.dataset.mode === mode);
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  }
  visibility(el("real-fields"), mode !== "demo");
  el("compose-title").textContent = mode === "demo" ? "CPU 自检" : "新建任务";
  el("demo-fields").hidden = mode !== "demo";
  el("processing-modes").hidden = mode === "demo";
  el("back-to-setup").hidden = mode !== "demo";
  visibility(el("advanced"), mode !== "demo");
  visibility(el("image-options"), mode === "image");
  visibility(el("batch-options"), batch);
  el("input-label").textContent = batch ? "输入瓦片目录" : "输入 GeoTIFF";
  el("input").placeholder = batch ? "/path/to/tiles" : "/path/to/image.tif";
  el("input-note").textContent = batch
    ? "选择直接包含已切分 GeoTIFF 的目录。"
    : "支持带坐标系的三波段 UInt8 RGB 影像。";
  for (const button of document.querySelectorAll(".optional-action")) button.hidden = mode === "demo";
  el("start-button").textContent = mode === "demo" ? "运行自检" : "开始运行";
  updateAtl03();
  message("form-error", "");
}

function makeNode(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function setState(node, state) {
  node.dataset.state = state;
  node.textContent = stateNames[state] || state;
}

function nameFor(job) {
  if (job.mode === "demo") return modeNames.demo;
  const flag = job.mode === "image" ? "--image" : "--input-dir";
  const path = job.argv[job.argv.indexOf(flag) + 1];
  return path.split("/").pop();
}

function formatTime(value) {
  return new Date(value).toLocaleString("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

function renderList(jobs) {
  const signature = JSON.stringify([jobId, jobs.map(job => [job.id, job.status, job.attempt])]);
  if (signature === displayList) return;
  displayList = signature;
  el("job-count").textContent = jobs.length;
  el("history-empty").hidden = jobs.length > 0;
  el("job-list").replaceChildren(...jobs.map(job => {
    const row = makeNode("button", undefined, "job-row");
    row.type = "button";
    row.classList.toggle("selected", job.id === jobId);
    row.setAttribute("aria-pressed", String(job.id === jobId));
    const label = makeNode("span");
    const info = job.mode === "demo" ? formatTime(job.created_at)
      : modeNames[job.mode] + " · " + actionNames[job.action] + " · " + formatTime(job.created_at);
    label.append(makeNode("span", nameFor(job), "job-title"), makeNode("span", info, "muted"));
    const badge = makeNode("span", undefined, "state");
    setState(badge, job.status);
    row.append(label, badge);
    row.addEventListener("click", () => {
      message("job-error", "");
      selectJob(job.id);
    });
    return row;
  }));
}

function setView(next) {
  view = next;
  for (const button of document.querySelectorAll(".tab-bar button")) {
    const chosen = button.dataset.view === view;
    button.classList.toggle("active", chosen);
    button.setAttribute("aria-pressed", String(chosen));
    el(button.dataset.view).hidden = !chosen;
  }
  if (view === "results" && selected?.id === jobId) updatePreview(selected);
}

function defaultView(job) {
  if (job.action !== "run") return "log";
  return job.artifacts.length && job.status === "success" ? "results" : "progress";
}

function renderDetails(job) {
  const justFinished = selected?.id === job.id && active.has(selected.status) && job.status === "success";
  selected = job;
  el("workspace-empty").hidden = true;
  el("detail").hidden = false;
  if (!view) setView(defaultView(job));
  el("detail-title").textContent = nameFor(job);
  el("detail-time").textContent = modeNames[job.mode] + " · " + actionNames[job.action] + " · 第 " + job.attempt + " 次";
  setState(el("detail-status"), job.status);
  el("cancel").hidden = !active.has(job.status);
  el("resume").hidden = job.mode === "demo" || job.action !== "run" || !resumable.has(job.status);
  el("command").textContent = job.command;
  el("output-path").textContent = job.output;
  const log = el("log-text");
  const follow = log.scrollHeight - log.scrollTop - log.clientHeight < 30;
  if (log.textContent !== job.log) log.textContent = job.log;
  if (follow) log.scrollTop = log.scrollHeight;

  let progressText;
  if (job.status === "failed") {
    const failed = job.stages.find(stage => stage.status === "failed");
    const code = job.exit_code === null ? "" : "，退出码 " + job.exit_code;
    const loginError = job.requires_earthdata && /earthdata (authentication|login|credentials)|(?:401|403) (?:unauthorized|forbidden)/i.test(job.log);
    progressText = loginError ? "Earthdata 认证失败。可在「任务配置」清除凭据，核对账号密码和应用授权后再续跑。"
      : (failed ? "第 " + failed.step + " 阶段失败" : "任务执行失败") + code + "。详情见日志。";
  } else if (job.tiles.length) {
    const complete = job.tiles.filter(tile => tile.status === "success" || tile.status === "skipped").length;
    progressText = "已完成 " + complete + " 块，已上报 " + job.tiles.length + " 块瓦片";
  } else if (job.stages.length) {
    progressText = "阶段记录";
  } else if (active.has(job.status)) {
    progressText = "正在等待阶段信息。";
  } else {
    progressText = "当前状态为" + stateNames[job.status] + "。";
  }
  el("progress-note").textContent = progressText;
  const stages = job.tiles.length ? job.tiles : job.stages;
  el("stage-list").replaceChildren(...stages.map(stage => {
    const row = makeNode("li");
    const badge = makeNode("span", undefined, "state");
    setState(badge, stage.status);
    row.append(makeNode("span", stageLabel(stage)), badge);
    return row;
  }));
  const retainedOnly = job.status === "success" &&
    job.artifacts.some(item => item.name === "global_patch_diagnostics.json") &&
    !job.artifacts.some(item => item.name === "city_global_patch_height_1m_uint16.tif");
  el("result-note").textContent = retainedOnly
    ? "联合优化未通过检查，已保留瓦片成果。具体原因见诊断。"
    : job.artifacts.length ? "可下载的成果" : "当前没有可下载的成果。";
  el("downloads").replaceChildren(...job.artifacts.map(item => {
    const row = makeNode("li");
    const link = makeNode("a", artifactLabel(item.name));
    link.href = "/api/jobs/" + job.id + "/download?" + new URLSearchParams({artifact: item.id});
    const size = item.bytes >= 1048576
      ? (item.bytes / 1048576).toFixed(2) + " MB"
      : (item.bytes / 1024).toFixed(1) + " KB";
    const file = makeNode("span", undefined, "download-file");
    file.append(link);
    if (link.textContent !== item.name) file.append(makeNode("small", item.name, "muted download-filename"));
    row.append(file, makeNode("span", size, "muted"));
    return row;
  }));
  if (justFinished && job.artifacts.length && view === "progress") setView("results");
  else if (view === "results") updatePreview(job);
}

function updatePreview(job) {
  const rasters = job.artifacts.filter(item => /\.tiff?$/i.test(item.name));
  const key = JSON.stringify([job.id, job.attempt, rasters.map(item => [item.id, item.bytes])]);
  if (key === previewKey) return;
  previewKey = key;
  el("preview-panel").hidden = !rasters.length;
  el("preview-raster").replaceChildren(...rasters.map(item => {
    const option = makeNode("option", artifactLabel(item.name));
    option.value = item.id;
    return option;
  }));
  el("preview-outline").checked = true;
  loadResultPreview();
}

async function loadResultPreview() {
  const request = ++previewRequest;
  const artifact = el("preview-raster").value;
  el("preview-map").hidden = true;
  el("preview-lod1").hidden = true;
  if (!artifact) return;
  const lod1 = previewMode === "lod1";
  message("preview-status", lod1 ? "正在生成 LOD1 模型…" : "正在生成预览…");
  try {
    const endpoint = lod1 ? "lod1" : "preview";
    const result = await api("/api/jobs/" + jobId + "/" + endpoint + "?" + new URLSearchParams({artifact}));
    if (request !== previewRequest) return;
    if (lod1) {
      showLod1(result);
      return;
    }
    el("preview-height").src = result.height_png;
    if (result.footprint_png) el("preview-footprints").src = result.footprint_png;
    else el("preview-footprints").removeAttribute("src");
    el("preview-outline-label").hidden = !result.footprint_png;
    el("preview-footprints").hidden = !result.footprint_png || !el("preview-outline").checked;
    const count = result.buildings == null ? "" : " · " + result.buildings + " 栋建筑";
    el("preview-meta").textContent = result.width + " × " + result.height + " 像素" + count;
    el("height-legend").hidden = !result.has_heights;
    if (result.has_heights) {
      const canvas = el("height-colors");
      const context = canvas.getContext("2d");
      const gradient = context.createLinearGradient(0, 0, canvas.width, 0);
      result.palette.forEach((color, index) => gradient.addColorStop(index / (result.palette.length - 1), color));
      context.fillStyle = gradient;
      context.fillRect(0, 0, canvas.width, canvas.height);
      el("height-max").textContent = new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 1})
        .format(result.height_max) + " 米";
    }
    message("preview-status", result.has_heights ? "" : "该栅格没有有效高度像素。");
    el("preview-map").hidden = false;
    mapViewer.show(JSON.stringify([previewKey, artifact]));
  } catch (error) {
    if (request === previewRequest) message("preview-status", error.message);
  }
}

function createMapViewer(viewport, scene, raster) {
  const maxScale = 12;
  const zoomStep = 1.25;
  const pointers = new Map();
  let scale = 1;
  let x = 0;
  let y = 0;
  let currentImage;

  function render() {
    const {clientWidth: width, clientHeight: height} = viewport;
    if (!width || !height || !raster.naturalWidth) return;
    const fit = Math.min(width / raster.naturalWidth, height / raster.naturalHeight);
    const maxX = Math.max(0, (raster.naturalWidth * fit * scale - width) / 2);
    const maxY = Math.max(0, (raster.naturalHeight * fit * scale - height) / 2);
    x = Math.min(maxX, Math.max(-maxX, x));
    y = Math.min(maxY, Math.max(-maxY, y));
    scene.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    el("map-zoom-level").value = Math.round(scale * 100) + "%";
    el("map-zoom-out").disabled = scale === 1;
    el("map-zoom-in").disabled = scale === maxScale;
  }

  function zoom(factor, from = {x: 0, y: 0}, to = from) {
    const next = Math.min(maxScale, Math.max(1, scale * factor));
    const ratio = next / scale;
    x = to.x - (from.x - x) * ratio;
    y = to.y - (from.y - y) * ratio;
    scale = next;
    render();
  }

  function reset() {
    x = y = 0;
    scale = 1;
    render();
  }

  function point(event) {
    const rect = viewport.getBoundingClientRect();
    return {x: event.clientX - rect.left - rect.width / 2, y: event.clientY - rect.top - rect.height / 2};
  }

  function gesture() {
    const [first, second] = pointers.values();
    if (!second) return {...first, distance: 0};
    return {x: (first.x + second.x) / 2, y: (first.y + second.y) / 2,
            distance: Math.hypot(first.x - second.x, first.y - second.y)};
  }

  viewport.addEventListener("wheel", event => {
    event.preventDefault();
    // Normalize mouse wheel steps and keep the point under the cursor fixed.
    const delta = event.deltaY * (event.deltaMode === 0 ? 1 : 30);
    zoom(Math.exp(-Math.min(250, Math.max(-250, delta)) * .003), point(event));
  }, {passive: false});
  viewport.addEventListener("pointerdown", event => {
    if (event.button !== 0 || pointers.size >= 2) return;
    pointers.set(event.pointerId, point(event));
    viewport.setPointerCapture(event.pointerId);
    viewport.dataset.dragging = "true";
  });
  viewport.addEventListener("pointermove", event => {
    if (!pointers.has(event.pointerId)) return;
    const previous = gesture();
    pointers.set(event.pointerId, point(event));
    const next = gesture();
    zoom(previous.distance > 0 ? next.distance / previous.distance : 1, previous, next);
  });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
    viewport.addEventListener(name, event => {
      pointers.delete(event.pointerId);
      viewport.dataset.dragging = String(pointers.size > 0);
    });
  }
  viewport.addEventListener("keydown", event => {
    const movement = {ArrowLeft: [48, 0], ArrowRight: [-48, 0], ArrowUp: [0, 48], ArrowDown: [0, -48]};
    if (event.key in movement) {
      const [dx, dy] = movement[event.key];
      zoom(1, {x: 0, y: 0}, {x: dx, y: dy});
    } else if (event.key === "+" || event.key === "=") zoom(zoomStep);
    else if (event.key === "-") zoom(1 / zoomStep);
    else if (event.key === "Home") reset();
    else return;
    event.preventDefault();
  });
  el("map-zoom-in").addEventListener("click", () => zoom(zoomStep));
  el("map-zoom-out").addEventListener("click", () => zoom(1 / zoomStep));
  el("map-reset").addEventListener("click", reset);
  raster.addEventListener("load", render);
  new ResizeObserver(render).observe(viewport);

  return {show(key) {
    if (key !== currentImage) reset();
    currentImage = key;
    render();
  }};
}

const mapViewer = createMapViewer(el("map-viewport"), el("map-scene"), el("preview-height"));

function showLod1(result) {
  if (!result.buildings) {
    message("preview-status", "此成果没有带有效正高度的建筑物。");
    return;
  }
  if (!lod1Viewer) lod1Viewer = createLod1Viewer(el("lod1-canvas"));
  lod1Viewer.load(result);
  const maximum = new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 1}).format(result.height_max);
  const skipped = result.skipped ? " · 跳过 " + result.skipped + " 栋" : "";
  el("lod1-meta").textContent = result.buildings + " 栋建筑 · 最高 " + maximum + " 米" + skipped;
  message("preview-status", "");
  el("preview-lod1").hidden = false;
}

for (const button of document.querySelectorAll("[data-preview-mode]")) {
  button.addEventListener("click", () => {
    if (previewMode === button.dataset.previewMode) return;
    previewMode = button.dataset.previewMode;
    for (const option of document.querySelectorAll("[data-preview-mode]")) {
      option.setAttribute("aria-pressed", String(option.dataset.previewMode === previewMode));
    }
    loadResultPreview();
  });
}
el("lod1-zoom-in").addEventListener("click", () => lod1Viewer.zoomBy(1.22));
el("lod1-zoom-out").addEventListener("click", () => lod1Viewer.zoomBy(1 / 1.22));
el("lod1-reset").addEventListener("click", () => lod1Viewer.reset());
el("expand-preview").addEventListener("click", () => {
  el("preview-dialog-body").append(el("preview-content"));
  el("preview-dialog").showModal();
});
el("close-preview").addEventListener("click", () => el("preview-dialog").close());
el("preview-dialog").addEventListener("close", () => el("preview-panel").append(el("preview-content")));

el("preview-raster").addEventListener("change", loadResultPreview);
el("preview-outline").addEventListener("change", () => {
  el("preview-footprints").hidden = !el("preview-outline").checked;
});

function selectJob(id, nextView) {
  if (jobId !== id) {
    previewKey = undefined;
    previewRequest++;
    el("preview-panel").hidden = true;
  }
  jobId = id;
  localStorage.setItem("hab-job", id);
  setView(nextView);
  update().then(() => {
    if (jobId === id && !el("detail").hidden && window.matchMedia("(max-width: 835px)").matches) {
      el("detail").scrollIntoView({block: "start"});
    }
  });
}

async function update() {
  try {
    const jobs = await api("/api/jobs");
    el("connection").textContent = "本机服务已连接";
    el("connection").dataset.online = "true";
    if (!jobs.length) {
      el("detail").hidden = true;
      el("workspace-empty").hidden = false;
      renderList(jobs);
      return;
    }
    if (!jobs.some(job => job.id === jobId)) {
      jobId = jobs[0].id;
      view = undefined;
    }
    renderList(jobs);
    const requested = jobId;
    const job = await api("/api/jobs/" + requested);
    if (requested === jobId) renderDetails(job);
  } catch (error) {
    el("connection").textContent = "本机服务连接失败";
    el("connection").dataset.online = "false";
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (submitting) return;
  const action = event.submitter?.value || "run";
  const values = Object.fromEntries(new FormData(form));
  for (const key of ["input", "atl03_dir", "gpus", "target_crs"]) {
    if (key in values) values[key] = values[key].trim();
  }
  Object.assign(values, {mode, action, global_optimize: mode === "batch" && el("global-optimize").checked});
  const onlineRun = mode !== "demo" && action === "run" && values.atl03_source === "online";
  submitting = true;
  message("form-error", "");
  for (const button of form.querySelectorAll('[type="submit"]')) button.disabled = true;
  try {
    if (onlineRun) {
      await refreshEarthdata();
      if (!earthdataSaved && (!values.earthdata_username?.trim() || !values.earthdata_password?.trim())) {
        message("form-error", "通过 Earthdata 下载时，请填写登录名和密码。");
        el(values.earthdata_username?.trim() ? "earthdata-password" : "earthdata-username").focus();
        return;
      }
    }
    if (!onlineRun || earthdataSaved) {
      delete values.earthdata_username;
      delete values.earthdata_password;
    }
    const job = await api("/api/jobs", values);
    setEarthdataSaved(job.earthdata_saved);
    if (action === "run") setSidebar("history");
    selectJob(job.id);
  } catch (error) {
    message("form-error", error.message);
  } finally {
    submitting = false;
    for (const button of form.querySelectorAll('[type="submit"]')) button.disabled = false;
  }
});

for (const button of document.querySelectorAll(".mode")) {
  button.addEventListener("click", () => changeMode(button.dataset.mode));
}
for (const button of document.querySelectorAll("[data-sidebar]")) {
  button.addEventListener("click", () => setSidebar(button.dataset.sidebar));
}
el("back-to-setup").addEventListener("click", () => changeMode(lastRealMode));
form.addEventListener("invalid", event => {
  if (el("advanced").contains(event.target)) el("advanced").open = true;
}, true);
el("atl03-source").addEventListener("change", updateAtl03);
for (const button of document.querySelectorAll(".tab-bar button")) {
  button.addEventListener("click", () => setView(button.dataset.view));
}
async function jobAction(action, requested, values = {}) {
  const job = await api("/api/jobs/" + requested + "/" + action, values);
  if (action === "resume") {
    setEarthdataSaved(job.earthdata_saved);
    if (requested === jobId) setView();
  }
  await update();
}

for (const action of ["cancel", "resume"]) {
  el(action).addEventListener("click", async () => {
    message("job-error", "");
    const requested = jobId;
    try {
      if (action === "resume" && selected?.id === requested && selected.requires_earthdata) {
        await refreshEarthdata();
        if (requested !== jobId) return;
        if (!earthdataSaved) {
          resumeTarget = requested;
          message("earthdata-error", "");
          el("earthdata-dialog").showModal();
          return;
        }
      }
      await jobAction(action, requested);
    } catch (error) {
      message("job-error", error.message);
    }
  });
}

el("clear-earthdata").addEventListener("click", async () => {
  try {
    setEarthdataSaved((await api("/api/earthdata/clear", {})).saved);
    el("earthdata-username").focus();
  } catch (error) {
    message("form-error", error.message);
  }
});
el("close-earthdata").addEventListener("click", () => el("earthdata-dialog").close());
el("earthdata-dialog").addEventListener("close", () => {
  el("earthdata-form").reset();
  resumeTarget = undefined;
});
el("earthdata-form").addEventListener("submit", async event => {
  event.preventDefault();
  const button = event.submitter;
  if (!resumeTarget || button.disabled) return;
  const values = Object.fromEntries(new FormData(event.currentTarget));
  button.disabled = true;
  try {
    await jobAction("resume", resumeTarget, values);
    el("earthdata-dialog").close();
  } catch (error) {
    message("earthdata-error", error.message);
  } finally {
    button.disabled = false;
  }
});
el("copy").addEventListener("click", async () => {
  if (!selected) return;
  try {
    await navigator.clipboard.writeText(selected.command);
    el("copy").textContent = "已复制";
    setTimeout(() => { el("copy").textContent = "复制命令"; }, 1500);
  } catch {
    el("command-panel").open = true;
    const range = document.createRange();
    range.selectNodeContents(el("command"));
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
  }
});

function choose(path) {
  el(pickerTarget).value = path;
  el("picker").close();
}

async function browse(path = "") {
  message("picker-error", "");
  try {
    const query = new URLSearchParams({path, kind: pickerKind});
    const result = await api("/api/files?" + query);
    pickerPath = result.path;
    el("folder-path").textContent = result.path || "开放的数据目录";
    el("parent").disabled = !result.parent;
    el("parent").onclick = () => browse(result.parent);
    el("use-folder").hidden = pickerKind !== "directory" || !pickerPath;
    el("file-entries").replaceChildren(...result.entries.map(entry => {
      const item = makeNode("button", entry.name + (entry.directory ? "  /" : ""));
      item.type = "button";
      item.addEventListener("click", () => entry.directory ? browse(entry.path) : choose(entry.path));
      return item;
    }));
    if (!result.entries.length) el("file-entries").append(makeNode("p", "目录中没有可选项目。", "empty"));
  } catch (error) {
    message("picker-error", error.message);
  }
}

for (const button of document.querySelectorAll(".picker-button")) {
  button.addEventListener("click", () => {
    pickerTarget = button.dataset.target;
    pickerKind = pickerTarget === "input" && mode === "image" ? "image" : "directory";
    el("picker-title").textContent = pickerKind === "image" ? "选择本机影像" : "选择本机目录";
    el("picker").showModal();
    browse();
  });
}
el("roots").addEventListener("click", () => browse());
el("close-picker").addEventListener("click", () => el("picker").close());
el("use-folder").addEventListener("click", () => choose(pickerPath));

async function initialize() {
  try {
    const config = await api("/api/config");
    const depthAnything = makeNode("optgroup");
    depthAnything.label = "Depth Anything";
    const depthAnythingNames = {"DEPTH-ANYTHING-V1": "Depth Anything V1", "DEPTH-ANYTHING-V2": "Depth Anything V2"};
    for (const name of config.models) {
      const option = makeNode("option", depthAnythingNames[name] || name);
      option.value = name;
      (depthAnythingNames[name] ? depthAnything : el("model")).append(option);
    }
    if (depthAnything.childElementCount) el("model").append(depthAnything);
    el("model").value = defaultDepthModel;
    for (const [field, models] of Object.entries(config.building_models)) {
      for (const name of models) form.elements[field].append(makeNode("option", name));
    }
    el("work-root").textContent = config.work_root;
    changeMode(mode);
    await refreshEarthdata();
  } catch (error) {
    message("form-error", error.message);
  }
  await update();
  setInterval(update, 2000);
}

initialize();

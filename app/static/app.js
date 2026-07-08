// Frontend logic: upload handling + rendering only. All inference happens
// server-side via /api/predict — this file never touches model logic.

const el = (id) => document.getElementById(id);

// Honest probability formatting: avoid claiming absolute certainty. A calibrated
// 99.98% shouldn't read as a flat "100.0%".
function fmtProb(p) {
  const v = p * 100;
  if (v >= 99.95) return '>99.9%';
  if (v <= 0.05) return '<0.1%';
  return `${v.toFixed(1)}%`;
}

const dropzone = el('dropzone');
const fileInput = el('fileInput');
const sections = {
  upload: el('uploadSection'),
  loading: el('loadingSection'),
  error: el('errorSection'),
  result: el('resultSection'),
};

// Single source of truth for what's visible. The big dropzone is hidden while
// loading/showing a result so the verdict is the focus; Clear returns to idle.
function setState(state) {
  const vis = {
    idle:    { upload: 1, loading: 0, result: 0, error: 0 },
    loading: { upload: 0, loading: 1, result: 0, error: 0 },
    result:  { upload: 0, loading: 0, result: 1, error: 0 },
    error:   { upload: 1, loading: 0, result: 0, error: 1 },
  }[state];
  for (const key in sections) sections[key].hidden = !vis[key];
}

let objectUrl = null;

function fail(message) {
  sections.error.textContent = message;
  setState('error');
}

async function analyze(file) {
  if (!file || !file.type.startsWith('image/')) {
    return fail('Please choose an image file (PNG, JPG, or WebP).');
  }
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  el('loadingImg').src = objectUrl;   // show what's being analyzed
  setState('loading');

  const body = new FormData();
  body.append('file', file);
  try {
    const res = await fetch('/api/predict', { method: 'POST', body });
    if (!res.ok) {
      const { detail } = await res.json().catch(() => ({}));
      throw new Error(detail || `Request failed (${res.status})`);
    }
    render(await res.json());
    setState('result');
    sections.result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    fail(err.message || 'Something went wrong analyzing the image.');
  }
}

function render(r) {
  const isAI = r.verdict === 'ai';
  const pctText = fmtProb(r.p_ai);
  const pctNum = r.p_ai * 100;
  const thr = r.threshold.toFixed(2);

  el('verdictBlock').className = `verdict ${isAI ? 'verdict--ai' : 'verdict--real'}`;
  el('verdictLabel').textContent = isAI ? 'AI-generated' : 'Real photograph';
  el('verdictSummary').textContent =
    `The model estimates a ${pctText} probability that this image is AI-generated — `
    + `${isAI ? 'above' : 'below'} the ${thr} decision threshold.`;

  el('probValue').textContent = pctText;
  const fill = el('probFill');
  fill.className = `meter__fill ${isAI ? 'meter__fill--ai' : 'meter__fill--real'}`;
  requestAnimationFrame(() => { fill.style.width = `${pctNum}%`; });
  const mark = el('thresholdMark');
  mark.style.left = `${r.threshold * 100}%`;
  mark.querySelector('span').textContent = thr;

  el('originalImg').src = r.original;
  el('heatmapImg').src = r.heatmap;

  el('techModel').textContent = r.model_arch;
  el('techThreshold').textContent = thr;
  el('techTemp').textContent = r.temperature.toFixed(2);
  el('techProb').textContent = pctText;
  el('techConfidence').textContent = fmtProb(r.confidence);
}

function reset() {
  fileInput.value = '';
  if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  setState('idle');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Keep the header badge in sync with the actual loaded model (no hardcoding).
fetch('/api/health').then((r) => r.json()).then((h) => {
  el('modelBadge').textContent = `${h.model} · threshold ${h.threshold.toFixed(2)}`;
}).catch(() => {});

// --- events ---------------------------------------------------------------
fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) analyze(e.target.files[0]);
});

['dragover', 'dragenter'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('is-dragover');
  })
);
['dragleave', 'drop'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('is-dragover');
  })
);
dropzone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]);
});
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

el('resetBtn').addEventListener('click', reset);

// Paste an image (Cmd/Ctrl+V) to analyze it — handy for screenshots.
window.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (item) analyze(item.getAsFile());
});

// Esc clears the current result.
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !sections.result.hidden) reset();
});

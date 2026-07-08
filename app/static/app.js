// Frontend logic: upload handling + rendering only. All inference happens
// server-side via /api/predict — this file never touches model logic.

const el = (id) => document.getElementById(id);
const dropzone = el('dropzone');
const fileInput = el('fileInput');
const sections = {
  upload: el('uploadSection'),
  loading: el('loadingSection'),
  error: el('errorSection'),
  result: el('resultSection'),
};

function show(name) {
  for (const [key, node] of Object.entries(sections)) {
    node.hidden = key !== name && !(name === 'result' && key === 'upload');
  }
  // keep the uploader visible alongside results so re-upload is easy
  sections.upload.hidden = name === 'loading';
}

function fail(message) {
  sections.error.textContent = message;
  sections.error.hidden = false;
  sections.loading.hidden = true;
}

async function analyze(file) {
  if (!file || !file.type.startsWith('image/')) {
    return fail('Please choose an image file (PNG, JPG, or WebP).');
  }
  sections.error.hidden = true;
  sections.result.hidden = true;
  show('loading');

  const body = new FormData();
  body.append('file', file);

  try {
    const res = await fetch('/api/predict', { method: 'POST', body });
    if (!res.ok) {
      const { detail } = await res.json().catch(() => ({}));
      throw new Error(detail || `Request failed (${res.status})`);
    }
    render(await res.json());
  } catch (err) {
    fail(err.message || 'Something went wrong analyzing the image.');
  } finally {
    sections.loading.hidden = true;
  }
}

function render(r) {
  const isAI = r.verdict === 'ai';
  const pct = (r.p_ai * 100).toFixed(1);

  const pill = el('verdictPill');
  pill.textContent = isAI ? 'AI-generated' : 'Real photograph';
  pill.className = `pill ${isAI ? 'pill--ai' : 'pill--real'}`;

  el('verdictSummary').textContent = isAI
    ? `The model estimates a ${pct}% probability that this image is AI-generated, above the ${r.threshold} decision threshold.`
    : `The model estimates a ${pct}% probability that this image is AI-generated, below the ${r.threshold} decision threshold.`;

  el('probValue').textContent = `${pct}%`;
  const fill = el('probFill');
  fill.className = `meter__fill ${isAI ? 'meter__fill--ai' : 'meter__fill--real'}`;
  requestAnimationFrame(() => { fill.style.width = `${pct}%`; });
  el('thresholdMark').style.left = `${r.threshold * 100}%`;

  el('originalImg').src = r.original;
  el('heatmapImg').src = r.heatmap;

  el('techModel').textContent = r.model_arch;
  el('techThreshold').textContent = r.threshold.toFixed(2);
  el('techProb').textContent = `${pct}%`;
  el('techConfidence').textContent = `${(r.confidence * 100).toFixed(1)}%`;
  el('techTime').textContent = `${r.inference_ms.toFixed(0)} ms`;

  show('result');
  sections.result.hidden = false;
  sections.result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Keep the header badge in sync with the actual loaded model (no hardcoding).
fetch('/api/health').then((r) => r.json()).then((h) => {
  el('modelBadge').textContent = `${h.model} · threshold ${h.threshold}`;
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

el('resetBtn').addEventListener('click', () => {
  fileInput.value = '';
  sections.result.hidden = true;
  sections.error.hidden = true;
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

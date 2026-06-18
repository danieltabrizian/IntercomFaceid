import os
import threading
import logging
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

app = FastAPI()

_event_logger = None
_face_recognizer = None
_blur_calibration = None

SNAPSHOTS_DIR = '/data/snapshots'
FACE_SNAPSHOTS_DIR = '/data/face_snapshots'

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Intercom</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0f172a; --surface: #1e293b; --surface2: #27364b; --border: #334155;
  --text: #e2e8f0; --muted: #64748b; --accent: #3b82f6;
  --green: #10b981; --red: #ef4444; --amber: #f59e0b; --purple: #8b5cf6;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
header {
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 0 24px; display: flex; align-items: center; gap: 12px; height: 56px;
}
header h1 { font-size: 17px; font-weight: 600; flex: 1; }
.arduino-status { display: flex; align-items: center; gap: 7px; font-size: 13px; color: var(--muted); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.dot.online { background: var(--green); box-shadow: 0 0 6px var(--green); }
nav {
  background: var(--surface); padding: 0 24px;
  display: flex; gap: 2px; border-bottom: 1px solid var(--border);
}
nav button {
  background: none; border: none; color: var(--muted); padding: 14px 18px;
  cursor: pointer; font-size: 14px; font-weight: 500;
  border-bottom: 3px solid transparent; transition: color .15s, border-color .15s;
}
nav button.active { color: var(--accent); border-bottom-color: var(--accent); }
nav button:hover:not(.active) { color: var(--text); }
.tab { display: none; padding: 20px 24px; max-width: 1200px; margin: 0 auto; }
.tab.active { display: block; }

/* Events */
.events { display: flex; flex-direction: column; gap: 8px; }
.ev {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 14px; display: flex; align-items: center; gap: 12px;
}
.ev-thumb {
  width: 60px; height: 60px; border-radius: 8px; object-fit: cover;
  background: var(--surface2); flex-shrink: 0; cursor: pointer;
  border: 1px solid var(--border);
}
.ev-icon {
  width: 60px; height: 60px; border-radius: 8px; background: var(--surface2);
  display: flex; align-items: center; justify-content: center;
  font-size: 26px; flex-shrink: 0; border: 1px solid var(--border);
}
.ev-body { flex: 1; min-width: 0; }
.ev-row { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; flex-wrap: wrap; }
.badge {
  font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 100px;
  text-transform: uppercase; letter-spacing: .5px; white-space: nowrap;
}
.b-bell    { background: #172554; color: #60a5fa; }
.b-ok      { background: #052e16; color: #4ade80; }
.b-denied  { background: #450a0a; color: #f87171; }
.b-mig     { background: #1c1917; color: #fbbf24; }
.b-raw     { background: #1e1b4b; color: #a5b4fc; }
.b-ard     { background: #292524; color: #fbbf24; }
.ev-time   { font-size: 12px; color: var(--muted); }
.ev-detail { font-size: 13px; color: var(--muted); font-family: monospace; word-break: break-all; }
.ev-name   { font-size: 14px; font-weight: 600; }
.ev-sim    { font-size: 12px; color: var(--muted); }

/* Analytics */
.stats { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.stat {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 20px; flex: 1; min-width: 120px;
}
.stat .val { font-size: 30px; font-weight: 700; line-height: 1; }
.stat .lbl { font-size: 12px; color: var(--muted); margin-top: 5px; text-transform: uppercase; letter-spacing: .5px; }

/* Command pills */
.section-title {
  font-size: 11px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: .5px; margin-bottom: 10px;
}
.cmd-bar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.cmd-pill {
  display: flex; align-items: center; gap: 6px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 100px;
  padding: 6px 14px; cursor: pointer; font-size: 13px; color: var(--muted);
  transition: border-color .15s, color .15s, background .15s;
  font-family: monospace;
}
.cmd-pill:hover { border-color: var(--accent); color: var(--text); }
.cmd-pill.active { background: #172554; border-color: var(--accent); color: var(--accent); }
.cmd-pill .cmd-count {
  background: var(--surface2); border-radius: 100px;
  padding: 1px 7px; font-size: 11px; font-weight: 700; color: var(--muted);
}
.cmd-pill.active .cmd-count { background: #1d3461; color: #93c5fd; }

/* Charts */
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
@media(max-width:700px){ .charts { grid-template-columns: 1fr; } .stats { flex-direction: column; } }
.chart-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px;
}
.chart-card h3 { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 14px; }

/* Heatmap */
.heatmap-card { margin-bottom: 16px; }
.hm-wrap { overflow-x: auto; }
.hm-grid {
  display: grid;
  grid-template-columns: 34px repeat(24, minmax(22px, 1fr));
  gap: 2px;
  min-width: 560px;
}
.hm-cell {
  border-radius: 3px; aspect-ratio: 1;
  font-size: 8px; color: rgba(255,255,255,.7);
  display: flex; align-items: center; justify-content: center;
  cursor: default; transition: opacity .1s;
}
.hm-cell:hover { opacity: .75; }
.hm-hlabel {
  font-size: 9px; color: var(--muted);
  display: flex; align-items: flex-end; justify-content: center;
  padding-bottom: 3px; height: 20px;
}
.hm-dlabel {
  font-size: 10px; color: var(--muted);
  display: flex; align-items: center; justify-content: flex-end;
  padding-right: 6px;
}
.hm-corner { height: 20px; }

/* Faces */
.faces-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 14px; }
.face-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
}
.face-photo, .face-ph {
  width: 100%; aspect-ratio: 1; object-fit: cover; display: block; background: var(--surface2);
}
.face-ph { display: flex; align-items: center; justify-content: center; font-size: 60px; }
.face-body { padding: 12px; }
.face-name { font-weight: 600; font-size: 14px; margin-bottom: 2px; }
.face-meta { font-size: 12px; color: var(--muted); margin-bottom: 10px; }
.btn-del {
  width: 100%; background: #2d0c0c; color: var(--red); border: 1px solid #5c1111;
  border-radius: 6px; padding: 7px; cursor: pointer; font-size: 12px; font-weight: 600;
  transition: background .15s;
}
.btn-del:hover { background: #5c1111; }
.bench-btn {
  background: #172554; color: var(--accent); border: 1px solid var(--accent);
  border-radius: 6px; padding: 8px 16px; cursor: pointer; font-size: 13px; font-weight: 600;
  transition: background .15s;
}
.bench-btn:hover:not(:disabled) { background: #1d3461; }
.bench-btn:disabled { opacity: .5; cursor: default; }
.bench-table { font-size: 13px; border-collapse: collapse; }
.bench-table td { padding: 4px 16px 4px 0; }

/* Lightbox */
.lb {
  display: none; position: fixed; inset: 0; background: rgba(0,0,0,.88);
  z-index: 999; align-items: center; justify-content: center;
}
.lb.open { display: flex; }
.lb img { max-width: 90vw; max-height: 90vh; border-radius: 8px; }
.lb-close {
  position: absolute; top: 16px; right: 16px; background: var(--surface);
  border: 1px solid var(--border); color: var(--text); width: 34px; height: 34px;
  border-radius: 50%; cursor: pointer; font-size: 16px;
}

.empty, .loading { text-align: center; padding: 48px 24px; color: var(--muted); font-size: 14px; }
</style>
</head>
<body>
<header>
  <h1>🔔 Intercom</h1>
  <div class="arduino-status">
    <div class="dot" id="ard-dot"></div>
    <span id="ard-label">Arduino</span>
  </div>
</header>
<nav>
  <button class="active" data-tab="events">Activity</button>
  <button data-tab="analytics">Analytics</button>
  <button data-tab="faces">Faces</button>
</nav>

<div id="events" class="tab active">
  <div class="events" id="events-list"><div class="loading">Loading…</div></div>
</div>

<div id="analytics" class="tab">
  <div class="stats" id="stats"></div>

  <div class="section-title">Filter by code</div>
  <div class="cmd-bar" id="cmd-bar"></div>

  <div class="charts">
    <div class="chart-card"><h3>Hour of day</h3><canvas id="hourChart"></canvas></div>
    <div class="chart-card"><h3>Day of week</h3><canvas id="dowChart"></canvas></div>
  </div>

  <div class="chart-card heatmap-card">
    <h3>Activity heatmap — day × hour</h3>
    <div class="hm-wrap"><div class="heatmap" id="heatmap"></div></div>
  </div>

  <div class="section-title" style="margin-top:24px">System benchmark</div>
  <div class="chart-card heatmap-card">
    <button id="bench-btn" class="bench-btn">Run benchmark</button>
    <span style="font-size:12px;color:var(--muted);margin-left:10px">Runs each model on ~20 live frames, as if a face were present (~10s).</span>
    <div id="bench-out" style="margin-top:14px"></div>
  </div>

  <div class="section-title" style="margin-top:24px">Blur calibration</div>
  <div class="stats" id="calib-stats"></div>
  <div class="chart-card heatmap-card">
    <h3>Face-crop sharpness — processed vs. matched</h3>
    <canvas id="calibChart"></canvas>
    <div id="calib-note" style="font-size:12px;color:var(--muted);margin-top:12px;line-height:1.6"></div>
  </div>
</div>

<div id="faces" class="tab">
  <div class="faces-grid" id="faces-grid"><div class="loading">Loading…</div></div>
</div>

<div class="lb" id="lb" onclick="closeLb()">
  <button class="lb-close" onclick="closeLb()">✕</button>
  <img id="lb-img" src="" />
</div>

<script>
let hChart = null, dChart = null;
let analyticsData = null;
let selectedCmd = '__all__';

const DAYS = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const HOURS = Array.from({length:24}, (_,i) => i + ':00');

const chartOpts = {
  responsive: true,
  plugins: { legend: { display: false } },
  scales: {
    x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
    y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b', stepSize: 1 }, beginAtZero: true }
  }
};

// Pill colors cycle for different commands
const PILL_COLORS = ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444','#06b6d4','#f97316'];

document.querySelectorAll('nav button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('nav button').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    const tab = b.dataset.tab;
    document.getElementById(tab).classList.add('active');
    if (tab === 'analytics') loadAnalytics();
    if (tab === 'faces') loadFaces();
  });
});

function openLb(src) { document.getElementById('lb-img').src = src; document.getElementById('lb').classList.add('open'); }
function closeLb() { document.getElementById('lb').classList.remove('open'); }

function fmt(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
}

const BADGE = {
  bell_ring:            ['b-bell',  '🔔 Bell'],
  hex_received:         ['b-bell',  '📶 Signal'],
  recognition_started:  ['b-raw',   '🔍 Scanning'],
  face_recognized:      ['b-ok',    '✅ Recognised'],
  face_denied:          ['b-denied','🚫 Denied'],
  face_migrated:        ['b-mig',   '⚡ Migrated'],
  door_unlocked:        ['b-ok',    '🔓 Unlocked'],
  serial_command:       ['b-raw',   '📡 Serial'],
  arduino_connected:    ['b-ard',   '🔌 Connected'],
  arduino_disconnected: ['b-ard',   '⚠️ Disconnected'],
};
const ICON = {
  bell_ring: '🔔', hex_received: '📶', recognition_started: '🔍',
  face_recognized: '👤', face_denied: '❓', face_migrated: '⚡',
  door_unlocked: '🔓', serial_command: '📡',
  arduino_connected: '🔌', arduino_disconnected: '⚠️'
};

function evHtml(e) {
  const [bcls, blbl] = BADGE[e.type] || ['b-raw', e.type];
  let thumb = e.snapshot
    ? `<img class="ev-thumb" src="snapshots/${e.snapshot}" onclick="openLb('snapshots/${e.snapshot}')" onerror="this.parentNode.querySelector('.ev-icon').style.display='flex';this.style.display='none'" /><div class="ev-icon" style="display:none">${ICON[e.type]||'•'}</div>`
    : `<div class="ev-icon">${ICON[e.type]||'•'}</div>`;
  function timingHtml(e) {
    const parts = [];
    if (e.detect_frames)  parts.push(`🔍 detect ${e.detect_avg_ms}ms × ${e.detect_frames}f`);
    if (e.embed_frames)   parts.push(`🧠 embed ${e.embed_avg_ms}ms × ${e.embed_frames}f`);
    if (e.forced_processed) parts.push(`🎯 ${e.forced_processed} forced`);
    if (e.skipped_blurry)   parts.push(`🌫 ${e.skipped_blurry} skipped`);
    if (e.no_face_frames)   parts.push(`👻 ${e.no_face_frames} no-face`);
    if (e.duration_s)     parts.push(`${e.duration_s}s`);
    return parts.length
      ? `<div style="font-size:11px;color:var(--muted);margin-top:3px">${parts.join(' &nbsp;·&nbsp; ')}</div>`
      : '';
  }
  let detail = '';
  if (e.type === 'recognition_started')
    detail = `<div class="ev-detail">Face recognition loop started</div>`;
  else if (e.type === 'face_recognized') {
    detail = `<div class="ev-name">${e.name||''}</div><div class="ev-sim">${e.similarity ? Math.round(e.similarity*100)+'% match' : ''}</div>${timingHtml(e)}`;
  } else if (e.type === 'face_denied')
    detail = `<div class="ev-detail">No match${e.similarity != null ? ' — best '+Math.round(e.similarity*100)+'%' : ''}</div>${timingHtml(e)}`;
  else if (e.type === 'face_migrated')
    detail = `<div class="ev-detail">${e.name||''} → SFace (${e.embeddings||0} embeddings)</div>`;
  else if (e.type === 'hex_received')
    detail = `<div class="ev-detail">${e.command||''}</div>`;
  else if (e.type === 'door_unlocked')
    detail = `<div class="ev-detail">Door opened</div>`;
  else if (e.type === 'serial_command')
    detail = `<div class="ev-detail">${e.command||''}</div>`;
  else if (e.message)
    detail = `<div class="ev-detail">${e.message}</div>`;
  return `<div class="ev">${thumb}<div class="ev-body"><div class="ev-row"><span class="badge ${bcls}">${blbl}</span><span class="ev-time">${fmt(e.timestamp)}</span></div>${detail}</div></div>`;
}

async function loadEvents() {
  try {
    const r = await fetch('api/events');
    const events = await r.json();
    const el = document.getElementById('events-list');
    el.innerHTML = events.length ? events.map(evHtml).join('') : '<div class="empty">No events yet.</div>';
  } catch(err) { console.error(err); }
}

async function loadAnalytics() {
  try {
    const r = await fetch('api/analytics');
    analyticsData = await r.json();
    renderStats();
    renderCmdBar();
    updateCharts();
  } catch(err) { console.error(err); }
  loadCalibration();
}

async function runBenchmark() {
  const btn = document.getElementById('bench-btn');
  const out = document.getElementById('bench-out');
  btn.disabled = true;
  btn.textContent = 'Running… (~10s)';
  out.innerHTML = '';
  try {
    const r = await fetch('api/benchmark', { method: 'POST' });
    const d = await r.json();
    if (d.error) {
      out.innerHTML = `<span style="color:var(--red)">${d.error}</span>`;
    } else {
      out.innerHTML = renderBench(d);
    }
  } catch (e) {
    out.innerHTML = `<span style="color:var(--red)">${e}</span>`;
  }
  btn.disabled = false;
  btn.textContent = 'Run benchmark';
}

function benchRow(label, val, highlight) {
  const v = (val == null) ? '—' : `${val} ms`;
  return `<tr><td style="color:var(--muted)">${label}</td>
    <td style="font-weight:600;${highlight ? 'color:var(--green)' : ''}">${v}</td></tr>`;
}

function renderBench(d) {
  let h = `<div style="font-size:12px;color:var(--muted);margin-bottom:10px">
    ${d.frames} frames · ${d.resolution} · ${d.faces_detected} had a real face
    <br>embedding is forced on a synthetic crop when no face is present, so timing is consistent</div>
    <table class="bench-table">`;
  h += benchRow('detect (SCRFD)', d.detect_ms);
  h += benchRow('embed (ArcFace)', d.embed_ms);
  h += benchRow('TOTAL per frame', d.total_ms, true);
  h += '</table>';
  return h;
}

let calibChart = null;
async function loadCalibration() {
  try {
    const r = await fetch('api/calibration');
    const c = await r.json();
    const hist = c.histogram || [];
    const bw = c.bin_width || 20;

    document.getElementById('calib-stats').innerHTML = `
      <div class="stat"><div class="val">${c.total_processed||0}</div><div class="lbl">Frames Processed</div></div>
      <div class="stat"><div class="val">${c.total_matched||0}</div><div class="lbl">Matched</div></div>
      <div class="stat"><div class="val">${c.forced_pct||0}%</div><div class="lbl">Forced (failsafe)</div></div>
      <div class="stat"><div class="val">${c.skipped_blurry||0}</div><div class="lbl">Skipped Blurry</div></div>
    `;

    const labels = hist.map(h => `${h.floor}-${h.floor+bw}`);
    if (calibChart) calibChart.destroy();
    calibChart = new Chart(document.getElementById('calibChart'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Processed', data: hist.map(h=>h.n),       backgroundColor: '#475569', borderRadius: 3 },
          { label: 'Matched',   data: hist.map(h=>h.matched), backgroundColor: '#10b981', borderRadius: 3 },
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: true, labels: { color: '#94a3b8', boxWidth: 12 } } },
        scales: {
          x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b', maxRotation: 90, minRotation: 45 } },
          y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' }, beginAtZero: true }
        }
      }
    });

    const safe = c.suggested_threshold_safe;
    const p5 = c.suggested_threshold_p5;
    document.getElementById('calib-note').innerHTML = `
      Current threshold: <b style="color:var(--text)">${c.current_blur_threshold ?? '—'}</b>
      &nbsp;·&nbsp; force-after: <b style="color:var(--text)">${c.current_force_after_ms ?? '—'}ms</b><br>
      Lowest sharpness that has ever matched: <b style="color:var(--green)">${safe != null ? safe : 'no data yet'}</b>
      ${p5 != null ? `&nbsp;·&nbsp; 5th-percentile of matches: <b style="color:var(--green)">${p5}</b>` : ''}
      <br><span style="color:var(--muted)">Once green bars cluster clearly above a value, set BLUR_THRESHOLD just below it
      and raise force-after so the gate can actually skip.</span>
    `;
  } catch(err) { console.error(err); }
}

function renderStats() {
  const d = analyticsData;
  const topCmd = d.command_list && d.command_list.length ? d.command_list[0] : null;
  document.getElementById('stats').innerHTML = `
    <div class="stat"><div class="val">${d.total_serial_commands||0}</div><div class="lbl">Serial Commands</div></div>
    <div class="stat"><div class="val">${d.unique_commands||0}</div><div class="lbl">Unique Codes</div></div>
    <div class="stat"><div class="val">${d.bell_rings||0}</div><div class="lbl">Bell Rings</div></div>
    <div class="stat"><div class="val" style="font-size:16px;padding-top:4px">${topCmd ? topCmd.command : '—'}</div><div class="lbl">Most frequent code</div></div>
  `;
}

function renderCmdBar() {
  const d = analyticsData;
  const bar = document.getElementById('cmd-bar');
  const allTotal = d.total_serial_commands || 0;
  let html = `<button class="cmd-pill ${selectedCmd==='__all__'?'active':''}" data-cmd="__all__">
    <span>All codes</span><span class="cmd-count">${allTotal}</span>
  </button>`;
  (d.command_list || []).forEach((c, i) => {
    const color = PILL_COLORS[i % PILL_COLORS.length];
    const active = selectedCmd === c.command;
    // Store command in data-cmd to avoid HTML attribute quoting issues
    const safeCmd = c.command.replace(/&/g,'&amp;').replace(/"/g,'&quot;');
    html += `<button class="cmd-pill ${active?'active':''}" data-cmd="${safeCmd}"
      style="${active ? '' : `--pill-color:${color}`}">
      <span style="color:${active?'inherit':color}">${c.command}</span>
      <span class="cmd-count">${c.total}</span>
    </button>`;
  });
  bar.innerHTML = html;
  // Attach handlers after rendering — avoids all inline onclick quoting pitfalls
  bar.querySelectorAll('[data-cmd]').forEach(btn => {
    btn.addEventListener('click', () => selectCmd(btn.dataset.cmd));
  });
}

function selectCmd(cmd) {
  selectedCmd = cmd;
  renderCmdBar();
  updateCharts();
}

function getSelectedData() {
  const d = analyticsData;
  if (selectedCmd === '__all__') {
    const hourCounts = new Array(24).fill(0);
    const dowCounts  = new Array(7).fill(0);
    const heatmap    = Array.from({length:7}, () => new Array(24).fill(0));
    for (const c of Object.values(d.commands || {})) {
      c.hour_counts.forEach((v, i) => hourCounts[i] += v);
      c.dow_counts.forEach((v, i)  => dowCounts[i]  += v);
      c.heatmap.forEach((row, day) => row.forEach((v, hr) => heatmap[day][hr] += v));
    }
    return { hourCounts, dowCounts, heatmap };
  }
  const c = (d.commands || {})[selectedCmd];
  if (!c) return {
    hourCounts: new Array(24).fill(0),
    dowCounts:  new Array(7).fill(0),
    heatmap:    Array.from({length:7}, () => new Array(24).fill(0))
  };
  return { hourCounts: c.hour_counts, dowCounts: c.dow_counts, heatmap: c.heatmap };
}

function updateCharts() {
  const { hourCounts, dowCounts, heatmap } = getSelectedData();
  const color = selectedCmd === '__all__' ? '#3b82f6'
    : PILL_COLORS[(analyticsData.command_list||[]).findIndex(c=>c.command===selectedCmd) % PILL_COLORS.length];

  if (hChart) hChart.destroy();
  hChart = new Chart(document.getElementById('hourChart'), {
    type: 'bar',
    data: {
      labels: HOURS,
      datasets: [{ data: hourCounts, backgroundColor: color + 'cc', borderRadius: 4 }]
    },
    options: chartOpts
  });

  if (dChart) dChart.destroy();
  dChart = new Chart(document.getElementById('dowChart'), {
    type: 'bar',
    data: {
      labels: DAYS,
      datasets: [{ data: dowCounts, backgroundColor: color + 'aa', borderRadius: 4 }]
    },
    options: chartOpts
  });

  renderHeatmap(heatmap, color);
}

function renderHeatmap(heatmap, baseColor) {
  const max = Math.max(...heatmap.flat(), 1);
  // Parse hex color to rgb for alpha blending
  const r = parseInt(baseColor.slice(1,3),16);
  const g = parseInt(baseColor.slice(3,5),16);
  const b = parseInt(baseColor.slice(5,7),16);

  let html = '<div class="hm-grid">';
  // Header row: corner + hour labels
  html += '<div class="hm-corner"></div>';
  for (let h = 0; h < 24; h++) {
    html += `<div class="hm-hlabel">${h % 3 === 0 ? h : ''}</div>`;
  }
  // Data rows
  for (let d = 0; d < 7; d++) {
    html += `<div class="hm-dlabel">${DAYS[d]}</div>`;
    for (let h = 0; h < 24; h++) {
      const v = heatmap[d][h];
      const alpha = v === 0 ? 0 : 0.12 + (v / max) * 0.88;
      const bg = v === 0
        ? 'var(--surface2)'
        : `rgba(${r},${g},${b},${alpha.toFixed(2)})`;
      const label = v > 0 ? v : '';
      html += `<div class="hm-cell" style="background:${bg}" title="${DAYS[d]} ${h}:00 — ${v} event${v!==1?'s':''}">${label}</div>`;
    }
  }
  html += '</div>';
  document.getElementById('heatmap').innerHTML = html;
}

async function loadFaces() {
  try {
    const r = await fetch('api/faces');
    const faces = await r.json();
    const el = document.getElementById('faces-grid');
    if (!faces.length) { el.innerHTML = '<div class="empty">No faces enrolled.</div>'; return; }
    el.innerHTML = faces.map(f => {
      const safeId = CSS.escape(f.name);
      const safeName = f.name.replace(/&/g,'&amp;').replace(/"/g,'&quot;');
      return `
      <div class="face-card" id="fc-${safeId}" data-name="${safeName}">
        ${f.has_snapshot
          ? `<img class="face-photo" src="face-snapshots/${encodeURIComponent(f.name)}.jpg" onerror="this.nextElementSibling.style.display='flex';this.style.display='none'" /><div class="face-ph" style="display:none">👤</div>`
          : `<div class="face-ph">👤</div>`}
        <div class="face-body">
          <div class="face-name">${f.name}</div>
          <div class="face-meta">
            ${f.embedding_count} sample${f.embedding_count!==1?'s':''}
          </div>
          <button class="btn-del">Remove</button>
        </div>
      </div>`;
    }).join('');
    // Attach remove handlers after render — avoids onclick quoting issues
    el.querySelectorAll('.btn-del').forEach(btn => {
      const name = btn.closest('[data-name]').dataset.name;
      btn.addEventListener('click', () => delFace(name));
    });
  } catch(err) { console.error(err); }
}

async function delFace(name) {
  if (!confirm('Remove ' + name + ' from face registry?')) return;
  const r = await fetch('api/faces/' + encodeURIComponent(name) + '/delete', {method:'POST'});
  const d = await r.json();
  if (d.success) document.getElementById('fc-' + CSS.escape(name))?.remove();
}

document.getElementById('bench-btn').addEventListener('click', runBenchmark);

loadEvents();
setInterval(loadEvents, 10000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML


@app.get("/api/events")
async def get_events():
    if _event_logger is None:
        return []
    return _event_logger.get_recent(300)


@app.get("/api/analytics")
async def get_analytics():
    if _event_logger is None:
        return {}
    events = _event_logger.get_all()

    commands = {}   # command_str -> {total, hour_counts[24], dow_counts[7], heatmap[7][24]}
    bell_rings = 0

    for e in events:
        etype = e.get('type')
        if etype == 'bell_ring':
            bell_rings += 1
        if etype != 'serial_command':
            continue
        cmd = e.get('command', '')
        if not cmd:
            continue
        if cmd not in commands:
            commands[cmd] = {
                'total': 0,
                'hour_counts': [0] * 24,
                'dow_counts':  [0] * 7,
                'heatmap':     [[0] * 24 for _ in range(7)],
            }
        commands[cmd]['total'] += 1
        try:
            dt = datetime.fromisoformat(e['timestamp'])
            commands[cmd]['hour_counts'][dt.hour] += 1
            commands[cmd]['dow_counts'][dt.weekday()] += 1
            commands[cmd]['heatmap'][dt.weekday()][dt.hour] += 1
        except Exception:
            pass

    command_list = sorted(
        [{'command': k, 'total': v['total']} for k, v in commands.items()],
        key=lambda x: -x['total']
    )

    return {
        'total_serial_commands': sum(c['total'] for c in command_list),
        'unique_commands': len(commands),
        'bell_rings': bell_rings,
        'command_list': command_list,
        'commands': commands,
    }


@app.get("/api/faces")
async def get_faces():
    if _face_recognizer is None:
        return []
    return _face_recognizer.get_faces_info()


@app.post("/api/faces/{name}/delete")
async def delete_face(name: str):
    if _face_recognizer is None:
        return JSONResponse({'success': False})
    success = _face_recognizer.delete_face(name)
    return {'success': success}


@app.get("/api/calibration")
async def get_calibration():
    if _blur_calibration is None:
        return {}
    return _blur_calibration.summary()


@app.post("/api/benchmark")
def run_benchmark():
    # sync def → FastAPI runs this in a threadpool, so the blocking ~10s
    # benchmark doesn't stall the event loop / other requests.
    if _face_recognizer is None:
        return {'error': 'no recognizer'}
    return _face_recognizer.benchmark()


@app.get("/snapshots/{filename}")
async def get_snapshot(filename: str):
    path = os.path.join(SNAPSHOTS_DIR, os.path.basename(filename))
    if os.path.exists(path):
        return FileResponse(path, media_type='image/jpeg')
    return JSONResponse({'error': 'not found'}, status_code=404)


@app.get("/face-snapshots/{filename}")
async def get_face_snapshot(filename: str):
    path = os.path.join(FACE_SNAPSHOTS_DIR, os.path.basename(filename))
    if os.path.exists(path):
        return FileResponse(path, media_type='image/jpeg')
    return JSONResponse({'error': 'not found'}, status_code=404)


def start(event_logger, face_recognizer, port=8099, blur_calibration=None):
    global _event_logger, _face_recognizer, _blur_calibration
    _event_logger = event_logger
    _face_recognizer = face_recognizer
    _blur_calibration = blur_calibration
    logging.info(f"Starting web server on port {port}")
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='warning')


def start_in_thread(event_logger, face_recognizer, port=8099, blur_calibration=None):
    t = threading.Thread(target=start,
                         args=(event_logger, face_recognizer, port, blur_calibration),
                         daemon=True)
    t.start()
    return t

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
.tab { display: none; padding: 20px 24px; max-width: 1100px; margin: 0 auto; }
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
.b-raw     { background: #1e1b4b; color: #a5b4fc; }
.b-ard     { background: #292524; color: #fbbf24; }
.ev-time   { font-size: 12px; color: var(--muted); }
.ev-detail { font-size: 13px; color: var(--muted); font-family: monospace; word-break: break-all; }
.ev-name   { font-size: 14px; font-weight: 600; }
.ev-sim    { font-size: 12px; color: var(--muted); }

/* Analytics */
.stats { display: flex; gap: 12px; margin-bottom: 20px; }
.stat {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 20px; flex: 1;
}
.stat .val { font-size: 30px; font-weight: 700; line-height: 1; }
.stat .lbl { font-size: 12px; color: var(--muted); margin-top: 5px; text-transform: uppercase; letter-spacing: .5px; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media(max-width:700px){ .charts { grid-template-columns: 1fr; } .stats { flex-direction: column; } }
.chart-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px;
}
.chart-card h3 { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 14px; }

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
  <div class="charts">
    <div class="chart-card"><h3>Bell rings by hour of day</h3><canvas id="hourChart"></canvas></div>
    <div class="chart-card"><h3>Bell rings by day of week</h3><canvas id="dowChart"></canvas></div>
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
let hChart=null, dChart=null;
const chartOpts = {
  responsive:true, plugins:{legend:{display:false}},
  scales:{
    x:{grid:{color:'#1e293b'},ticks:{color:'#64748b'}},
    y:{grid:{color:'#1e293b'},ticks:{color:'#64748b',stepSize:1},beginAtZero:true}
  }
};

document.querySelectorAll('nav button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    const tab = b.dataset.tab;
    document.getElementById(tab).classList.add('active');
    if(tab==='analytics') loadAnalytics();
    if(tab==='faces') loadFaces();
  });
});

function openLb(src){ document.getElementById('lb-img').src=src; document.getElementById('lb').classList.add('open'); }
function closeLb(){ document.getElementById('lb').classList.remove('open'); }

function fmt(iso){
  const d=new Date(iso);
  return d.toLocaleDateString()+' '+d.toLocaleTimeString();
}

const BADGE = {
  bell_ring:       ['b-bell',  '🔔 Bell Ring'],
  face_recognised: ['b-ok',    '✅ Recognised'],
  face_denied:     ['b-denied','🚫 Denied'],
  raw_command:     ['b-raw',   '📡 Serial'],
  arduino_connected:    ['b-ard','🔌 Connected'],
  arduino_disconnected: ['b-ard','⚠️ Disconnected'],
};
const ICON = {
  bell_ring:'🔔', face_recognised:'👤', face_denied:'❓',
  raw_command:'📡', arduino_connected:'🔌', arduino_disconnected:'⚠️'
};

function evHtml(e){
  const [bcls, blbl] = BADGE[e.type] || ['b-raw', e.type];
  let thumb = e.snapshot
    ? `<img class="ev-thumb" src="snapshots/${e.snapshot}" onclick="openLb('snapshots/${e.snapshot}')" onerror="this.parentNode.querySelector('.ev-icon').style.display='flex';this.style.display='none'" /><div class="ev-icon" style="display:none">${ICON[e.type]||'•'}</div>`
    : `<div class="ev-icon">${ICON[e.type]||'•'}</div>`;
  let detail = '';
  if(e.type==='face_recognised')
    detail = `<div class="ev-name">${e.person||''}</div><div class="ev-sim">${e.similarity?Math.round(e.similarity*100)+'% match':''}</div>`;
  else if(e.type==='face_denied')
    detail = `<div class="ev-detail">Best similarity: ${e.similarity!=null?Math.round(e.similarity*100)+'%':'no face detected'}</div>`;
  else if(e.raw_command)
    detail = `<div class="ev-detail">${e.raw_command}</div>`;
  else if(e.message)
    detail = `<div class="ev-detail">${e.message}</div>`;
  return `<div class="ev">${thumb}<div class="ev-body"><div class="ev-row"><span class="badge ${bcls}">${blbl}</span><span class="ev-time">${fmt(e.timestamp)}</span></div>${detail}</div></div>`;
}

async function loadEvents(){
  try{
    const r = await fetch('api/events');
    const events = await r.json();
    const el = document.getElementById('events-list');
    el.innerHTML = events.length ? events.map(evHtml).join('') : '<div class="empty">No events yet.</div>';
  }catch(err){ console.error(err); }
}

async function loadAnalytics(){
  try{
    const r = await fetch('api/analytics');
    const d = await r.json();
    const peakH = d.hour_counts ? d.hour_counts.indexOf(Math.max(...d.hour_counts)) : 0;
    const days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
    const peakD = d.dow_counts ? days[d.dow_counts.indexOf(Math.max(...d.dow_counts))] : '-';
    document.getElementById('stats').innerHTML = `
      <div class="stat"><div class="val">${d.total_events||0}</div><div class="lbl">Total Events</div></div>
      <div class="stat"><div class="val">${d.bell_rings||0}</div><div class="lbl">Bell Rings</div></div>
      <div class="stat"><div class="val">${peakH}:00</div><div class="lbl">Peak Hour</div></div>
      <div class="stat"><div class="val">${peakD}</div><div class="lbl">Busiest Day</div></div>
    `;
    if(hChart) hChart.destroy();
    hChart = new Chart(document.getElementById('hourChart'),{
      type:'bar',
      data:{labels:Array.from({length:24},(_,i)=>i+':00'),datasets:[{data:d.hour_counts||new Array(24).fill(0),backgroundColor:'#3b82f6',borderRadius:4}]},
      options:chartOpts
    });
    if(dChart) dChart.destroy();
    dChart = new Chart(document.getElementById('dowChart'),{
      type:'bar',
      data:{labels:days,datasets:[{data:d.dow_counts||new Array(7).fill(0),backgroundColor:'#8b5cf6',borderRadius:4}]},
      options:chartOpts
    });
  }catch(err){ console.error(err); }
}

async function loadFaces(){
  try{
    const r = await fetch('api/faces');
    const faces = await r.json();
    const el = document.getElementById('faces-grid');
    if(!faces.length){ el.innerHTML='<div class="empty">No faces enrolled.</div>'; return; }
    el.innerHTML = faces.map(f=>`
      <div class="face-card" id="fc-${CSS.escape(f.name)}">
        ${f.has_snapshot
          ? `<img class="face-photo" src="face-snapshots/${encodeURIComponent(f.name)}.jpg" onerror="this.nextElementSibling.style.display='flex';this.style.display='none'" /><div class="face-ph" style="display:none">👤</div>`
          : `<div class="face-ph">👤</div>`}
        <div class="face-body">
          <div class="face-name">${f.name}</div>
          <div class="face-meta">${f.embedding_count} sample${f.embedding_count!==1?'s':''}</div>
          <button class="btn-del" onclick="delFace('${f.name.replace(/'/g,"\\'")}')">Remove</button>
        </div>
      </div>`).join('');
  }catch(err){ console.error(err); }
}

async function delFace(name){
  if(!confirm('Remove '+name+' from face registry?')) return;
  const r = await fetch('api/faces/'+encodeURIComponent(name)+'/delete',{method:'POST'});
  const d = await r.json();
  if(d.success) document.getElementById('fc-'+CSS.escape(name))?.remove();
}

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
    hour_counts = [0] * 24
    dow_counts = [0] * 7
    bell_rings = 0
    for e in events:
        if e.get('type') == 'bell_ring':
            bell_rings += 1
            try:
                dt = datetime.fromisoformat(e['timestamp'])
                hour_counts[dt.hour] += 1
                dow_counts[dt.weekday()] += 1
            except Exception:
                pass
    return {
        'total_events': len(events),
        'bell_rings': bell_rings,
        'hour_counts': hour_counts,
        'dow_counts': dow_counts,
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


def start(event_logger, face_recognizer, port=8099):
    global _event_logger, _face_recognizer
    _event_logger = event_logger
    _face_recognizer = face_recognizer
    logging.info(f"Starting web server on port {port}")
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='warning')


def start_in_thread(event_logger, face_recognizer, port=8099):
    t = threading.Thread(target=start, args=(event_logger, face_recognizer, port), daemon=True)
    t.start()
    return t

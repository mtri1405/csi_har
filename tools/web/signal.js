/* Local canvas plots: no external chart libraries or internet connection. */
let signalData = null;
let monitorId = null;
let signalRequest = 0;
let monitorStarting = false;
let signalPolling = false;

function fillSignalOptions(id, options) {
  const element = document.getElementById(id);
  const old = element.value;
  const signature = JSON.stringify(options);
  if (element.dataset.options === signature) return;
  element.dataset.options = signature;
  element.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
  if (options.some(([value]) => value === old)) element.value = old;
}

function updateSignalControls(data) {
  fillSignalOptions('signalFile', (data.recordings || []).map(f => [f.path, `${f.label} / ${f.name}`]));
  fillSignalOptions('signalPort', (data.ports || []).map(p => [p.device, `${p.device} — ${p.description}`]));
  const running = Object.entries(data.jobs || {}).find(([,j]) => j.action === 'monitor' && j.status === 'running');
  if (running) monitorId = running[0];
  const button = document.getElementById('monitorButton');
  button.disabled = !!running || monitorStarting;
  button.textContent = button.disabled ? 'Đang quan sát…' : 'Bắt đầu quan sát';
}

function changeSignalSource() {
  const live = document.getElementById('signalSource').value === 'live';
  document.getElementById('csvControls').hidden = live;
  document.getElementById('liveControls').hidden = !live;
  signalRequest++;
  signalData = null;
  drawSignal();
  document.getElementById('signalStatus').textContent = live ? 'Bắt đầu quan sát hoặc chờ luồng đang mở.' : 'Chọn file và bấm Mở tín hiệu CSV.';
}

async function loadSignalFile() {
  const request = ++signalRequest;
  const file = document.getElementById('signalFile').value;
  if (!file) return;
  document.getElementById('signalStatus').textContent = 'Đang đọc CSV…';
  try {
    const data = await api('/api/signal?file=' + encodeURIComponent(file));
    if (request !== signalRequest) return;
    signalData = data;
    describeSignal(`CSV: ${file.split(/[\\/]/).pop()}`, data);
    drawSignal();
  } catch (error) {
    if (request === signalRequest) {
      signalData = null; drawSignal();
      document.getElementById('signalStatus').textContent = error.message;
    }
  }
}

async function startMonitor() {
  if (monitorStarting) return;
  monitorStarting = true;
  signalRequest++;
  document.getElementById('monitorButton').disabled = true;
  try {
    const result = await api('/api/run', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'monitor', port:document.getElementById('signalPort').value})});
    monitorId = result.job_id;
    active = monitorId;
    signalData = null;
    drawSignal();
    document.getElementById('signalStatus').textContent = 'Đang chờ CSI từ STA…';
  } catch (error) {
    document.getElementById('signalStatus').textContent = error.message;
  } finally {
    monitorStarting = false;
  }
}

async function stopMonitor() {
  if (!monitorId) return;
  try {
    await api('/api/stop', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({job_id:monitorId})});
    document.getElementById('signalStatus').textContent = 'Đã dừng quan sát.';
  } catch (error) { document.getElementById('signalStatus').textContent = error.message; }
}

function describeSignal(source, data) {
  const gap = data.max_gap_s > .25 ? ' | Có khoảng mất tín hiệu > 0,25 giây' : '';
  document.getElementById('signalStatus').textContent = `${source} | ${data.total || 0} mẫu hợp lệ | ${(data.duration_s || 0).toFixed(2)} giây | ${(data.rate_hz || 0).toFixed(1)} mẫu/giây | Hiển thị ${data.count} mẫu${data.rejected ? ` | Bỏ qua ${data.rejected} dòng lỗi` : ''}${gap}`;
}

async function pollSignal() {
  if (signalPolling || !monitorId || document.getElementById('signalSource').value !== 'live') return;
  signalPolling = true;
  const request = signalRequest;
  try {
    const data = await api('/api/signal?job=' + encodeURIComponent(monitorId));
    if (request !== signalRequest) return;
    signalData = data;
    describeSignal(data.status !== 'running' ? 'Đã dừng' : data.stale ? 'Chưa có tín hiệu / mất tín hiệu' : 'Trực tiếp (xem trước tối đa 10 mẫu/giây)', data);
    drawSignal();
  } catch (error) { document.getElementById('signalStatus').textContent = error.message; }
  finally { signalPolling = false; }
}

function prepareCanvas(id) {
  const canvas = document.getElementById(id);
  const width = Math.max(280, canvas.clientWidth);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio; canvas.height = 260 * ratio;
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  ctx.fillStyle = '#08101d'; ctx.fillRect(0,0,width,260);
  ctx.font = '12px system-ui'; ctx.fillStyle = '#9db0ca';
  return {ctx, width};
}

function drawSignal() {
  const chart = prepareCanvas('signalChart');
  const heat = prepareCanvas('signalHeatmap');
  const frames = signalData?.frames || [];
  if (!frames.length) {
    chart.ctx.fillText('Chưa có dữ liệu CSI',50,130);
    return;
  }
  const field = document.getElementById('signalScale').value;
  const carrier = Math.max(0,Math.min(106,Number(document.getElementById('signalCarrier').value) || 0));
  const selected = frames.map(f=>f[field][carrier]);
  const means = frames.map(f=>f[field].reduce((a,b)=>a+b,0)/107);
  let min = Infinity, max = -Infinity;
  for (const values of [selected,means]) for (const v of values) { min=Math.min(min,v); max=Math.max(max,v); }
  const padding = Math.max((max-min)*.1,.01); min-=padding; max+=padding;
  const t0=frames[0].t, duration=Math.max(frames.at(-1).t-t0,.001);
  const {ctx,width}=chart, left=55, top=20, height=200, plotWidth=width-left-15;
  const x=t=>left+(t-t0)/duration*plotWidth;
  const y=v=>top+height-(v-min)/(max-min)*height;
  ctx.strokeStyle='#263955';
  for(let i=0;i<=4;i++) {
    const v=min+(max-min)*i/4;
    ctx.beginPath();ctx.moveTo(left,y(v));ctx.lineTo(width-15,y(v));ctx.stroke();
    ctx.fillText(v.toFixed(2),2,y(v)+4);
    ctx.fillText((t0+duration*i/4).toFixed(1)+'s',left+plotWidth*i/4-10,245);
  }
  for(const [values,color] of [[selected,'#50b5ff'],[means,'#ffb454']]) {
    ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();
    frames.forEach((f,i)=> { if(i===0 || (f.gap_before ?? (f.t-frames[i-1].t>.25))) ctx.moveTo(x(f.t),y(values[i])); else ctx.lineTo(x(f.t),y(values[i])); });ctx.stroke();
  }
  let low=Infinity,high=-Infinity;
  for(const f of frames) for(const v of f[field]) {low=Math.min(low,v);high=Math.max(high,v);}
  const hctx=heat.ctx, hw=heat.width-70;
  for(let i=0;i<frames.length;i++) {
    const start=50+(frames[i].t-t0)/duration*hw;
    const next=i+1<frames.length?50+(frames[i+1].t-t0)/duration*hw:start+2;
    const cellWidth=i+1<frames.length && (frames[i+1].gap_before ?? (frames[i+1].t-frames[i].t>.25))?2:Math.max(1,next-start);
    frames[i][field].forEach((v,k)=> {
      const intensity=(v-low)/Math.max(high-low,1e-6);
      hctx.fillStyle=`hsl(${240-intensity*200} 85% ${15+intensity*55}%)`;
      hctx.fillRect(start,15+k*210/107,cellWidth,210/107+1);
    });
  }
  hctx.fillStyle='#9db0ca'; hctx.fillText('0',10,25);hctx.fillText('106',10,225);
  hctx.fillText(`${t0.toFixed(1)}s`,50,245);hctx.fillText(`${frames.at(-1).t.toFixed(1)}s`,heat.width-65,245);
  hctx.fillText(`${low.toFixed(2)} → ${high.toFixed(2)}`,heat.width/2-40,245);
}
window.addEventListener('resize',drawSignal);
setInterval(pollSignal,300);

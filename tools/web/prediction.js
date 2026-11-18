/* Pure view model is testable without a browser. No phase2 result survives empty/stale. */
function predictionView(job, now = Date.now()/1000) {
  const view = {message:'Chưa dự đoán', p1:'Chưa có dữ liệu', p2:'Đang khóa',
    detail:'Phase 2 chỉ chạy khi phase 1 kết luận PRESENT.', p1Detail:'',
    empty:null, present:null, standing:null, walking:null, active:false, final:'—'};
  if (!job || job.action !== 'predict') return view;
  if (job.status !== 'running') {
    view.message = job.status === 'failed' ? 'Dự đoán gặp lỗi — xem log' : 'Đã dừng dự đoán';
    return view;
  }
  const p = job.prediction;
  if (!p) { view.message='Đang mở Serial và nạp model…'; return view; }
  // Phone and PC clocks can differ. Prefer elapsed age computed by the server.
  const age = Number.isFinite(job.prediction_age_s) ? job.prediction_age_s
    : job.prediction_updated ? now-job.prediction_updated : 0;
  const stale = age > 4;
  if (p.status === 'no_signal' || stale) { view.message='Mất tín hiệu CSI — chờ dữ liệu mới'; return view; }
  if (p.status === 'countdown') { view.message=`Hiệu chuẩn bắt đầu sau ${p.remaining_s}s — hãy rời vùng cảm biến`; return view; }
  if (p.status === 'calibrating') {
    view.message=`Đang hiệu chuẩn EMPTY: ${(p.elapsed_s||0).toFixed(1)}/${p.target_s}s — giữ phòng trống`;
    return view;
  }
  if (p.status === 'calibration_complete') { view.message='Hiệu chuẩn xong — chờ cửa sổ CSI mới'; return view; }
  if (p.status === 'waiting_window') { view.message='Đang chờ đủ một cửa sổ CSI hợp lệ'; return view; }
  if (!p.phase1) { view.message='Model cũ: build dataset và train lại hai phase'; return view; }
  const phase1 = p.phase1;
  view.empty=phase1.probabilities.empty; view.present=phase1.probabilities.present;
  view.p1Detail=`Ngưỡng P(present): ${phase1.threshold.toFixed(2)} · Phiếu present: ${phase1.present_votes}/${phase1.vote_count} (mục tiêu ${phase1.vote_target}) · Cửa sổ hiện tại: ${phase1.raw_decision.toUpperCase()}`;
  if (p.status === 'stabilizing' || phase1.decision == null) {
    view.message='Đang tích lũy phiếu phase 1';view.p1='Chờ ổn định';return view;
  }
  view.p1=phase1.decision==='present' ? 'PRESENT — Có người' : 'EMPTY — Không có người';
  view.message=`Đang dự đoán · Baseline: ${p.baseline_source==='startup_empty_calibration' ? 'hiệu chuẩn phòng trống' : 'empty thuộc train'}`;
  if (phase1.decision !== 'present') {
    view.final='EMPTY — Không có người';view.detail='Phase 1 = EMPTY → bỏ qua model phase 2.';return view;
  }
  if (p.phase2?.status === 'active') {
    view.active=true;
    view.standing=p.phase2.probabilities.standing;
    view.walking=p.phase2.probabilities.walking;
    view.p2=p.phase2.decision==='walking' ? 'WALKING — Đi lại' : 'STANDING — Đứng yên';
    view.detail='Phase 1 = PRESENT → model motion phân biệt standing / walking.';
    view.final=`PRESENT → ${view.p2}`;
  }
  return view;
}

function renderPrediction(job) {
  const view=predictionView(job);
  const byId=id=>document.getElementById(id);
  byId('liveMessage').textContent=view.message;
  byId('phase1Decision').textContent=view.p1;
  byId('phase1Detail').textContent=view.p1Detail;
  byId('phase2Decision').textContent=view.p2;
  byId('phase2Detail').textContent=view.detail;
  byId('phase2Card').classList.toggle('blocked',!view.active);
  byId('prediction').textContent=view.final;
  for (const [key,name] of [['empty','Empty'],['present','Present'],['standing','Standing'],['walking','Walking']]) {
    const value=view[key];
    byId('prob'+name).textContent=value==null ? '—' : `${(100*value).toFixed(1)}%`;
    byId('bar'+name).value=value==null ? 0 : 100*value;
  }
}

if (typeof module !== 'undefined') module.exports = {predictionView};

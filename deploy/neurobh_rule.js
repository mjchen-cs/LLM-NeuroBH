/*
 * Custom ABR rule for dash.js v4.x that delegates every decision to deploy/abr_server.py.
 * NeuroBHRule.policy selects the algorithm on the server (neurobh, llm-neurobh, pensieve,
 * bba, bola, festive, dynamic, robustmpc).  See deploy/player.html for a ready-made page.
 *
 * Usage in the player page:
 *   <script src="dash.all.min.js"></script>
 *   <script src="neurobh_rule.js"></script>
 *   const player = dashjs.MediaPlayer().create();
 *   player.updateSettings({ streaming: { abr: { useDefaultABRRules: false } } });
 *   player.addABRCustomRule('qualitySwitchRules', 'NeuroBHRule', NeuroBHRule);
 *   NeuroBHRule.player = player;  NeuroBHRule.session = 'exp-' + Date.now();
 *   // optional: bind a natural-language service intent to this session
 *   fetch(NEUROBH_SERVER + '/intent', {method:'POST', body: JSON.stringify(
 *        {session: NeuroBHRule.session, text: 'Live shopping stream: latency under 8 s, keep at least 2 s buffered'})});
 *   player.initialize(video, url, true);
 *
 * Bandwidth samples (see deploy/media_server.py):
 *   1) server-side stable-connection sample of the segment just received:
 *      GET <media origin>/neurobh/bw?req=<X-NeuroBH-Req>          (NeuroBHRule.serverBw = true)
 *   2) else the response header X-NeuroBH-Bandwidth
 *   3) else the client-side chunk throughput (bytes / download time)
 * Baselines use 3) by default, as in an unmodified player (set NeuroBHRule.serverBw).
 * NeuroBHRule.info = contents of sizes.json (real per-segment sizes and VMAF), optional.
 */
var NEUROBH_SERVER = 'http://localhost:8333';

var NeuroBHRule;
function NeuroBHRuleClass() {
  const context = this.context;
  const factory = dashjs.FactoryMaker;
  const SwitchRequest = factory.getClassFactoryByName('SwitchRequest');
  const DashMetrics = factory.getSingletonFactoryByName('DashMetrics');
  let samples = [], dlTimes = [], levels = [], maxBuffer = 20, prevLevel = 0, chunk = 0;

  function header(r, name) {
    const m = (r._responseHeaders || '').match(new RegExp(name + ':\\s*([^\\r\\n]+)', 'i'));
    return m ? m[1].trim() : null;
  }

  function serverSample(reqId) {
    if (!reqId || !NeuroBHRule.mediaOrigin) return null;
    try {
      const x = new XMLHttpRequest();
      x.open('GET', NeuroBHRule.mediaOrigin + '/neurobh/bw?req=' + encodeURIComponent(reqId), false);
      x.send();
      const d = JSON.parse(x.responseText);
      return d && d.kbps ? d.kbps : null;
    } catch (e) { return null; }
  }

  let lastReq = null;
  function lastSample(dashMetrics, mediaType) {
    const reqs = dashMetrics.getHttpRequests(mediaType).filter(r => r.type === 'MediaSegment' && r._tfinish);
    if (!reqs.length) return null;
    const r = reqs[reqs.length - 1];
    if (r === lastReq) return null;                     // no new segment since the last decision
    lastReq = r;
    const dt = (r._tfinish.getTime() - r.trequest.getTime()) / 1000;
    const bytes = r.trace.reduce((a, t) => a + t.b[0], 0);
    const client = bytes * 8 / 1000 / Math.max(dt, 1e-3);
    const reqId = header(r, 'x-neurobh-req');
    let server = null;
    if (NeuroBHRule.serverBw) {
      server = serverSample(reqId);
      if (server === null) { const h = header(r, 'x-neurobh-bandwidth'); server = h ? parseFloat(h) : null; }
    }
    return { kbps: server !== null ? server : client, client: client, server: server, dt: dt, req: reqId,
             bytes: bytes };
  }

  function getMaxIndex(rulesContext) {
    const mediaType = rulesContext.getMediaType();
    if (mediaType !== 'video') return SwitchRequest(context).create();
    const dashMetrics = DashMetrics(context).getInstance();
    const mediaInfo = rulesContext.getMediaInfo();
    const bitrates = mediaInfo.bitrateList.map(b => b.bandwidth);       // bps
    const ls = lastSample(dashMetrics, mediaType);
    if (ls) { samples.push(ls.kbps); dlTimes.push(ls.dt); }
    const segDur = (NeuroBHRule.info && NeuroBHRule.info.seg_dur) || 4.0;
    const buffer = dashMetrics.getCurrentBufferLevel(mediaType) || 0;
    const video = NeuroBHRule.player ? NeuroBHRule.player.getVideoElement() : null;
    const t = video ? video.currentTime : 0;
    const info = NeuroBHRule.info;
    const nChunks = info ? info.n_chunks : 1000;
    const idx = Math.min(Math.round((t + buffer) / segDur), nChunks - 1);   // index of the next segment
    const nextSizes = info ? info.sizes.map(lv => lv[idx]) : bitrates.map(b => b / 8 * segDur);
    const nextVmaf = info && info.vmaf ? info.vmaf.map(lv => lv[idx]) : null;
    const body = {
      session: NeuroBHRule.session || 'default', policy: NeuroBHRule.policy || 'neurobh',
      network: NeuroBHRule.network || '', buffer: buffer,
      max_buffer: maxBuffer, prev_level: prevLevel,
      samples_kbps: samples.slice(-8), dl_times: dlTimes.slice(-8), levels: levels.slice(-8),
      next_sizes: nextSizes, next_vmaf: nextVmaf,
      remaining: Math.max(nChunks - idx, 1), chunk: idx, is_dynamic: false,
      // per-chunk measurement log (written to data/realworld_chunks.csv by abr_server.py)
      last: ls ? { req: ls.req, client_kbps: ls.client, server_kbps: ls.server, dl_time: ls.dt,
                   bytes: ls.bytes } : null
    };
    chunk++;
    const xhr = new XMLHttpRequest();                              // synchronous: rules are sync
    xhr.open('POST', NEUROBH_SERVER + '/decide', false);
    xhr.setRequestHeader('Content-Type', 'application/json');
    let level = prevLevel;
    try {
      xhr.send(JSON.stringify(body));
      const d = JSON.parse(xhr.responseText);
      level = d.level; maxBuffer = d.max_buffer;
      if (NeuroBHRule.player) {
        NeuroBHRule.player.updateSettings({ streaming: { buffer: {
          bufferTimeAtTopQuality: maxBuffer, bufferTimeAtTopQualityLongForm: maxBuffer, stableBufferTime: maxBuffer } } });
      }
    } catch (e) { console.warn('[NeuroBH] server unreachable, keep level', e); }
    prevLevel = level; levels.push(level);
    const sr = SwitchRequest(context).create();
    sr.quality = level; sr.reason = { rule: 'NeuroBH' }; sr.priority = SwitchRequest.PRIORITY.STRONG;
    return sr;
  }
  return { getMaxIndex: getMaxIndex };
}
NeuroBHRuleClass.__dashjs_factory_name = 'NeuroBHRule';
NeuroBHRule = dashjs.FactoryMaker.getClassFactory(NeuroBHRuleClass);

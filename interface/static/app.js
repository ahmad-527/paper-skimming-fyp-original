'use strict';
const $ = id => document.getElementById(id);
const roles = [
  ['BACKGROUND','Background','bg'], ['OBJECTIVE','Objective','obj'],
  ['METHODS','Methods','meth'], ['RESULTS','Results','res'], ['CONCLUSIONS','Conclusions','conc']
];
let sample, result, analysedText = '', grouped = false, filter = 'ALL', busy = false, modelState = 'loading';
const text = $('abstract');
const sentenceCount = count => `${count} ${count === 1 ? 'sentence' : 'sentences'}`;
let cloud = false, cloudOrigin = null, cloudRequest = null, lastCloudResponse = null, lastFrameHeight = 0;
let currentReadoutId = null, liveExports = null, recordedExports = null, preparedResult = null, localExportUrls = [];
function cloudMessage(type, value) {
  window.parent.postMessage({isStreamlitMessage:true,type,...value},cloudOrigin || '*');
}
function frameHeight() {
  if (!cloud) return;
  const height = document.body.scrollHeight + 8;
  if (height !== lastFrameHeight) {lastFrameHeight=height;cloudMessage('streamlit:setFrameHeight',{height});}
}
window.addEventListener('message', event => {
  if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
  cloud = true; cloudOrigin = event.origin;
  const args = event.data.args || {};
  if (args.status) showStatus(args.status.state);
  if (args.thesis_url) document.querySelectorAll('a[href="/report.pdf"]').forEach(link => {link.href=args.thesis_url;});
  liveExports = args.live_exports || null; recordedExports = args.recorded_exports || null;
  const response = args.response;
  if (cloudRequest && response?.request_id === cloudRequest.id) {
    const pending = cloudRequest; cloudRequest = null; clearTimeout(pending.timer);
    lastCloudResponse = response.request_id;
    if (response.data) currentReadoutId = response.request_id;
    response.error ? pending.reject(Error(response.error)) : pending.resolve(response.data);
  } else if (!cloudRequest && !result && response?.request_id && response.request_id !== lastCloudResponse) {
    // Streamlit can recreate the iframe during a full page run. Restore its
    // session response so a completed request survives that browser remount.
    lastCloudResponse = response.request_id;
    if (response.data) {
      currentReadoutId = response.request_id;
      result = response.data; text.value = result.text; analysedText = result.text;
      grouped = false; filter = 'ALL'; render(); updateInput();
      announce(`${sentenceCount(result.sentences.length)} classified.`);
    } else if (response.error) {
      text.value = response.text || ''; updateInput(); setError(response.error);
    }
  }
  updateExportLinks();
  frameHeight();
});
if (window.parent !== window) cloudMessage('streamlit:componentReady',{apiVersion:1});
new ResizeObserver(frameHeight).observe(document.body);
function node(tag, className, value) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (value !== undefined) element.textContent = value;
  return element;
}
function announce(message) { $('announcement').textContent = message; }
function setError(message) { $('input-error').textContent = message; $('input-error').classList.toggle('hidden', !message); }
function updateInput() {
  $('character-count').textContent = `${text.value.length.toLocaleString()} / 20,000 characters`;
  $('stale').classList.toggle('hidden', !result || text.value.trim() === analysedText);
}
text.addEventListener('input', updateInput);
text.addEventListener('paste', event => {
  const incoming = event.clipboardData?.getData('text/plain');
  if (incoming === undefined) return;
  const remaining = text.value.length - (text.selectionEnd - text.selectionStart);
  if (remaining + incoming.length > 20_000) {
    event.preventDefault();
    setError('Use an abstract of 20,000 characters or fewer. The oversized paste was not inserted.');
    announce('Paste exceeds the character limit. Existing text was kept.');
  }
});
$('clear-button').addEventListener('click', () => { text.value = ''; updateInput(); setError(''); text.focus(); });
async function getSample() {
  if (!sample) { const response = await fetch('./sample.json'); if (!response.ok) throw Error('The example could not be loaded.'); sample = await response.json(); }
  return sample;
}
$('example-button').addEventListener('click', async () => {
  try { text.value = (await getSample()).text; updateInput(); setError(''); text.focus(); announce('Example abstract loaded.'); } catch(e) { setError(e.message); }
});
$('recorded-button').addEventListener('click', async () => {
  if (busy) return;
  try { result = structuredClone(await getSample()); text.value = result.text; analysedText = result.text; grouped = false; filter = 'ALL'; updateInput(); setError(''); render(); announce('Recorded example loaded: eight sentences.'); } catch(e) { setError(e.message); }
});
async function status() {
  if (cloud) return;
  try {
    const response = await fetch('/api/status'); if (!response.ok) throw Error();
    const data = await response.json(); if (!cloud) showStatus(data.state);
  } catch(e) { if (!cloud) showStatus('offline'); }
  if (modelState !== 'ready') setTimeout(status, 6000);
}
function showStatus(state) {
  modelState = state;
  const labels = {ready:'Historical model ready',loading:'Historical model loading',error:'Model setup needs attention',unconfigured:'Recorded example available',offline:'Model connection unavailable'};
  $('model-state').className = `status ${modelState}`;
  $('model-state').lastElementChild.textContent = labels[modelState] || 'Model unavailable';
}
setTimeout(status,300);
async function classify(submitted) {
  if (cloud) return new Promise((resolve,reject)=>{
    const id = crypto.randomUUID();
    const timer = setTimeout(()=>{cloudRequest=null;reject(Error('The model took too long. Please try again.'));},180000);
    cloudRequest = {id,resolve,reject,timer};
    cloudMessage('streamlit:setComponentValue',{dataType:'json',value:{request_id:id,text:submitted}});
  });
  const response = await fetch('/api/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:submitted}),signal:AbortSignal.timeout(180000)});
  const data = await response.json(); if (!response.ok) throw Error(data.error || 'The abstract could not be classified.');
  return data;
}
$('analyse-button').addEventListener('click', async () => {
  if (busy) return;
  if (!text.value.trim()) { setError('Paste an abstract or load an example first.'); text.focus(); return; }
  if (text.value.length > 20_000) { setError('Use an abstract of 20,000 characters or fewer.'); text.focus(); return; }
  if (modelState !== 'ready') { setError('The historical model is not ready yet. Explore the recorded example, or try again after the status changes.'); status(); return; }
  const submitted = text.value.trim(); busy = true; setError('');
  $('analyse-button').disabled = true; $('recorded-button').disabled = true;
  $('analyse-button').firstElementChild.textContent = 'Reading your abstract…';
  $('empty-state').classList.add('hidden'); $('results').classList.add('hidden'); $('loading-state').classList.remove('hidden');
  try {
    const data = await classify(submitted);
    result = data; analysedText = submitted; grouped = false; filter = 'ALL'; render(); updateInput(); announce(`${sentenceCount(result.sentences.length)} classified.`);
  } catch(e) { setError(e.name === 'TimeoutError' ? 'The model took too long. Please try again.' : e.message === 'Failed to fetch' ? 'Connection lost. Check the model host and try again.' : e.message); if (result) render(); else $('empty-state').classList.remove('hidden'); }
  finally { busy = false; $('analyse-button').disabled = false; $('recorded-button').disabled = false; $('analyse-button').firstElementChild.textContent = 'Skim this abstract'; $('loading-state').classList.add('hidden'); }
});
function render() {
  $('empty-state').classList.add('hidden'); $('results').classList.remove('hidden');
  $('sentence-count').textContent = sentenceCount(result.sentences.length);
  $('result-source').textContent = result.mode === 'recorded' ? 'Recorded 20k notebook example · preserved predictions' : 'Live · historical December 2024 checkpoint';
  $('position-note').classList.toggle('hidden', !result.position_note);
  $('order-view').setAttribute('aria-pressed', String(!grouped)); $('group-view').setAttribute('aria-pressed', String(grouped));
  const filters = $('role-filters'); filters.replaceChildren();
  for (const [label, name, cls] of [['ALL','All','all'], ...roles]) {
    const count = label === 'ALL' ? result.sentences.length : result.sentences.filter(s => s.label === label).length;
    const button = node('button',cls,name); button.dataset.role = label; button.append(node('span','',String(count))); button.setAttribute('aria-pressed',String(filter === label));
    button.addEventListener('click',()=>{ filter = label; render(); $('role-filters').querySelector(`[data-role="${label}"]`).focus(); }); filters.append(button);
  }
  const list = $('sentence-list'); list.replaceChildren();
  const visible = result.sentences.filter(s => filter === 'ALL' || s.label === filter);
  if (!visible.length) list.append(node('p','no-match','No sentences received this role. Try another filter.'));
  if (grouped) {
    for (const [label,name,cls] of roles) {
      const rows = visible.filter(s=>s.label===label); if (!rows.length) continue;
      const heading = node('h4','group-label'); heading.append(node('span',`role ${cls}`,name), node('span','',`${rows.length} ${rows.length===1?'sentence':'sentences'}`)); list.append(heading);
      rows.forEach(row=>list.append(sentenceCard(row)));
    }
  } else visible.forEach(row=>list.append(sentenceCard(row)));
  updateExportLinks();
}
function sentenceCard(row) {
  const [label,name,cls] = roles.find(r=>r[0] === row.label);
  const card = node('article','sentence-card'), button = node('button','sentence-button');
  button.setAttribute('aria-expanded','false'); button.setAttribute('aria-controls',`scores-${row.index}`);
  button.append(node('span','line-number',String(row.index+1).padStart(2,'0')));
  const main = node('span','sentence-main'), head = node('span','sentence-head');
  head.append(node('span',`role ${cls}`,name),node('span','score',`${(row.scores[label]*100).toFixed(1)}%`),node('span','expand','+'));
  main.append(head,node('span','sentence-text',row.text)); button.append(main);
  const details = node('div','sentence-detail hidden'); details.id = `scores-${row.index}`;
  details.append(node('p','detail-caption','Model scores across all five roles'));
  for (const [key,title] of roles) {
    const line = node('div','score-row');
    // Native progress avoids inline styles under the content security policy.
    const meter = document.createElement('progress'); meter.max = 1; meter.value = row.scores[key]; meter.setAttribute('aria-label',`${title} model score`);
    line.append(node('span','',title),meter,node('span','score-value',`${(row.scores[key]*100).toFixed(1)}%`)); details.append(line);
  }
  button.addEventListener('click',()=>{ const expanded = button.getAttribute('aria-expanded') === 'true'; button.setAttribute('aria-expanded',String(!expanded)); details.classList.toggle('hidden',expanded); head.lastElementChild.textContent = expanded ? '+' : '−'; });
  card.append(button,details); return card;
}
$('order-view').addEventListener('click',()=>{grouped=false;render();});
$('group-view').addEventListener('click',()=>{grouped=true;render();});
function openDialog(id) { if (id === 'export-dialog') $('copy-status').textContent = ''; $(id).showModal(); }
$('about-button').addEventListener('click',()=>openDialog('research-dialog'));
$('close-research').addEventListener('click',()=>$('research-dialog').close());
$('export-button').addEventListener('click',()=>openDialog('export-dialog'));
$('close-export').addEventListener('click',()=>$('export-dialog').close());
for (const id of ['research-dialog','export-dialog']) $(id).addEventListener('click',event=>{if(event.target === $(id)){const r=$(id).getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom) $(id).close();}});
function markdown() {
  return `# Paper Skimming readout\n\nSource: ${result.mode === 'recorded' ? 'Preserved notebook example' : result.model}\nInterface added: 9 October 2026, after original project completion.\nIntended domain: Biomedical randomized controlled trial abstracts.\nScores are not calibrated certainty.\n\n` + result.sentences.map(s=>`## ${s.index+1}. ${s.label}\n\n${s.text}\n\nModel score: ${(s.scores[s.label]*100).toFixed(1)}%\n`).join('\n');
}
function updateExportLinks() {
  if (!result) return;
  let links;
  if (cloud) {
    const exports = result.mode === 'recorded' ? recordedExports : liveExports?.request_id === currentReadoutId ? liveExports : null;
    const base = new URLSearchParams(window.location.search).get('streamlitUrl');
    links = {};
    for (const extension of ['json','md']) {
      const path = exports?.[extension];
      if (base && /^\/media\/[a-f0-9]+\.[a-z0-9]+$/.test(path || '')) {
        const url = new URL(path.slice(1), base);
        if (url.origin === cloudOrigin) links[extension] = url.href;
      }
    }
  } else {
    if (preparedResult !== result) {
      localExportUrls.forEach(url=>URL.revokeObjectURL(url));
      const json = JSON.stringify({...result,interface_added:'9 October 2026',scores_are_calibrated:false,intended_domain:'Biomedical randomized controlled trial abstracts'},null,2);
      localExportUrls = [URL.createObjectURL(new Blob([json],{type:'application/json'})),URL.createObjectURL(new Blob([markdown()],{type:'text/markdown'}))];
      preparedResult = result;
    }
    links = {json:localExportUrls[0],md:localExportUrls[1]};
  }
  for (const extension of ['json','md']) {
    const link = $(extension === 'json' ? 'download-json' : 'download-markdown');
    if (links[extension]) {link.href=links[extension];link.removeAttribute('aria-disabled');}
    else {link.removeAttribute('href');link.setAttribute('aria-disabled','true');}
  }
}
for (const id of ['download-json','download-markdown']) $(id).addEventListener('click',event=>{
  if (!$(id).hasAttribute('href')) {event.preventDefault();$('copy-status').textContent='Preparing downloads. Try again in a moment, or copy the readout.';}
  else announce('Export download requested.');
});
$('copy-result').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(markdown());$('copy-status').textContent='Copied to clipboard.';}catch(e){$('copy-status').textContent='Clipboard unavailable. Download the Markdown readout instead.';}});

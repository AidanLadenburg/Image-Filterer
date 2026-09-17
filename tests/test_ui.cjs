// Browser logic regression tests; no browser binary or npm dependencies needed.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('image_filterer/templates/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function browser(windowOptions = {}) {
  const elements = new Map();
  function element(selector) {
    if (!elements.has(selector)) elements.set(selector, {
      innerHTML: '', textContent: '', style: {}, dataset: {},
      classList: {add(){},remove(){},toggle(){}}, removeAttribute(){},
      querySelector: () => element(selector + '/child'),
    });
    return elements.get(selector);
  }
  const ctx = vm.createContext({console, Blob, Uint8Array, AbortController, URLSearchParams,
    window: {isSecureContext: true, ...windowOptions},
    document: {querySelector: element,querySelectorAll:()=>[],addEventListener(){}},
    localStorage: {setItem(){}}, setTimeout, clearTimeout,
  });
  vm.runInContext(script, ctx);
  vm.runInContext('updateBar = () => {}; renderTiles = rows => { globalThis.rendered = rows; };', ctx);
  return {ctx, elements, run: code => vm.runInContext(code, ctx)};
}

function directory() {
  const files = new Map();
  return {files, async getFileHandle(name, options) {
    if (!files.has(name) && !options?.create) throw Object.assign(new Error('missing'), {name:'NotFoundError'});
    return {getFile: async () => files.get(name), createWritable: async () => ({
      write: async blob => files.set(name,blob), close:async()=>{},
    })};
  }};
}

test('equal-size different photos export separately; identical exports skip', async () => {
  const {ctx} = browser(); const dir = directory();
  assert.equal(await ctx.writeFileToDir(dir,'photo.jpg',new Blob(['AAAA'])), 'written');
  assert.equal(await ctx.writeFileToDir(dir,'photo.jpg',new Blob(['BBBB'])), 'written');
  assert.equal(await dir.files.get('photo.jpg').text(),'AAAA');
  assert.equal(await dir.files.get('photo-2.jpg').text(),'BBBB');
  assert.equal(await ctx.writeFileToDir(dir,'photo.jpg',new Blob(['BBBB'])), 'skipped');
  assert.equal(dir.files.size,2);
});

test('permission failure never becomes a new export overwrite', async () => {
  const {ctx} = browser();
  await assert.rejects(ctx.writeFileToDir({getFileHandle:async()=>{
    throw Object.assign(new Error('denied'),{name:'NotAllowedError'});
  }},'photo.jpg',new Blob(['test'])), /denied/);
});

test('only latest grid request may update the visible shoot', async () => {
  const {ctx,run} = browser(); const pending=[];
  ctx.fetch = (url,options) => new Promise(resolve=>pending.push({resolve,signal:options.signal}));
  run('state.hasRun=true; state.runId=1');
  const first=ctx.loadPage(true);
  run('state.runId=2');
  const second=ctx.loadPage(true);
  assert.equal(pending.length,2);
  assert.equal(pending[0].signal.aborted,true);
  pending[1].resolve({ok:true,json:async()=>({bursts:[{run:2}],version:4,page:{total_matching:1}})});
  await second;
  pending[0].resolve({ok:true,json:async()=>({bursts:[{run:1}],version:2,page:{total_matching:1}})});
  await first;
  assert.equal(ctx.rendered[0].run,2);
  assert.equal(run('state.version'),4);
  assert.equal(run('state.loading'),false);
});

test('late star response cannot replace another shoot picks', async () => {
  const {ctx,run}=browser();let resolve;
  run('state.runId=1');
  ctx.fetch=()=>new Promise(r=>resolve=r);
  const request=ctx.refreshStars();
  run('state.runId=2; state.starsMine=new Set(["new-shoot"])');
  resolve({json:async()=>({mine:['old-shoot']})});await request;
  assert.equal(run('state.starsMine.has("new-shoot")'),true);
});

test('others export includes a frame starred by both me and another viewer', () => {
  const {ctx,run}=browser();
  run('state.starOwner="others"; state.starsMine=new Set(["shared"]); state.starsOthers=new Set(["shared"])');
  assert.deepEqual(Array.from(ctx.starredPathsForOwner()),['shared']);
});

test('old modal request is versioned and cannot replace the latest modal', async () => {
  const {ctx,run}=browser();const pending=[];
  ctx.fetch=url=>new Promise(resolve=>pending.push({url,resolve}));
  run('state.runId=1;state.version=2;renderStrip=burst=>{globalThis.displayed=burst.burst_id;};updateNav=()=>{}');
  const first=ctx.openModal({burst_id:0,burst_rank:1,burst_size:1});
  const second=ctx.openModal({burst_id:1,burst_rank:2,burst_size:1});
  assert.match(pending[0].url,/version=2/);
  pending[1].resolve({ok:true,json:async()=>({frames:[]})});await second;
  pending[0].resolve({ok:true,json:async()=>({frames:[]})});await first;
  assert.equal(ctx.displayed,1);
});

test('first published watch generation loads without a manual refresh', async () => {
  const {ctx,run}=browser();
  run('state.hasRun=true;state.runId=1;state.version=null;loadPage=async()=>{globalThis.loaded=true;}');
  ctx.fetch=async()=>({json:async()=>({version:2,n_frames:1})});
  await ctx.pollVersion();
  assert.equal(ctx.loaded,true);
});

test('a stale burst closes the modal and offers reload', async () => {
  const {ctx,run}=browser();
  run('state.runId=1;state.version=2;updateReloadBar=()=>{globalThis.reloadOffered=true;}');
  ctx.fetch=async()=>({ok:false,json:async()=>({stale:true,version:4,frames:[]})});
  await ctx.openModal({burst_id:0,burst_rank:1,burst_size:1});
  assert.equal(run('state.openBurst'),null);
  assert.equal(run('state.pending.version'),4);
  assert.equal(ctx.reloadOffered,true);
});

function exportBrowser(options) {
  const b = browser(options);
  b.ctx.setTimeout = () => 0;
  return b;
}

test('individual downloads honor the selected local destination', async () => {
  const {ctx,run} = exportBrowser({showDirectoryPicker(){}});
  const dir = directory(); dir.name = 'Chosen folder';
  dir.queryPermission = async () => 'granted';
  ctx.dir = dir;
  run('exportTarget="local"; exportDirHandle=dir; state.runId=7');
  let url;
  ctx.fetch = async u => { url=u; return {ok:true,blob:async()=>new Blob(['original'])}; };
  await ctx.downloadOne('/photos/frame.raw', null);
  assert.equal(await dir.files.get('frame.raw').text(), 'original');
  assert.match(url, /run_id=7/);
});

test('local folder picker cancellation does not write or change destinations', async () => {
  const {ctx,run}=exportBrowser({showDirectoryPicker:async()=>{throw Object.assign(new Error('cancel'),{name:'AbortError'});}});
  run('exportTarget="local"');
  ctx.fetch=async()=>assert.fail('cancelled export must not fetch');
  await ctx.downloadOne('/photo.jpg',null);
  assert.equal(run('exportTarget'),'local');
  assert.equal(run('exportBusy'),false);
});

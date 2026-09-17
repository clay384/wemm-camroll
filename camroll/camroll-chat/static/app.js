'use strict';

const FIXED_ALBUM_ID = 'eval-20260916-141627-camroll-e5a5fe2f';
const FIXED_INDEX_VERSION = 'bd64fa387f0c5d507ef845fb2493eed6';
const TOOL_NAMES = {
  search: '语义检索',
  grep: '关键词筛选',
  list_by_date: '按日期查询',
  get: '读取照片资料',
  view_image: 'Qwen 图片复核',
  query_photo_people: '查询照片人物',
  final_image_review: 'Qwen 最终图片复核',
  agent: 'Agent 执行',
  protocol: '协议校验'
};

const state = {
  busy: false,
  interactive: false,
  reloadRequired: false,
  albumId: FIXED_ALBUM_ID,
  album: null,
  currentCard: null,
  traces: new Map(),
  toolKeys: new Map(),
  evidenceByCard: new WeakMap(),
  traceCount: 0
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const text = value => value == null ? '' : String(value);
const isObject = value => value && typeof value === 'object' && !Array.isArray(value);

function setServiceState(kind, label) {
  const dot = $('.state-dot', $('#service-state'));
  dot.className = `state-dot state-${kind}`;
  $('#service-state-text').textContent = label;
}

function showNotice(message) {
  const notice = $('#global-notice');
  notice.textContent = message;
  notice.hidden = !message;
}

function truncate(value, length = 120) {
  const source = text(value).replace(/\s+/g, ' ').trim();
  return source.length > length ? `${source.slice(0, length)}…` : source;
}

function formatDuration(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(value < 10000 ? 2 : 1)} s`;
}

function formatNumber(value, digits = 0) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return value.toLocaleString('zh-CN', { maximumFractionDigits: digits });
}

function formatCost(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return `¥${value.toLocaleString('zh-CN', { minimumFractionDigits: 4, maximumFractionDigits: 6 })}`;
}

function parseMaybeJson(value) {
  if (typeof value !== 'string') return value;
  const trimmed = value.trim();
  if (!(trimmed.startsWith('{') || trimmed.startsWith('['))) return value;
  try { return JSON.parse(trimmed); } catch { return value; }
}

function parseToolObservation(value) {
  const parsed = parseMaybeJson(value);
  if (typeof parsed !== 'string') return parsed;
  const jsonStart = parsed.indexOf('{');
  if (jsonStart < 0) return parsed;
  try { return JSON.parse(parsed.slice(jsonStart)); } catch { return parsed; }
}

function publicText(value) {
  return text(value)
    .replace(/\/(?:home|Users|var|tmp|etc|opt|srv)\/[^\s"'<>]+/g, '[服务器路径已隐藏]')
    .replace(/(?:Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, 'Bearer [已隐藏]');
}

function sanitizeValue(value, depth = 0) {
  if (depth > 6) return '[内容已折叠]';
  if (typeof value === 'string') {
    if (value.startsWith('data:image/')) return '[图片数据已隐藏]';
    if (/^\/(?:home|Users|var|tmp|etc|opt|srv)\//.test(value)) return '[服务器路径已隐藏]';
    return value.length > 4000 ? `${value.slice(0, 4000)}…` : value;
  }
  if (Array.isArray(value)) return value.slice(0, 40).map(item => sanitizeValue(item, depth + 1));
  if (!isObject(value)) return value;
  const result = {};
  const blocked = /^(?:api_?key|authorization|credential|secret|system_?prompt|prompt|messages?|path|.*_path|.*_root|directory|image_url)$/i;
  Object.entries(value).slice(0, 80).forEach(([key, item]) => {
    if (blocked.test(key)) result[key] = '[已隐藏]';
    else result[key] = sanitizeValue(item, depth + 1);
  });
  return result;
}

function safeJson(value) {
  try { return JSON.stringify(sanitizeValue(parseMaybeJson(value)), null, 2); }
  catch { return '[无法展示的结果]'; }
}

function pickFirst(source, keys) {
  for (const key of keys) {
    if (source?.[key] !== undefined && source[key] !== null && source[key] !== '') return source[key];
  }
  return undefined;
}

function extractAlbum(payload) {
  const albums = Array.isArray(payload) ? payload : (payload?.albums || payload?.items || []);
  return albums.find(item => item?.id === FIXED_ALBUM_ID || item?.album_id === FIXED_ALBUM_ID) || albums[0] || null;
}

async function loadStatus() {
  setServiceState('loading', '正在检查 8773');
  showNotice('');
  try {
    const [healthResponse, albumsResponse] = await Promise.all([
      fetch('/api/health', { cache: 'no-store', credentials: 'same-origin' }),
      fetch('/api/albums', { cache: 'no-store', credentials: 'same-origin' })
    ]);
    if (!healthResponse.ok) throw new Error(`健康检查失败（HTTP ${healthResponse.status}）`);
    const health = await healthResponse.json();
    const albumsPayload = albumsResponse.ok ? await albumsResponse.json() : null;
    const album = extractAlbum(albumsPayload) || health.album || null;
    state.album = album;
    state.albumId = album?.id || album?.album_id || FIXED_ALBUM_ID;

    const indexStatus = album?.index_status || health.album?.index_status || 'unknown';
    if (indexStatus !== 'ready') {
      const labels = {
        index_changed: '冻结索引版本已变化',
        integrity_changed: '冻结索引完整性校验失败',
        unavailable: '冻结相册不可用'
      };
      throw new Error(labels[indexStatus] || '冻结相册状态异常');
    }
    const upstreamReady = health.upstream === 'ready' || health.upstream?.status === 'ready' || health.upstream_status === 'ready';
    if (!upstreamReady) throw new Error('CamRoll 8772 上游当前不可用');

    if (health.busy) {
      setServiceState('loading', '已有问题正在执行');
      showNotice('当前已有一个问题正在执行。此页面不会排队或自动重试；请稍后刷新页面。');
      setInteractive(false);
      return;
    }

    const indexVersion = album?.index_version || health.index_version || FIXED_INDEX_VERSION;
    $('#snapshot-version').textContent = indexVersion;
    $('#album-photo-count').textContent = `${album?.photo_count ?? 355} 张`;
    $('#album-person-count').textContent = `${album?.identity_count ?? album?.person_count ?? album?.reference_count ?? 8} 人`;
    $('#album-relation-count').textContent = `${album?.relationship_count ?? album?.relation_count ?? 6} 条`;
    $('#index-state').textContent = indexVersion === FIXED_INDEX_VERSION ? '就绪' : '版本异常';
    $('#index-state').classList.toggle('mini-state-error', indexVersion !== FIXED_INDEX_VERSION);
    if (indexVersion !== FIXED_INDEX_VERSION) {
      setServiceState('error', '冻结索引版本不一致');
      showNotice('冻结索引版本与页面预期不一致。为避免混用快照，当前不能提交问题。');
      setInteractive(false);
      return;
    }
    setServiceState('ready', '8773 与 CamRoll 已就绪');
    setInteractive(true);
  } catch (error) {
    setServiceState('error', '服务不可用');
    showNotice(`${error.message}。页面不会自动重试；服务恢复后请刷新页面。`);
    $('#index-state').textContent = '不可用';
    setInteractive(false);
  }
}

function setInteractive(enabled) {
  state.interactive = Boolean(enabled);
  updateControls();
}

function setBusy(busy) {
  state.busy = busy;
  updateControls();
  $('#send-button span:first-child').textContent = busy ? '执行中' : '发送问题';
}

function updateControls() {
  const disabled = !state.interactive || state.busy || state.reloadRequired;
  $('#question-input').disabled = disabled;
  $('#send-button').disabled = disabled;
  $$('.suggestion').forEach(button => { button.disabled = disabled; });
}

function requireReload(message) {
  state.reloadRequired = true;
  state.interactive = false;
  setServiceState('error', '冻结索引已变化');
  showNotice(`${publicText(message)} 当前页面已锁定，请刷新后重新核验冻结索引。`);
  updateControls();
}

function autoGrow() {
  const input = $('#question-input');
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function createConversation(question) {
  $('#welcome-card')?.remove();
  const fragment = $('#conversation-template').content.cloneNode(true);
  const card = $('.conversation-card', fragment);
  $('.user-question', card).textContent = question;
  state.evidenceByCard.set(card, new Map());
  $('#chat-history').append(card);
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });
  return card;
}

function resetTrace() {
  state.traces.clear();
  state.toolKeys.clear();
  state.traceCount = 0;
  $('#trace-timeline').replaceChildren();
  $('#trace-counter').textContent = '执行中';
}

function toolBase(payload) {
  return `tool:${payload.step ?? ''}:${payload.tool ?? ''}`;
}

function traceKey(type, payload) {
  if (type === 'tool_call') {
    const base = toolBase(payload);
    const keys = state.toolKeys.get(base) || [];
    const key = `${base}:${keys.length}`;
    keys.push(key);
    state.toolKeys.set(base, keys);
    return key;
  }
  if (type === 'tool_result') {
    const base = toolBase(payload);
    const keys = state.toolKeys.get(base) || [];
    const waiting = [...keys].reverse().find(key => state.traces.get(key)?.classList.contains('is-running'));
    if (waiting) return waiting;
    const key = `${base}:${keys.length}`;
    keys.push(key);
    state.toolKeys.set(base, keys);
    return key;
  }
  return `${type}:${state.traceCount}`;
}

function iconFor(type, status) {
  if (status === 'error') return '!';
  if (status === 'partial') return '≈';
  if (status === 'success') return '✓';
  if (type === 'tool_call' || type === 'tool_result') return 'T';
  if (type === 'model_status') return 'Q';
  if (type === 'answer') return 'A';
  return '•';
}

function traceTitle(type, payload) {
  if (type === 'request_started') return '请求已接收';
  if (type === 'model_status') return publicText(payload.label || payload.message || 'Qwen3.7-Flash 处理中');
  if (type === 'tool_call' || type === 'tool_result') return TOOL_NAMES[payload.tool] || `工具 · ${payload.tool || '未知'}`;
  if (type === 'answer' && payload.status === 'error') return '回答未完整完成';
  if (type === 'answer' && payload.status === 'partial') return '已返回证据有限的回答';
  if (type === 'answer') return '答案与证据已生成';
  if (type === 'error') return '请求执行失败';
  return payload.message || type;
}

function traceSummary(type, payload) {
  if (type === 'request_started') return `冻结索引 · M4${payload.request_id ? ` · ${truncate(payload.request_id, 22)}` : ''}`;
  if (type === 'model_status') return truncate(publicText(payload.summary || payload.detail || '等待模型返回可审计的下一步操作。'));
  if (type === 'tool_call') return summarizeToolArgs(payload.tool, payload.args || payload.arguments);
  if (type === 'tool_result') return summarizeToolResult(payload.tool, payload.observation ?? payload.result);
  if (type === 'answer' && payload.status === 'error') return `执行终止：${publicText(payload.stopped_reason || '未记录原因')}。已保留现有答案和调用指标。`;
  if (type === 'answer' && payload.status === 'partial') return '证据不足，已保留能够确认的回答和调用指标。';
  if (type === 'answer') return '已完成本次单轮问答。';
  if (type === 'error') return truncate(publicText(payload.message || payload.error || '上游返回错误。'));
  return truncate(payload.message || '');
}

function summarizeToolArgs(tool, raw) {
  const args = parseMaybeJson(raw);
  if (!isObject(args)) return truncate(args || '已提交工具参数');
  const imageIds = Array.isArray(args.image_ids) ? args.image_ids : (Array.isArray(args.ids) ? args.ids : []);
  if (tool === 'search') return `查询：${truncate(args.query || args.text || '相册语义', 66)}`;
  if (tool === 'list_by_date') return `日期范围：${truncate(args.date || args.start || args.date_from || '已指定', 60)}`;
  if (tool === 'get') return `读取：${truncate(args.id || args.image_id || imageIds.join('、') || '证据记录', 55)}`;
  if (tool === 'view_image') return `核验：${truncate(args.id || args.image_id || imageIds.join('、') || '已指定图片', 55)}`;
  if (tool === 'query_photo_people') return `人物查询：${truncate(args.id || args.image_id || imageIds.join('、') || args.person || '已指定条件', 55)}`;
  const safeArgs = sanitizeValue(args);
  return truncate(Object.entries(safeArgs).map(([key, value]) => `${key}=${formatMetadataValue(value)}`).join(' · '), 95);
}

function countResultRecords(value) {
  if (Array.isArray(value)) return value.length;
  if (!isObject(value)) return null;
  for (const key of ['results', 'images', 'records', 'items']) if (Array.isArray(value[key])) return value[key].length;
  return null;
}

function summarizeToolResult(tool, raw) {
  const result = parseToolObservation(raw);
  if (isObject(result) && result.error) return `工具未完成：${truncate(formatMetadataValue(result.error.reason || result.error.code || result.error), 65)}`;
  if (isObject(result) && result.status === 'error') {
    const error = result.error || result.data?.error || result.reason || result.code;
    return `工具未完成：${truncate(formatMetadataValue(error) || '未记录原因', 65)}`;
  }
  const data = isObject(result) && result.status === 'ok' && result.data !== undefined ? result.data : result;
  const count = countResultRecords(data);
  if (count !== null) return `返回 ${count} 条公开证据${tool === 'view_image' ? '，已完成图片复核' : ''}`;
  if (tool === 'view_image') return '图片内容已由 Qwen3.7-Flash 复核';
  return truncate(formatMetadataValue(isObject(data) ? (data.summary || data.caption || data.description || '工具调用完成') : data || '工具调用完成'), 95);
}

function toolResultFailed(payload) {
  const status = text(payload?.status).toLowerCase();
  if (['error', 'failed', 'failure'].includes(status)) return true;
  const result = parseToolObservation(payload?.observation ?? payload?.result ?? payload);
  if (!isObject(result)) return false;
  if (text(result.status).toLowerCase() === 'error' || result.error) return true;
  return Boolean(isObject(result.data) && result.data.error);
}

function traceDuration(payload) {
  const durationValue = payload.duration_ms ?? payload.elapsed_ms ?? payload.latency_ms;
  const duration = durationValue != null ? formatDuration(Number(durationValue)) : '';
  return [payload.cached ? '缓存' : '', duration].filter(Boolean).join(' · ');
}

function addOrUpdateTrace(type, payload = {}) {
  const list = $('#trace-timeline');
  const key = traceKey(type, payload);
  let item = state.traces.get(key);
  const isResult = type === 'tool_result';
  const answerStatus = type === 'answer' && ['error', 'partial'].includes(payload.status) ? payload.status : null;
  const status = type === 'error' || (isResult && toolResultFailed(payload)) ? 'error' :
    (answerStatus || (isResult || type === 'answer' || type === 'request_started' || payload.status === 'complete' ? 'success' : 'running'));

  if (!item) {
    item = document.createElement('li');
    item.className = `timeline-item is-${status}`;
    item.dataset.traceKey = key;
    if (type === 'tool_call' || type === 'tool_result') {
      item.dataset.tool = payload.tool || '';
      item.dataset.step = payload.step ?? '';
    }
    const marker = document.createElement('span');
    marker.className = 'timeline-marker';
    marker.setAttribute('aria-hidden', 'true');
    const body = document.createElement('div');
    body.className = 'timeline-body';
    const top = document.createElement('div');
    top.className = 'timeline-top';
    const title = document.createElement('h3');
    title.className = 'timeline-title';
    const duration = document.createElement('span');
    duration.className = 'timeline-duration';
    const summary = document.createElement('p');
    summary.className = 'timeline-summary';
    top.append(title, duration);
    body.append(top, summary);
    item.append(marker, body);
    list.append(item);
    state.traces.set(key, item);
    state.traceCount += 1;
  }

  item.className = `timeline-item is-${status}`;
  $('.timeline-marker', item).textContent = iconFor(type, status);
  $('.timeline-title', item).textContent = traceTitle(type, payload);
  $('.timeline-summary', item).textContent = traceSummary(type, payload);
  $('.timeline-duration', item).textContent = traceDuration(payload);

  const detailValue = isResult ? (payload.observation ?? payload.result) : (type === 'tool_call' ? (payload.args ?? payload.arguments) : null);
  if (detailValue != null) {
    let details = $('.timeline-details', item);
    if (!details) {
      details = document.createElement('details');
      details.className = 'timeline-details';
      const summary = document.createElement('summary');
      summary.textContent = '查看公开字段';
      const pre = document.createElement('pre');
      details.append(summary, pre);
      $('.timeline-body', item).append(details);
    }
    $('pre', details).textContent = safeJson(detailValue);
  }
  $('#trace-counter').textContent = `${state.traceCount} 个阶段`;
  item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  return item;
}

function settleRunningTraces(failed = false) {
  state.traces.forEach(item => {
    if (!item.classList.contains('is-running')) return;
    item.className = `timeline-item is-${failed ? 'error' : 'success'}`;
    $('.timeline-marker', item).textContent = failed ? '!' : '✓';
  });
}

function evidenceFor(card) {
  if (!card) return new Map();
  let evidence = state.evidenceByCard.get(card);
  if (!evidence) {
    evidence = new Map();
    state.evidenceByCard.set(card, evidence);
  }
  return evidence;
}

function collectEvidence(value, depth = 0, card = state.currentCard) {
  if (depth > 7 || value == null) return;
  const parsed = parseToolObservation(value);
  if (Array.isArray(parsed)) {
    parsed.forEach(item => collectEvidence(item, depth + 1, card));
    return;
  }
  if (!isObject(parsed)) return;
  const id = parsed.image_id || (typeof parsed.id === 'string' && parsed.id.startsWith('img_') ? parsed.id : null);
  const ids = [];
  if (id) ids.push(id);
  ['image_ids', 'matched_image_ids'].forEach(key => {
    if (!Array.isArray(parsed[key])) return;
    parsed[key].forEach(imageId => {
      if (typeof imageId === 'string' && imageId.startsWith('img_') && !ids.includes(imageId)) ids.push(imageId);
    });
  });
  if (ids.length) {
    const evidence = evidenceFor(card);
    const publicMeta = {
      date: pickFirst(parsed, ['captured_at', 'date_taken', 'datetime', 'date', 'time', 'date_start']),
      location: pickFirst(parsed, ['location', 'place', 'address', 'location_name']),
      people: pickFirst(parsed, ['people', 'persons', 'identities']),
      description: pickFirst(parsed, ['caption', 'description', 'description_preview', 'summary', 'observation', 'event_name', 'name', 'event']),
      filename: pickFirst(parsed, ['file_name', 'filename'])
    };
    Object.keys(publicMeta).forEach(key => publicMeta[key] == null && delete publicMeta[key]);
    ids.forEach(imageId => {
      const current = evidence.get(imageId) || {};
      evidence.set(imageId, { ...current, ...publicMeta });
    });
  }
  Object.entries(parsed).forEach(([key, item]) => {
    if (!/(?:path|root|directory|image_url|prompt|message|secret|key)/i.test(key)) collectEvidence(item, depth + 1, card);
  });
}

function mergeEvidenceMetadata(card, value) {
  if (!value) return;
  if (Array.isArray(value)) {
    collectEvidence(value, 0, card);
    return;
  }
  if (!isObject(value)) return;
  Object.entries(value).forEach(([imageId, metadata]) => {
    if (!/^img_[A-Za-z0-9_-]+$/.test(imageId) || !isObject(metadata)) return;
    collectEvidence({ image_id: imageId, ...metadata }, 0, card);
  });
}

function reconcileToolTrace(rows, card = state.currentCard) {
  if (!Array.isArray(rows)) return;
  const occurrences = new Map();
  rows.forEach(row => {
    if (!isObject(row)) return;
    collectEvidence(row.result ?? row.observation, 0, card);
    const base = toolBase(row);
    const index = occurrences.get(base) || 0;
    occurrences.set(base, index + 1);
    const key = (state.toolKeys.get(base) || [])[index];
    const finalPayload = {
      ...row,
      observation: row.result ?? row.observation ?? row.summary,
      status: row.status,
      duration_ms: row.duration_ms
    };
    const item = (key ? state.traces.get(key) : null) || addOrUpdateTrace('tool_result', finalPayload);
    const failed = toolResultFailed(finalPayload);
    item.className = `timeline-item is-${failed ? 'error' : 'success'}`;
    $('.timeline-marker', item).textContent = failed ? '!' : '✓';
    $('.timeline-duration', item).textContent = traceDuration(row);
    $('.timeline-summary', item).textContent = row.summary ? truncate(publicText(row.summary), 95) : summarizeToolResult(row.tool, row.result ?? row.observation);
    const detailValue = row.result ?? row.observation;
    if (detailValue != null) {
      let details = $('.timeline-details', item);
      if (!details) {
        details = document.createElement('details');
        details.className = 'timeline-details';
        const label = document.createElement('summary');
        label.textContent = '查看公开字段';
        const pre = document.createElement('pre');
        details.append(label, pre);
        $('.timeline-body', item).append(details);
      }
      $('pre', details).textContent = safeJson(detailValue);
    }
  });
}

function mediaUrl(imageId, variant = 'thumb') {
  return `/api/albums/${encodeURIComponent(state.albumId)}/media/${encodeURIComponent(imageId)}?variant=${encodeURIComponent(variant)}`;
}

function normalizeIds(value) {
  return [...new Set((Array.isArray(value) ? value : []).filter(id => typeof id === 'string' && /^img_[A-Za-z0-9_-]+$/.test(id)))];
}

function renderGallery(card, answer) {
  const selected = normalizeIds(answer.selected_image_ids);
  const retrieved = normalizeIds(answer.retrieved_image_ids);
  const ids = [...new Set([...selected, ...retrieved])];
  if (!ids.length) return;

  const section = $('.evidence-section', card);
  const gallery = $('.evidence-gallery', card);
  $('.image-count', card).textContent = `共 ${ids.length} 张`;
  gallery.replaceChildren();
  ids.forEach(id => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'evidence-card';
    button.setAttribute('aria-label', `查看图片证据 ${id}`);
    const image = document.createElement('img');
    image.src = mediaUrl(id, 'thumb');
    image.alt = `相册证据 ${id}`;
    image.loading = 'lazy';
    image.addEventListener('error', () => {
      image.alt = `图片 ${id} 暂时无法加载`;
      button.classList.add('image-unavailable');
    }, { once: true });
    const body = document.createElement('span');
    body.className = 'evidence-card-body';
    const badges = document.createElement('span');
    badges.className = 'evidence-badges';
    if (selected.includes(id)) badges.append(makeBadge('最终采用', 'selected'));
    if (retrieved.includes(id)) badges.append(makeBadge('召回候选', 'retrieved'));
    const idLabel = document.createElement('span');
    idLabel.className = 'evidence-id';
    idLabel.textContent = id;
    body.append(badges, idLabel);
    button.append(image, body);
    button.addEventListener('click', () => openImage(card, id, selected.includes(id), retrieved.includes(id)));
    gallery.append(button);
  });
  section.hidden = false;
}

function makeBadge(label, className) {
  const badge = document.createElement('span');
  badge.className = `evidence-badge ${className}`;
  badge.textContent = label;
  return badge;
}

function formatMetadataValue(value) {
  if (value == null) return '';
  if (typeof value === 'string') return publicText(value);
  if (typeof value === 'number' || typeof value === 'boolean') return text(value);
  if (Array.isArray(value)) return value.map(item => formatMetadataValue(item)).filter(Boolean).join('、');
  if (isObject(value)) return safeJson(value);
  return publicText(value);
}

function openImage(card, id, selected, retrieved) {
  const dialog = $('#image-dialog');
  const image = $('#dialog-image');
  const meta = evidenceFor(card).get(id) || {};
  image.src = mediaUrl(id, 'full');
  image.alt = `相册证据大图 ${id}`;
  $('#image-dialog-title').textContent = formatMetadataValue(meta.filename) || '照片预览';
  $('#dialog-meta').replaceChildren();
  const badges = document.createElement('div');
  badges.className = 'dialog-badges';
  if (selected) badges.append(makeBadge('最终采用', 'selected'));
  if (retrieved) badges.append(makeBadge('召回候选', 'retrieved'));
  const heading = document.createElement('h3');
  heading.textContent = id;
  const list = document.createElement('dl');
  const labels = { date: '拍摄时间', location: '地点', people: '人物', description: '公开观察' };
  Object.entries(labels).forEach(([key, label]) => {
    if (meta[key] == null) return;
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = label;
    dd.textContent = formatMetadataValue(meta[key]);
    list.append(dt, dd);
  });
  if (!list.children.length) {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = '元数据';
    dd.textContent = '本次轨迹没有返回可公开展示的照片元数据。';
    list.append(dt, dd);
  }
  $('#dialog-meta').append(badges, heading, list);
  if (!dialog.open) dialog.showModal();
}

function renderMetrics(card, answer) {
  const calls = Array.isArray(answer.model_call_metrics) ? answer.model_call_metrics : [];
  const usage = isObject(answer.usage) ? answer.usage : {};
  if (!calls.length && !Object.keys(usage).length) return;
  const section = $('.metrics-section', card);
  const summary = $('.metrics-summary', card);
  summary.replaceChildren();
  const promptTokens = usage.prompt_tokens ?? calls.reduce((sum, item) => sum + (Number(item.prompt_tokens) || 0), 0);
  const completionTokens = usage.completion_tokens ?? calls.reduce((sum, item) => sum + (Number(item.completion_tokens) || 0), 0);
  const estimatedCost = usage.estimated_cny ?? calls.reduce((sum, item) => sum + (Number(item.estimated_cny) || 0), 0);
  const reservedCost = usage.reserved_cny ?? calls.reduce((sum, item) => sum + (Number(item.reserved_cny) || 0), 0);
  const totalLatency = answer.latency_s != null ? answer.latency_s * 1000 : null;
  [
    ['Qwen 调用', `${calls.length || usage.requests || 0} 次`],
    ['输入 Token', formatNumber(promptTokens)],
    ['输出 Token', formatNumber(completionTokens)],
    ['已估费用', formatCost(Number(estimatedCost))],
    ['不确定预留', formatCost(Number(reservedCost))],
    ['问答总耗时', formatDuration(totalLatency)]
  ].forEach(([label, value]) => {
    const box = document.createElement('div');
    box.className = 'metric-card';
    const name = document.createElement('span');
    const number = document.createElement('strong');
    name.textContent = label;
    number.textContent = value;
    box.append(name, number);
    summary.append(box);
  });

  const callsRoot = $('.model-calls', card);
  callsRoot.replaceChildren();
  calls.forEach((call, index) => {
    const row = document.createElement('div');
    row.className = 'model-call';
    const title = document.createElement('strong');
    const value = document.createElement('span');
    const role = call.role || call.call_type || 'agent';
    title.textContent = `${index + 1}. ${role} · ${call.model || 'qwen3.7-flash'}`;
    const status = call.status || '未记录';
    const error = call.error_code || call.error?.code || call.error?.reason || call.error;
    row.classList.toggle('is-error', ['error', 'failed', 'late_response'].includes(text(status).toLowerCase()) || Boolean(error));
    const parts = [
      `状态 ${status}`,
      `输入 ${formatNumber(Number(call.prompt_tokens))}`,
      `输出 ${formatNumber(Number(call.completion_tokens))}`,
      `错误 ${error ? truncate(formatMetadataValue(error), 60) : '无'}`,
      formatDuration(call.duration_ms ?? call.elapsed_ms),
      call.ttft_ms != null ? `TTFT ${formatDuration(call.ttft_ms)}` : null,
      call.tokens_per_second != null ? `${formatNumber(call.tokens_per_second, 1)} token/s` : null,
      call.estimated_cny != null ? `已估 ${formatCost(call.estimated_cny)}` : null,
      call.reserved_cny != null ? `预留 ${formatCost(call.reserved_cny)}` : null
    ].filter(Boolean);
    value.textContent = parts.join(' · ');
    row.append(title, value);
    callsRoot.append(row);
  });
  section.hidden = false;
}

function normalizeAnswer(payload) {
  const source = isObject(payload.answer) ? payload.answer : payload;
  return {
    ...source,
    final_text: source.final_text || source.response || (typeof payload.answer === 'string' ? payload.answer : '') || payload.response || payload.final_text,
    selected_image_ids: source.selected_image_ids || payload.selected_image_ids || [],
    retrieved_image_ids: source.retrieved_image_ids || payload.retrieved_image_ids || [],
    model_call_metrics: source.model_call_metrics || payload.model_call_metrics || [],
    usage: source.usage || payload.usage || {},
    latency_s: source.latency_s ?? payload.latency_s,
    stopped_reason: source.stopped_reason ?? payload.stopped_reason,
    turn_outcome: source.turn_outcome ?? payload.turn_outcome,
    tool_trace: source.tool_trace || payload.tool_trace || [],
    evidence_metadata: source.evidence_metadata || payload.evidence_metadata || {}
  };
}

function answerDisposition(answer) {
  const outcome = text(answer.turn_outcome).toLowerCase();
  const reason = text(answer.stopped_reason).toLowerCase();
  if (outcome === 'error') return 'error';
  if (reason === 'insufficient_evidence' || outcome === 'insufficient_evidence') return 'partial';
  if (reason && reason !== 'ok') return 'error';
  return 'success';
}

function completeConversation(payload) {
  const card = state.currentCard;
  if (!card) return;
  const answer = normalizeAnswer(payload);
  $('.answer-waiting', card).hidden = true;
  const content = $('.answer-content', card);
  content.textContent = publicText(answer.final_text || '本次执行完成，但没有返回可展示的答案。');
  content.hidden = false;
  mergeEvidenceMetadata(card, answer.evidence_metadata);
  const answerState = $('.answer-state', card);
  const disposition = answerDisposition(answer);
  answerState.textContent = disposition === 'error' ? '执行失败' : (disposition === 'partial' ? '证据不足' : '已完成');
  answerState.className = `answer-state ${disposition === 'error' ? 'state-failed' : (disposition === 'partial' ? 'state-partial' : 'state-complete')}`;
  reconcileToolTrace(answer.tool_trace || [], card);
  renderGallery(card, answer);
  renderMetrics(card, answer);
  settleRunningTraces(disposition === 'error');
  addOrUpdateTrace('answer', { ...payload, ...answer, status: disposition });
  $('#trace-counter').textContent = `${state.traceCount} 个阶段 · ${disposition === 'error' ? '失败' : (disposition === 'partial' ? '部分完成' : '完成')}`;
}

function failConversation(message) {
  const card = state.currentCard;
  if (card) {
    $('.answer-waiting', card).hidden = true;
    const error = $('.answer-error', card);
    error.textContent = `${publicText(message)} 页面不会自动重试；再次发送会产生一次新的云端调用。`;
    error.hidden = false;
    const answerState = $('.answer-state', card);
    answerState.textContent = '未完成';
    answerState.className = 'answer-state state-failed';
  }
  settleRunningTraces(true);
  addOrUpdateTrace('error', { message });
  $('#trace-counter').textContent = `${state.traceCount} 个阶段 · 失败`;
}

function handleEvent(type, payload) {
  const data = isObject(payload) ? payload : { message: text(payload) };
  if (type === 'tool_call' || type === 'tool_result') collectEvidence(data.observation ?? data.result ?? data, 0, state.currentCard);
  if (type === 'answer') completeConversation(data);
  else if (type === 'error') {
    const error = new Error(data.message || data.error || '上游执行失败');
    error.reloadRequired = ['index_changed', 'index_or_policy_mismatch'].includes(data.code);
    throw error;
  }
  else addOrUpdateTrace(type, data);
}

function dispatchSseBlock(block) {
  let eventName = 'message';
  const dataLines = [];
  block.split(/\r?\n/).forEach(line => {
    if (!line || line.startsWith(':')) return;
    const separator = line.indexOf(':');
    const field = separator < 0 ? line : line.slice(0, separator);
    let value = separator < 0 ? '' : line.slice(separator + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') eventName = value;
    if (field === 'data') dataLines.push(value);
  });
  if (!dataLines.length) return false;
  let payload;
  const raw = dataLines.join('\n');
  try { payload = JSON.parse(raw); } catch { payload = { message: raw }; }
  handleEvent(eventName, payload);
  return eventName === 'answer';
}

async function consumeSse(response) {
  if (!response.body) throw new Error('浏览器未收到可读取的事件流');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let completed = false;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || '';
    for (const block of blocks) if (block.trim()) completed = dispatchSseBlock(block) || completed;
    if (done) break;
  }
  if (buffer.trim()) completed = dispatchSseBlock(buffer) || completed;
  if (!completed) throw new Error('连接已结束，但没有收到最终答案，调用状态可能不确定。');
}

async function ask(question) {
  if (state.busy || !state.interactive || state.reloadRequired) return;
  setBusy(true);
  showNotice('');
  resetTrace();
  state.currentCard = createConversation(question);
  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      cache: 'no-store',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({ album_id: state.albumId, question })
    });
    if (!response.ok) {
      let detail = '';
      try {
        const body = await response.json();
        detail = body.detail || body.error || body.message || '';
      } catch { /* response has no safe JSON detail */ }
      if (response.status === 429) throw new Error('当前已有一个问题正在执行，请稍后再试。');
      if (response.status === 409) {
        const error = new Error(detail || '冻结索引版本已经变化，请停止使用并联系维护人员。');
        error.reloadRequired = true;
        throw error;
      }
      throw new Error(detail || `服务请求失败（HTTP ${response.status}）`);
    }
    await consumeSse(response);
  } catch (error) {
    if (error.reloadRequired) requireReload(error.message || '冻结索引版本已经变化。');
    failConversation(error.message || '请求未完成。');
  } finally {
    setBusy(false);
    state.currentCard = null;
    $('#question-input').focus();
  }
}

$('#question-form').addEventListener('submit', event => {
  event.preventDefault();
  const input = $('#question-input');
  const question = input.value.trim();
  if (!question || state.busy || !state.interactive || state.reloadRequired) return;
  input.value = '';
  autoGrow();
  ask(question);
});

$('#question-input').addEventListener('input', autoGrow);
$('#question-input').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $('#question-form').requestSubmit();
  }
});

$$('.suggestion').forEach(button => button.addEventListener('click', () => {
  $('#question-input').value = button.textContent.trim();
  autoGrow();
  $('#question-input').focus();
}));

$('#dialog-close').addEventListener('click', () => $('#image-dialog').close());
$('#image-dialog').addEventListener('click', event => {
  if (event.target === $('#image-dialog')) $('#image-dialog').close();
});

loadStatus();

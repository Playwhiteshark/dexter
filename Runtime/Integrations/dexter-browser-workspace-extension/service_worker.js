const HOST_NAME = 'com.dexter.workspace_bridge';
const TAB_GROUP_NONE = -1;

let nativePort = null;
let reconnectTimer = null;
let browserKindPromise = null;

async function detectBrowserKind() {
  if (browserKindPromise) return browserKindPromise;
  browserKindPromise = (async () => {
    try {
      if (self.navigator && self.navigator.brave && typeof self.navigator.brave.isBrave === 'function') {
        if (await self.navigator.brave.isBrave()) return 'brave';
      }
    } catch (_) {}
    const ua = (self.navigator && self.navigator.userAgent) || '';
    if (/Edg\//.test(ua)) return 'edge';
    return 'chrome';
  })();
  return browserKindPromise;
}

function postResponse(requestId, ok, result, error) {
  if (!nativePort) return;
  try {
    nativePort.postMessage({
      type: 'response',
      request_id: requestId,
      ok: !!ok,
      result: result || null,
      error: error || null
    });
  } catch (_) {}
}

function effectiveTabUrl(tab) {
  // pendingUrl is the destination of an in-flight navigation; url is only the
  // last committed main-frame URL. Prefer pendingUrl so saving during a page
  // load cannot accidentally persist an interim about:blank white page.
  return (tab && (tab.pendingUrl || tab.url)) || '';
}

async function captureWindows() {
  const windows = await chrome.windows.getAll({populate: true, windowTypes: ['normal']});
  const result = [];

  for (const win of windows) {
    if (win.incognito) continue;

    const groupIds = new Set();
    const tabs = [];
    for (const tab of (win.tabs || [])) {
      if (typeof tab.groupId === 'number' && tab.groupId !== TAB_GROUP_NONE) {
        groupIds.add(tab.groupId);
      }
      tabs.push({
        id: tab.id,
        index: tab.index,
        url: effectiveTabUrl(tab),
        committed_url: tab.url || '',
        pending_url: tab.pendingUrl || '',
        status: tab.status || '',
        title: tab.title || '',
        pinned: !!tab.pinned,
        active: !!tab.active,
        group_key: (typeof tab.groupId === 'number' && tab.groupId !== TAB_GROUP_NONE)
          ? `group:${tab.groupId}`
          : null
      });
    }

    const groups = [];
    for (const groupId of groupIds) {
      try {
        const group = await chrome.tabGroups.get(groupId);
        groups.push({
          key: `group:${groupId}`,
          title: group.title || '',
          color: group.color || 'grey',
          collapsed: !!group.collapsed
        });
      } catch (_) {}
    }

    result.push({
      id: win.id,
      focused: !!win.focused,
      incognito: !!win.incognito,
      state: win.state,
      left: win.left,
      top: win.top,
      width: win.width,
      height: win.height,
      tabs,
      groups
    });
  }
  return {windows: result};
}

function groupMetadataMap(groups) {
  const map = new Map();
  for (const group of Array.isArray(groups) ? groups : []) {
    if (group && typeof group.key === 'string' && group.key) map.set(group.key, group);
  }
  return map;
}

async function createSavedTab(windowId, record, active) {
  if (!record || typeof record.url !== 'string' || !record.url) return null;
  try {
    return await chrome.tabs.create({
      windowId,
      url: record.url,
      active: !!active,
      pinned: !!record.pinned
    });
  } catch (err) {
    return {__failed: true, error: String(err && err.message ? err.message : err), record};
  }
}

function bucketExistingTabs(tabs) {
  const byUrl = new Map();
  for (const tab of tabs) {
    const url = effectiveTabUrl(tab);
    if (!url) continue;
    if (!byUrl.has(url)) byUrl.set(url, []);
    byUrl.get(url).push(tab);
  }
  return byUrl;
}

function takeReusableTab(byUrl, url, usedIds) {
  const bucket = byUrl.get(url) || [];
  while (bucket.length) {
    const tab = bucket.shift();
    if (Number.isInteger(tab.id) && !usedIds.has(tab.id)) return tab;
  }
  return null;
}

async function waitForActiveNavigation(tabId, expectedUrl, timeoutMs = 6000) {
  const deadline = Date.now() + timeoutMs;
  let last = null;
  while (Date.now() < deadline) {
    try {
      const tab = await chrome.tabs.get(tabId);
      last = {
        url: tab.url || '',
        pendingUrl: tab.pendingUrl || '',
        status: tab.status || ''
      };
      // Once the requested URL has committed and is no longer pending, the tab
      // is no longer the blank transient page that caused the v6 white-screen
      // failure. Do not require full network-idle/complete for SPA pages.
      if (!tab.pendingUrl && tab.url && tab.url !== 'about:blank') {
        return {ready: true, ...last};
      }
    } catch (_) {}
    await new Promise(resolve => setTimeout(resolve, 80));
  }
  return {ready: false, ...(last || {})};
}

async function restoreGroups(windowId, items, groups, failures) {
  const groupMeta = groupMetadataMap(groups);
  const byKey = new Map();

  for (const item of items) {
    const key = item.record && item.record.group_key;
    if (!key) {
      if (typeof item.tab.groupId === 'number' && item.tab.groupId !== TAB_GROUP_NONE) {
        try { await chrome.tabs.ungroup(item.tab.id); } catch (_) {}
      }
      continue;
    }
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key).push(item);
  }

  for (const [key, groupedItems] of byKey.entries()) {
    const tabIds = groupedItems.map(item => item.tab.id).filter(Number.isInteger);
    if (!tabIds.length) continue;

    // If this is the same live browser session, preserve the existing native
    // group rather than tearing it down and rebuilding it.
    let groupId = null;
    for (const item of groupedItems) {
      if (typeof item.tab.groupId === 'number' && item.tab.groupId !== TAB_GROUP_NONE) {
        if (`group:${item.tab.groupId}` === key) {
          groupId = item.tab.groupId;
          break;
        }
      }
    }

    try {
      if (groupId === null) {
        for (const item of groupedItems) {
          if (typeof item.tab.groupId === 'number' && item.tab.groupId !== TAB_GROUP_NONE) {
            try { await chrome.tabs.ungroup(item.tab.id); } catch (_) {}
          }
        }
        groupId = await chrome.tabs.group({tabIds, createProperties: {windowId}});
      } else {
        const needsAdding = groupedItems
          .filter(item => item.tab.groupId !== groupId)
          .map(item => item.tab.id);
        if (needsAdding.length) await chrome.tabs.group({tabIds: needsAdding, groupId});
      }

      const meta = groupMeta.get(key) || {};
      const update = {
        title: typeof meta.title === 'string' ? meta.title : '',
        collapsed: !!meta.collapsed
      };
      const allowedColors = new Set(['grey','blue','red','yellow','green','pink','purple','cyan','orange']);
      if (allowedColors.has(meta.color)) update.color = meta.color;
      await chrome.tabGroups.update(groupId, update);
    } catch (err) {
      failures.push({group_key: key, error: String(err && err.message ? err.message : err)});
    }
  }
}

async function restoreWindow(payload) {
  const windowId = Number(payload.window_id);
  if (!Number.isInteger(windowId)) throw new Error('restore_window requires a numeric window_id');

  const savedTabs = Array.isArray(payload.tabs) ? payload.tabs : [];
  if (!savedTabs.length) throw new Error('saved browser window has no tabs to restore');

  const existingWindow = await chrome.windows.get(windowId, {populate: true});
  const existingTabs = existingWindow.tabs || [];
  const byUrl = bucketExistingTabs(existingTabs);
  const usedExistingIds = new Set();
  const items = [];
  const failures = [];

  const activeIndex = Math.max(0, Math.min(
    Number.isInteger(payload.active_index) ? payload.active_index : 0,
    savedTabs.length - 1
  ));

  // Reuse exact-URL tabs first. This is intentionally reconciliation rather
  // than destroy/recreate: restoring an already-open ChatGPT/SPA tab should not
  // throw away its live renderer/session merely to get the same URL back.
  for (let i = 0; i < savedTabs.length; i += 1) {
    const record = savedTabs[i];
    const reused = takeReusableTab(byUrl, record && record.url, usedExistingIds);
    if (reused) {
      usedExistingIds.add(reused.id);
      items.push({savedIndex: i, tab: reused, record, reused: true});
      continue;
    }

    const created = await createSavedTab(windowId, record, i === activeIndex);
    if (!created) continue;
    if (created.__failed) {
      failures.push({index: i, url: record?.url || '', error: created.error});
      continue;
    }
    items.push({savedIndex: i, tab: created, record, reused: false});
  }

  if (!items.length) {
    throw new Error(`could not reconcile any saved tabs (${failures.length} failed)`);
  }

  // Only remove tabs that are not part of the saved state after replacements
  // exist. Tabs that already match remain untouched.
  const keepIds = new Set(items.map(item => item.tab.id).filter(Number.isInteger));
  const extraIds = existingTabs
    .map(tab => tab.id)
    .filter(id => Number.isInteger(id) && !keepIds.has(id));
  if (extraIds.length) {
    try { await chrome.tabs.remove(extraIds); } catch (_) {}
  }

  // Restore order/pin state without reopening reused pages.
  for (let i = 0; i < items.length; i += 1) {
    const item = items[i];
    try {
      await chrome.tabs.move(item.tab.id, {index: i});
      await chrome.tabs.update(item.tab.id, {pinned: !!item.record.pinned});
      item.tab = await chrome.tabs.get(item.tab.id);
    } catch (_) {}
  }

  await restoreGroups(windowId, items, payload.groups, failures);

  const activeItem = items.find(item => item.savedIndex === activeIndex) || items[0];
  let activeNavigation = {ready: true};
  try {
    await chrome.tabs.update(activeItem.tab.id, {active: true});
    await chrome.windows.update(windowId, {focused: true});
    activeNavigation = await waitForActiveNavigation(activeItem.tab.id, activeItem.record.url);
  } catch (err) {
    activeNavigation = {ready: false, error: String(err && err.message ? err.message : err)};
  }

  return {
    tab_count: items.length,
    requested_tab_count: savedTabs.length,
    reused_tab_count: items.filter(item => item.reused).length,
    created_tab_count: items.filter(item => !item.reused).length,
    removed_extra_tab_count: extraIds.length,
    active_tab_ready: !!activeNavigation.ready,
    active_tab_observed_url: activeNavigation.url || activeNavigation.pendingUrl || '',
    failed_count: failures.length,
    failures
  };
}

async function handleNativeRequest(message) {
  if (!message || message.type !== 'request' || !message.request_id) return;
  try {
    let result;
    if (message.op === 'capture') {
      result = await captureWindows();
    } else if (message.op === 'restore_window') {
      result = await restoreWindow(message.payload || {});
    } else {
      throw new Error(`Unsupported Dexter browser operation: ${message.op}`);
    }
    postResponse(message.request_id, true, result, null);
  } catch (err) {
    postResponse(
      message.request_id,
      false,
      null,
      String(err && err.message ? err.message : err)
    );
  }
}

async function connectNative() {
  if (nativePort) return;
  try {
    const kind = await detectBrowserKind();
    const port = chrome.runtime.connectNative(HOST_NAME);
    nativePort = port;
    port.onMessage.addListener(handleNativeRequest);
    port.onDisconnect.addListener(() => {
      if (nativePort === port) nativePort = null;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connectNative, 1200);
    });
    port.postMessage({type: 'hello', browser_kind: kind, extension_id: chrome.runtime.id});
  } catch (_) {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connectNative, 1800);
  }
}

chrome.runtime.onStartup.addListener(connectNative);
chrome.runtime.onInstalled.addListener(connectNative);
connectNative();

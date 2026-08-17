const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

let sessionId;
let root;
let sessionFile;
let writeTimer;
let heartbeatTimer;

function currentState() {
  const folders = (vscode.workspace.workspaceFolders || []).map(folder => folder.uri.fsPath);
  const workspaceFile = vscode.workspace.workspaceFile ? vscode.workspace.workspaceFile.toString(true) : null;
  const active = vscode.window.activeTextEditor;
  return {
    session_id: sessionId,
    pid: process.pid,
    workspace_name: vscode.workspace.name || null,
    workspace_file: workspaceFile,
    workspace_folders: folders,
    active_file: active ? active.document.uri.fsPath : null,
    focused: !!vscode.window.state.focused,
    editor_restore_strategy: 'vscode_native_session',
    updated_unix: Date.now() / 1000
  };
}

function atomicWriteJson(target, value) {
  fs.mkdirSync(path.dirname(target), {recursive: true});
  const temp = `${target}.${process.pid}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(value), 'utf8');
  fs.renameSync(temp, target);
}

function scheduleStateWrite() {
  if (writeTimer) clearTimeout(writeTimer);
  writeTimer = setTimeout(() => {
    try { atomicWriteJson(sessionFile, currentState()); } catch (_) {}
  }, 80);
}

function writeHeartbeat() {
  try { atomicWriteJson(sessionFile, currentState()); } catch (_) {}
}

function activate(context) {
  root = path.join(process.env.LOCALAPPDATA || context.globalStorageUri.fsPath, 'Dexter', 'vscode_bridge');
  sessionId = `vscode-${process.pid}-${crypto.randomBytes(5).toString('hex')}`;
  sessionFile = path.join(root, 'sessions', `${sessionId}.json`);
  fs.mkdirSync(path.dirname(sessionFile), {recursive: true});

  writeHeartbeat();
  const disposables = [
    vscode.window.onDidChangeActiveTextEditor(scheduleStateWrite),
    vscode.window.onDidChangeWindowState(scheduleStateWrite),
    vscode.workspace.onDidChangeWorkspaceFolders(scheduleStateWrite)
  ];
  disposables.forEach(d => context.subscriptions.push(d));

  // Dexter only needs workspace identity from VS Code. VS Code already owns
  // editor/tab persistence (including dirty and untitled buffers), so this
  // bridge deliberately does not close/reopen individual editors.
  heartbeatTimer = setInterval(writeHeartbeat, 2000);
  context.subscriptions.push({dispose: () => heartbeatTimer && clearInterval(heartbeatTimer)});
  context.subscriptions.push({dispose: () => {
    try { fs.unlinkSync(sessionFile); } catch (_) {}
  }});
}

function deactivate() {
  if (heartbeatTimer) clearInterval(heartbeatTimer);
  if (sessionFile) {
    try { fs.unlinkSync(sessionFile); } catch (_) {}
  }
}

module.exports = {activate, deactivate};

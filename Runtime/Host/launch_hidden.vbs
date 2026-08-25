Option Explicit

Dim fso, shell, scriptDir, repoRoot, pythonw, command
Dim localAppData, logDir, logPath
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
repoRoot = fso.GetAbsolutePathName(fso.BuildPath(scriptDir, "..\.."))
pythonw = fso.BuildPath(repoRoot, ".venv\Scripts\pythonw.exe")

If Not fso.FileExists(pythonw) Then
    MsgBox "Dexter's root virtual environment is missing." & vbCrLf & _
           "Run 'uv sync' from:" & vbCrLf & repoRoot, _
           vbCritical, "Dexter"
    WScript.Quit 1
End If

' Keep the normal shortcut on the validated production TTS variant. The
' backend preserves inherited developer/user overrides, so setting this here
' also fixes the hidden-launch path that otherwise still defaulted to Q8.
shell.Environment("PROCESS")("DEXTER_TTS_VARIANT") = "q4"

' pythonw launched directly from wscript can have no usable stderr handle. The
' NeuTTS child inherits Host stderr while reserving stdout for its JSON protocol,
' so keep stderr attached to a real file even though the whole launch stays
' hidden. This also gives production-only startup failures somewhere permanent
' to report themselves.
localAppData = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%")
logDir = fso.BuildPath(localAppData, "Dexter\Diagnostics")
If Not fso.FolderExists(logDir) Then
    fso.CreateFolder(logDir)
End If
logPath = fso.BuildPath(logDir, "host-stderr.log")

shell.CurrentDirectory = repoRoot
command = "cmd.exe /d /c " & Chr(34) & Chr(34) & pythonw & Chr(34) & _
          " -m Runtime.Host.app 2>>" & Chr(34) & logPath & Chr(34) & Chr(34)

' wscript.exe and window style 0 keep the cmd/pythonw launch fully hidden. cmd
' exists only to provide stderr redirection; pythonw remains the long-lived Host.
shell.Run command, 0, False

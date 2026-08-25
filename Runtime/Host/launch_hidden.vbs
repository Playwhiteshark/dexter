Option Explicit

Dim fso, shell, scriptDir, repoRoot, pythonw, command
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

shell.CurrentDirectory = repoRoot
command = Chr(34) & pythonw & Chr(34) & " -m Runtime.Host.app"

' wscript.exe is a GUI-subsystem launcher and window style 0 keeps the spawned
' process hidden. pythonw.exe is used as well, so Dexter does not depend on a
' console window at either launch layer.
shell.Run command, 0, False

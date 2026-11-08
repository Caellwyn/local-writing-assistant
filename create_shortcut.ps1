# Creates an "Ollama Chat" shortcut on your desktop that opens app.py with no console window.
# Run again any time (e.g. after moving the project folder).

$project = $PSScriptRoot
$pythonw = Join-Path $project ".venv\Scripts\pythonw.exe"
$app     = Join-Path $project "app.py"
$link    = Join-Path ([Environment]::GetFolderPath("Desktop")) "Ollama Chat.lnk"

if (-not (Test-Path $pythonw)) { throw "Missing $pythonw - create the venv first (see README)." }

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)
$sc.TargetPath       = $pythonw
$sc.Arguments        = "`"$app`""
$sc.WorkingDirectory = $project
$sc.Description      = "Chat with local Ollama models"
$icon = Join-Path $project "ui\icon.ico"
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.Save()

Write-Host "Created $link"

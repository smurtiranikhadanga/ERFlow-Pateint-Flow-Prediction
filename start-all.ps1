$scriptPath = $MyInvocation.MyCommand.Path
$dir = Split-Path $scriptPath

Write-Host "Starting Backend..."
Start-Process "powershell" -ArgumentList "-NoExit", "-Command", "cd '$dir'; .\.venv\Scripts\Activate.ps1; python backend/main.py"

Write-Host "Starting Chatbot..."
Start-Process "powershell" -ArgumentList "-NoExit", "-Command", "cd '$dir'; .\.venv\Scripts\Activate.ps1; python chatbot/main.py"

Write-Host "Starting Frontend..."
Start-Process "powershell" -ArgumentList "-NoExit", "-Command", "cd '$dir'; npm run dev"

Write-Host "All services started in separate windows."

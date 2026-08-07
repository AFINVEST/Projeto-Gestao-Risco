# agendar_task_scheduler.ps1
# Cria tarefa agendada no Windows para rodar update_dash_risco_v2.bat todos os dias uteis as 18h.
#
# USO (uma vez, como Administrador):
#   powershell -ExecutionPolicy Bypass -File agendar_task_scheduler.ps1
#
# Ou passe hora customizada:
#   powershell -ExecutionPolicy Bypass -File agendar_task_scheduler.ps1 -Horario "18:30"
#
# Para remover a tarefa depois:
#   Unregister-ScheduledTask -TaskName "DashRiscoUpdateDiario" -Confirm:$false

param(
    [string]$Horario = "18:00",
    [string]$Diretorio = "Z:\Asset Management\Equipe\Marcos\Risco\Projeto-Gestao-Risco",
    [string]$TaskName = "DashRiscoUpdateDiario"
)

Write-Host "=" * 78
Write-Host "Agendador Task Scheduler - Dash Risco"
Write-Host "=" * 78

$batPath = Join-Path $Diretorio "update_dash_risco_v2.bat"

if (-not (Test-Path $batPath)) {
    Write-Host "[erro] Nao encontrou $batPath"
    exit 1
}

Write-Host "Diretorio    : $Diretorio"
Write-Host "Script .bat  : $batPath"
Write-Host "Horario      : $Horario (segunda a sexta)"
Write-Host "Task name    : $TaskName"
Write-Host ""

# Remove tarefa existente (se houver) — idempotente
try {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
    Write-Host "[info] Tarefa existente '$TaskName' removida (para recriar)."
} catch {
    Write-Host "[info] Nenhuma tarefa existente com nome '$TaskName'."
}

# Cria acao: roda o .bat dentro do diretorio do projeto
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c cd /d `"$Diretorio`" && update_dash_risco_v2.bat >> update_dash_risco.log 2>&1" `
    -WorkingDirectory $Diretorio

# Trigger: dias uteis (segunda a sexta) no horario
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At $Horario

# Rodar como o usuario atual, mesmo se nao estiver logado (StoragePassword se necessario)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Configuracoes: se falhar, tenta de novo 3x com 10min
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# Registra
try {
    Register-ScheduledTask -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description "Pipeline diario Dash Risco AF - Update Cota, Snapshot, Email" | Out-Null
    Write-Host ""
    Write-Host "[OK] Tarefa '$TaskName' agendada com sucesso."
    Write-Host "     Rodara segunda a sexta as $Horario"
    Write-Host "     Log em: $Diretorio\update_dash_risco.log"
    Write-Host ""
    Write-Host "Para testar manualmente:"
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host ""
    Write-Host "Para ver status:"
    Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
    Write-Host ""
    Write-Host "Para remover:"
    Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
} catch {
    Write-Host "[erro] Falha ao registrar tarefa: $_"
    Write-Host ""
    Write-Host "Tente rodar este script como Administrador:"
    Write-Host "  Clique com direito -> Executar como Administrador"
    exit 1
}

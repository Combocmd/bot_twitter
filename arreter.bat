@echo off
chcp 65001 >nul
title Arret du bot @UniverseCule
cd /d "%~dp0"
setlocal enabledelayedexpansion

echo.
if exist "data\bot.lock" (
    set /p BOTPID=<"data\bot.lock"
    echo   Arret du bot ^(PID !BOTPID!^)...
) else (
    echo   Aucun verrou trouve, arret des processus restants...
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*src.main*' -and $_.Name -match 'python' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

del /q "data\bot.lock" 2>nul

powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*src.main*' -and $_.Name -match 'python' }; if ($p) { Write-Host '   Echec: le bot tourne encore.' } else { Write-Host '   Bot arrete.' }"

echo.
pause

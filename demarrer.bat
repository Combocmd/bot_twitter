@echo off
chcp 65001 >nul
title Bot @UniverseCule
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   Environnement Python introuvable.
    echo   Lancez d'abord :  python setup.py
    echo.
    pause
    exit /b 1
)

set PYTHONIOENCODING=utf-8

echo.
echo   Demarrage du bot @UniverseCule...
echo   Laissez cette fenetre ouverte. Ctrl+C ou arreter.bat pour stopper.
echo.

".venv\Scripts\python.exe" -m src.main start
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
    echo   Le bot est arrete.
) else (
    echo   Le bot n'a pas demarre correctement ^(code %CODE%^).
)
pause

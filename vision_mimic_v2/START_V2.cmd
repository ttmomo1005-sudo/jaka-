@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
if not defined VISION_PYTHON set "VISION_PYTHON=python"
"%VISION_PYTHON%" -B -u "%~dp0launch.py"
pause

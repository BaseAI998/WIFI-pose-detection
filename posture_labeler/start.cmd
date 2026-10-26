@echo off
title Pose Annotation - Local Server
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "labelerExit=%errorlevel%"
pause
exit /b %labelerExit%

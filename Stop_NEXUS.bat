@echo off
title NEXUS Shutdown Sequence
echo ==================================================
echo Sending termination signal to NEXUS Core...
echo ==================================================
taskkill /F /IM python.exe
echo.
echo NEXUS has been successfully shut down.
pause

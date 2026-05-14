@echo off
title NEXUS Trading Engine
echo ==================================================
echo Starting NEXUS Institutional Engine...
echo ==================================================
cd /d "%~dp0"
python src/goldvx.py
pause

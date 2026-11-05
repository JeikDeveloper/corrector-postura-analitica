@echo off
rem Abre la página web de analítica de postura (doble clic). Ctrl+C en esta ventana para detenerla.
cd /d "%~dp0"
".venv\Scripts\postura.exe" web
pause

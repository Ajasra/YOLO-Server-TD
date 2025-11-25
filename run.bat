@echo off
echo Starting YOLO OSC Server...
uv run .\main.py --max-fps 30 --debug
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo -------------------------------------------
    echo Script finished with error code %ERRORLEVEL%
    echo -------------------------------------------
)
pause


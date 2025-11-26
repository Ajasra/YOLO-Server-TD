@echo off
echo Starting YOLO OSC Server...
uv run .\main.py --max-fps 15 --debug --source NDI
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo -------------------------------------------
    echo Script finished with error code %ERRORLEVEL%
    echo -------------------------------------------
)
pause


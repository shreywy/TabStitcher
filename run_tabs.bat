@echo off
title Tab Extractor

echo Updating Python and required packages... (may take a minute on first start)
python -m pip install --upgrade pip >nul 2>&1
python -m pip install -r requirements.txt >nul 2>&1
echo Requirements installed/verified.
echo.

:loop
set /p vidurl=Paste YouTube link (or type EXIT to quit): 

if /i "%vidurl%"=="exit" goto end

python tab_extractor.py --url "%vidurl%"

echo.
echo Done! Output saved.
echo.

goto loop

:end
echo Exiting...
pause

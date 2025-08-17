@echo off
title Tab Extractor
:loop
echo.
set /p vidurl=Paste YouTube link (or type EXIT to quit): 

if /i "%vidurl%"=="exit" goto end

python tab_extractor.py --url "%vidurl%"

echo.
echo Done! Output saved.
echo.

goto loop

:end
echo Exiting...

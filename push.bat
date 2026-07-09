@echo off
echo ===================================================
echo   Pushing Smart Portfolio Upgrades to GitHub
echo ===================================================
echo Uploading files...
git -c credential.helper= push origin main
echo ===================================================
echo   Push completed! You can close this window now.
echo ===================================================
pause

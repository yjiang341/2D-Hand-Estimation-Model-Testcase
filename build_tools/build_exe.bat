@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

echo Installing/Updating build dependency...
python -m pip install --upgrade pyinstaller

echo Building GUI executable...
python -m PyInstaller --noconfirm --clean --distpath build_tools\dist --workpath build_tools\build build_tools\HandPoseAudioBridge.spec

echo Build complete. Output folder:
echo %PROJECT_ROOT%\build_tools\dist\HandPoseAudioBridge
endlocal

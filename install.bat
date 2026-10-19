@echo off
rem flysim installer for Windows 10/11 (64-bit). Safe to run again: finished steps are skipped.
rem
rem   install.bat              the simulator, the local mind (Qwen3-4B + llama.cpp) and the connectome data
rem   install.bat --voice      also the optional voice server (Kokoro text-to-speech, Whisper speech-to-text, CPU)
rem   install.bat --cuda13     force the llama.cpp CUDA 13 build (default when the driver is 580 or newer)
rem   install.bat --cuda12     force the llama.cpp CUDA 12 build (drivers 551.61 to 579)
rem   install.bat --cpu        force the llama.cpp CPU build (no NVIDIA GPU; the mind runs slowly)
rem   install.bat --dry-run    check Python, disk, GPU and every download URL; install nothing
rem
rem Needs: Python 3.12 (64-bit), an internet connection, about 7 GB free (9 GB with --voice).
rem Set FLYSIM_PYTHON=C:\path\to\python.exe to use a particular Python 3.12.
setlocal EnableExtensions
cd /d "%~dp0"
title flysim install
set "ROOT=%~dp0"
rem pyshim stops Python's platform module from querying WMI, which hangs on some machines (see README)
set "PYTHONPATH=%ROOT%pain_server\pyshim"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set VOICE=0
set DRY=0
set BUILD=

:args
if "%~1"=="" goto args_done
if /i "%~1"=="--voice" set VOICE=1
if /i "%~1"=="--dry-run" set DRY=1
if /i "%~1"=="--cpu" set BUILD=cpu
if /i "%~1"=="--cuda12" set BUILD=cuda12
if /i "%~1"=="--cuda13" set BUILD=cuda13
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h" goto usage
shift
goto args
:args_done

echo.
echo === 1/7  Python 3.12 (64-bit)
set "PY="
if defined FLYSIM_PYTHON set "PY=%FLYSIM_PYTHON%"
if defined PY goto py_check
for /f "usebackq delims=" %%p in (`py -3.12 -c "import sys; print(sys.executable)" 2^>nul`) do set "PY=%%p"
if defined PY goto py_check
for /f "usebackq delims=" %%p in (`python -c "import sys; print(sys.executable)" 2^>nul`) do set "PY=%%p"
if not defined PY goto no_python
:py_check
"%PY%" -c "import sys, struct; sys.exit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 else 1)" 2>nul
if errorlevel 1 goto no_python
for /f "usebackq delims=" %%v in (`call "%PY%" -c "import sys; print(sys.version.split()[0])"`) do echo Python %%v at %PY%

echo.
echo === 2/7  Free disk space
set NEED=7
if "%VOICE%"=="1" set NEED=9
"%PY%" tools\fetch.py disk %NEED%
if errorlevel 1 goto fail_disk

echo.
echo === 3/7  GPU and llama.cpp build
if defined BUILD goto build_known
for /f "usebackq delims=" %%b in (`call "%PY%" tools\fetch.py detect`) do set "BUILD=%%b"
:build_known
echo llama.cpp build: %BUILD%
if "%BUILD%"=="cpu" echo No usable NVIDIA GPU or driver: the mind and the brain will run on the CPU, which is slow.

if "%DRY%"=="1" goto dry

echo.
echo === 4/7  Simulator environment (.venv)
if exist ".venv\Scripts\python.exe" goto app_venv_ok
"%PY%" -m venv .venv
if errorlevel 1 goto fail_venv
:app_venv_ok
".venv\Scripts\python.exe" -m pip install -r requirements\app.txt -c requirements\app-constraints.txt
if errorlevel 1 goto fail_pip
echo Fetching the FlyBody meshes flygym uses (134 MB, cached in %%USERPROFILE%%\.cache\flygym_assets) ...
".venv\Scripts\python.exe" -c "from flygym.utils.assets_lazy_loading import lazy_load_asset_dir; from flygym.compose.fly.flybody import FLYBODY_FULLSIZE_MESH_DIR; print(lazy_load_asset_dir(FLYBODY_FULLSIZE_MESH_DIR))"
if errorlevel 1 goto fail_meshes

echo.
echo === 5/7  Mind server environment (.venv_mind)
if exist ".venv_mind\Scripts\python.exe" goto mind_venv_ok
"%PY%" -m venv .venv_mind
if errorlevel 1 goto fail_venv
:mind_venv_ok
".venv_mind\Scripts\python.exe" -m pip install -r requirements\mind.txt
if errorlevel 1 goto fail_pip

echo.
echo === 6/7  Downloads: Qwen3-4B Q4_K_M (2.5 GB), llama.cpp %BUILD%, FlyWire connectome (135 MB)
"%PY%" tools\fetch.py get model %BUILD% data
if errorlevel 1 goto fail_download

echo.
echo === 7/7  Voice server (optional)
if not "%VOICE%"=="1" goto no_voice
if exist ".venv_voice\Scripts\python.exe" goto voice_venv_ok
"%PY%" -m venv .venv_voice
if errorlevel 1 goto fail_venv
:voice_venv_ok
".venv_voice\Scripts\python.exe" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail_pip
".venv_voice\Scripts\python.exe" -m pip install -r requirements\voice.txt -c requirements\voice-constraints.txt
if errorlevel 1 goto fail_pip
".venv_voice\Scripts\python.exe" tools\fetch_voice_models.py
if errorlevel 1 goto fail_download
goto done
:no_voice
echo skipped (install.bat --voice adds it)

:done
echo.
echo Installed. Start it with run.bat.
echo The first start compiles GPU kernels and loads the model: allow 2 to 5 minutes.
goto end

:dry
echo.
echo === Dry run: the downloads (nothing is written)
"%PY%" tools\fetch.py get model %BUILD% data --dry-run
echo.
echo Would create .venv, .venv_mind and run:
echo   .venv\Scripts\python.exe -m pip install -r requirements\app.txt -c requirements\app-constraints.txt
echo   .venv_mind\Scripts\python.exe -m pip install -r requirements\mind.txt
if not "%VOICE%"=="1" goto end
echo   .venv_voice\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
echo   .venv_voice\Scripts\python.exe -m pip install -r requirements\voice.txt -c requirements\voice-constraints.txt
echo   .venv_voice\Scripts\python.exe tools\fetch_voice_models.py
goto end

:usage
findstr /b /c:"rem " "%~f0"
goto end

:no_python
echo.
echo Python 3.12 (64-bit) was not found.
if defined PY echo   %PY% is another version; flysim's pinned packages are tested on 3.12 only.
echo   Install it with:  winget install -e --id Python.Python.3.12
echo   or from https://www.python.org/downloads/windows/ (the "Windows installer (64-bit)").
echo   Then run install.bat again. To use a Python 3.12 that is not on PATH:
echo   set FLYSIM_PYTHON=C:\path\to\python.exe
goto fail
:fail_disk
echo Free some space on this drive, or move the flysim folder to a drive with more room.
goto fail
:fail_venv
echo Could not create a virtual environment with %PY%.
goto fail
:fail_pip
echo pip failed (the message is above). Check the internet connection and run install.bat again.
goto fail
:fail_meshes
echo Could not download the FlyBody meshes from datasets.epfl.ch. Check the connection and run install.bat again.
goto fail
:fail_download
echo A download failed (the message is above). Run install.bat again: finished files are kept and partial ones resume.
goto fail
:fail
echo.
echo Install stopped.
endlocal
pause
exit /b 1
:end
endlocal
pause
exit /b 0

@echo off
rem flysim: starts the mind server (Qwen3-4B on llama.cpp, port 8095), the voice server if installed (port 8097),
rem then the simulator (http://127.0.0.1:8765), which opens its own browser window when the brain is built.
rem The servers run in this same window (their output goes to logs\). Close the window, or press Ctrl+C, to stop
rem everything.
rem
rem   run.bat                  local mind on the GPU (CPU when there is no NVIDIA GPU)
rem   run.bat --cpu            brain and mind on the CPU (slow; for machines without a usable GPU)
rem   run.bat --openrouter     the mind is an online model through OpenRouter instead of the local Qwen3-4B
rem                            (needs OPENROUTER_API_KEY; the fly's drives reach it only as text, not as steering)
rem   run.bat --no-voice       do not start the voice server
rem   Anything else is passed to the simulator, e.g.  run.bat --world labyrinth
rem   The motor layer defaults to bind (the mind's chosen intent drives the legs); run.bat --l1 off overrides it
rem
rem Settings (set NAME=value in this window first, or setx NAME value once):
rem   FLYSIM_GPU=cuda:0        the card for the brain and the mind
rem   FLYSIM_BACKEND=numba     brain on the CPU, which leaves the GPU to the mind
rem   FLYSIM_MIND_DEVICE=cpu   mind on the CPU, which leaves the GPU to the brain
rem   FLYSIM_MIND_PORT=8095    mind server port
rem   PAIN_THREADS=4           CPU threads for the mind
rem   OPENROUTER_API_KEY=...   for --openrouter;  FLYSIM_L2_MODEL=<OpenRouter model id> picks the model
setlocal EnableExtensions
cd /d "%~dp0"
title flysim
set "ROOT=%~dp0"
rem pyshim stops Python's platform module from querying WMI, which hangs on some machines (see README)
set "PYTHONPATH=%ROOT%pain_server\pyshim"
set MIND=local
set FORCECPU=0
set USEVOICE=1
set "APPARGS="

:args
if "%~1"=="" goto args_done
if /i "%~1"=="--cpu" (set FORCECPU=1& goto next)
if /i "%~1"=="--openrouter" (set MIND=openrouter& goto next)
if /i "%~1"=="--no-voice" (set USEVOICE=0& goto next)
set "APPARGS=%APPARGS% %1"
:next
shift
goto args
:args_done

if not exist ".venv\Scripts\python.exe" goto not_installed
if not exist "third_party\Drosophila_brain_model\Connectivity_783.parquet" goto not_installed

rem ---- where the brain runs
set HAVE_CUDA=0
nvidia-smi -L >nul 2>&1
if not errorlevel 1 set HAVE_CUDA=1
if "%FORCECPU%"=="1" set HAVE_CUDA=0
if "%FLYSIM_GPU%"=="" set FLYSIM_GPU=cuda:0
if "%FLYSIM_BACKEND%"=="" if "%HAVE_CUDA%"=="1" set FLYSIM_BACKEND=warp
if "%FLYSIM_BACKEND%"=="" set FLYSIM_BACKEND=numba
if "%FLYSIM_BACKEND%"=="numba" echo The brain runs on the CPU (numba).
if "%FLYSIM_BACKEND%"=="warp" echo The brain runs on %FLYSIM_GPU% (NVIDIA Warp).

if "%MIND%"=="openrouter" goto openrouter

rem ---- the local mind
if not exist ".venv_mind\Scripts\python.exe" goto not_installed
if not exist "models\Qwen3-4B-Q4_K_M.gguf" goto not_installed
if not exist "llama\llama.dll" goto not_installed
if "%FLYSIM_MIND_PORT%"=="" set FLYSIM_MIND_PORT=8095
set "FLYSIM_LOCAL_URL=http://127.0.0.1:%FLYSIM_MIND_PORT%/v1"
if "%FLYSIM_MIND_DEVICE%"=="" set "FLYSIM_MIND_DEVICE=%FLYSIM_GPU%"
if "%HAVE_CUDA%"=="0" set FLYSIM_MIND_DEVICE=cpu
if not exist "llama\ggml-cuda.dll" set FLYSIM_MIND_DEVICE=cpu
rem a thought on the CPU can take a minute: never time out on it
if "%FLYSIM_MIND_DEVICE%"=="cpu" set FLYSIM_NO_ABORT=1
if "%FLYSIM_MIND_DEVICE%"=="cpu" set FLYSIM_MIND_TIMEOUT=600
if "%FLYSIM_MIND_DEVICE%"=="cpu" set PAIN_CANCEL_JOIN_S=600
curl -s -m 2 %FLYSIM_LOCAL_URL%/models >nul 2>&1
if not errorlevel 1 goto mind_up
netstat -ano -p tcp | findstr /r /c:"127.0.0.1:%FLYSIM_MIND_PORT% .*LISTENING" /c:"0.0.0.0:%FLYSIM_MIND_PORT% .*LISTENING" >nul
if not errorlevel 1 goto mind_port_busy
echo Starting the mind server on %FLYSIM_MIND_DEVICE% (first load takes a minute or two; log: logs\mind_server.log) ...
start "" /b cmd /c ".venv_mind\Scripts\python.exe -u pain_server\server.py --backend llama --device %FLYSIM_MIND_DEVICE% --port %FLYSIM_MIND_PORT% > logs\mind_server.log 2>&1"
set tries=0
:mind_wait
timeout /t 3 /nobreak >nul
curl -s -m 2 %FLYSIM_LOCAL_URL%/models >nul 2>&1
if not errorlevel 1 goto mind_up
findstr /c:"Traceback" /c:"SystemExit" /c:"Error:" logs\mind_server.log >nul 2>&1
if not errorlevel 1 goto mind_died
set /a tries+=1
if %tries% GEQ 200 goto mind_timeout
goto mind_wait
:mind_up
echo The mind server is up on port %FLYSIM_MIND_PORT%.
goto voice

:openrouter
if "%OPENROUTER_API_KEY%%FLYSIM_L2_KEY%"=="" goto no_key
echo The mind is an online model through OpenRouter (model: %FLYSIM_L2_MODEL%; empty means the app default).

:voice
if "%USEVOICE%"=="0" goto app
if not exist ".venv_voice\Scripts\python.exe" goto app
curl -s -m 2 http://127.0.0.1:8097/health >nul 2>&1
if not errorlevel 1 goto app
echo Starting the voice server (log: logs\voice_server.log). Switch Voice on in the app's Mind tab or Settings.
start "" /b cmd /c ".venv_voice\Scripts\python.exe -u voice\fly_voice_server.py > logs\voice_server.log 2>&1"

:app
echo Starting the simulator: building the body and loading 138,639 neurons takes about 15-60 s.
echo The browser window opens when it is ready. Close this window, or press Ctrl+C, to stop everything.
".venv\Scripts\python.exe" -m flysim.app --backend %FLYSIM_BACKEND% --device %FLYSIM_GPU% --mind %MIND% --l2 steer --l1 bind %APPARGS%
set APPERR=%errorlevel%
if "%APPERR%"=="0" goto quit
echo The simulator stopped with an error; the message is above.
pause
:quit
endlocal & exit /b %APPERR%

:not_installed
echo flysim is not fully installed: run install.bat first.
pause
exit /b 1
:no_key
echo --openrouter needs an OpenRouter API key. In this window:
echo   set OPENROUTER_API_KEY=your-key
echo   run.bat --openrouter
echo (or setx OPENROUTER_API_KEY your-key once, then open a new window). Keys: https://openrouter.ai/keys
pause
exit /b 1
:mind_port_busy
echo Port %FLYSIM_MIND_PORT% is taken by another program that is not the flysim mind server.
echo Close it (netstat -ano ^| findstr :%FLYSIM_MIND_PORT% shows its PID), or pick another port:
echo   set FLYSIM_MIND_PORT=8096
pause
exit /b 1
:mind_died
echo The mind server stopped while loading. The last lines of logs\mind_server.log:
powershell -NoProfile -Command "Get-Content -Tail 25 'logs\mind_server.log'"
pause
exit /b 1
:mind_timeout
echo The mind server did not answer within 10 minutes; see logs\mind_server.log
pause
exit /b 1

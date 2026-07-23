@echo off
setlocal EnableExtensions EnableDelayedExpansion

title Instalador Venv Krea-2 Trainer - NVIDIA / Krea-2 Trainer Venv Installer - NVIDIA

:: ========================================================
:: CONFIGURACION / CONFIGURATION
:: ========================================================

set "BASE_DIR=%~dp0"
set "PYTHON_INSTALLER=%BASE_DIR%python-3.13.1-amd64.exe"
set "PYTHON_EXE="

echo ========================================================
echo   INSTALADOR KREA-2 LORA TRAINER / KREA-2 LORA TRAINER INSTALLER
echo   Entorno Python 3.13.1 + PyTorch CUDA / Python 3.13.1 + PyTorch CUDA Env
echo   Compatible con GPUs NVIDIA modernas / Compatible with modern NVIDIA GPUs
echo ========================================================
echo.
echo Carpeta del instalador / Installer folder:
echo %BASE_DIR%
echo.

echo [1/8] Comprobando Python 3.13 / Checking Python 3.13...

where python >nul 2>&1
if errorlevel 1 goto FIND_LOCAL_PYTHON

echo Python encontrado en PATH / Python found in PATH:
python --version
echo.
echo Comprobando version compatible / Checking compatible version...

python -c "import sys; exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1
if errorlevel 1 goto NOT_313_IN_PATH

echo [OK] Python 3.13 detectado / Python 3.13 detected.
set "PYTHON_EXE=python"
goto PYTHON_OK

:NOT_313_IN_PATH
echo [ADVERTENCIA/WARNING] Python encontrado pero no es 3.13 / Python found but it is not 3.13.

:: --------------------------------------------------------
:: Buscar Python 3.13 en ubicaciones habituales / Search Python 3.13 in common paths
:: --------------------------------------------------------

:FIND_LOCAL_PYTHON
echo.
echo Buscando instalaciones existentes de Python 3.13... / Searching for existing Python 3.13 installations...

if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
    echo [OK] Encontrado / Found:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

if exist "%ProgramFiles%\Python313\python.exe" (
    set "PYTHON_EXE=%ProgramFiles%\Python313\python.exe"
    echo [OK] Encontrado / Found:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

if exist "%ProgramFiles(x86)%\Python313\python.exe" (
    set "PYTHON_EXE=%ProgramFiles(x86)%\Python313\python.exe"
    echo [OK] Encontrado / Found:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

:: ========================================================
:: INSTALAR PYTHON 3.13.1 / INSTALL PYTHON 3.13.1
:: ========================================================

echo.
echo [INFO] Python 3.13 no esta disponible en el sistema / Python 3.13 is not available on the system.
echo.

if exist "%PYTHON_INSTALLER%" goto DO_INSTALL

echo ========================================================
echo   ANALIZANDO E INICIANDO DESCARGA AUTOMATICA / ANALYZING AND STARTING AUTO DOWNLOAD
echo ========================================================
echo.
echo Analizando arquitectura del sistema... / Analyzing system architecture...

rem PyTorch requiere un sistema operativo de 64 bits (AMD64 / x64)
set "ARCH=amd64"
if "%PROCESSOR_ARCHITECTURE%"=="x86" (
    if not defined PROCESSOR_ARCHITEW6432 (
        echo [ERROR] Sistema de 32 bits detectado / 32-bit system detected.
        echo PyTorch y CUDA requieren obligatoriamente 64 bits / PyTorch and CUDA strictly require 64 bits.
        pause
        exit /b 1
    )
)

echo Arquitectura compatible detectada: 64 bits %PROCESSOR_ARCHITECTURE% / Compatible architecture detected: 64 bits %PROCESSOR_ARCHITECTURE%
echo Descargando Python 3.13.1 x64 desde el sitio oficial... / Downloading Python 3.13.1 x64 from official site...
echo.

rem Intento de descarga 1: curl
curl -L "https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe" -o "%PYTHON_INSTALLER%"

if exist "%PYTHON_INSTALLER%" goto DOWNLOAD_OK

rem Intento de descarga 2: PowerShell
echo [INFO] curl fallo. Intentando con PowerShell... / curl failed. Trying with PowerShell...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe' -OutFile '%PYTHON_INSTALLER%'" >nul 2>&1

if not exist "%PYTHON_INSTALLER%" (
    echo ========================================================
    echo [ERROR] No se pudo descargar Python / Could not download Python.
    echo ========================================================
    echo No ha sido posible descargar automaticamente / Automatic download failed.
    echo Descarguelo manualmente desde su navegador / Download manually from browser:
    echo https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe
    echo Guardelo como "python-3.13.1-amd64.exe" junto a este archivo BAT / Save it as "python-3.13.1-amd64.exe" next to this BAT file.
    echo.
    pause
    exit /b 1
)

:DOWNLOAD_OK
echo [OK] Descargado correctamente / Successfully downloaded: %PYTHON_INSTALLER%
echo.

:DO_INSTALL
echo ========================================================
echo   INSTALANDO PYTHON 3.13.1 / INSTALLING PYTHON 3.13.1
echo ========================================================
echo.
echo El instalador se ejecutara de fondo / Installer will run in the background.
echo Se instalara para el usuario actual anadiendose al PATH / Installing for current user and adding to PATH.
echo Esto puede tardar unos minutos. Espere... / This may take a few minutes. Please wait...
echo.

"%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 Include_test=0

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo instalar Python 3.13.1 / Could not install Python 3.13.1.
    echo.
    pause
    exit /b 1
)

echo [OK] Instalacion de Python finalizada / Python installation finished.
echo.

:: --------------------------------------------------------
:: Buscar Python instalado despues de la instalacion
:: --------------------------------------------------------

if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
    goto PYTHON_OK
)

if exist "%ProgramFiles%\Python313\python.exe" (
    set "PYTHON_EXE=%ProgramFiles%\Python313\python.exe"
    goto PYTHON_OK
)

for /f "delims=" %%A in ('where python 2^>nul') do (
    set "PYTHON_EXE=%%A"
    goto PYTHON_OK
)

echo.
echo [ERROR] Python instalado pero no localizado / Python installed but not located.
echo Reinicie Windows y vuelva a ejecutar / Restart Windows and run again.
echo.
pause
exit /b 1

:: ========================================================
:: PYTHON OK
:: ========================================================

:PYTHON_OK

echo.
echo ========================================================
echo   PYTHON DETECTADO / PYTHON DETECTED
echo ========================================================
echo.
echo Ejecutable / Executable:
echo %PYTHON_EXE%
echo.

"%PYTHON_EXE%" --version

if errorlevel 1 (
    echo.
    echo [ERROR] Python no se ejecuta correctamente / Python cannot run properly.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import sys; exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] La version de Python no es compatible / Python version is not compatible.
    echo Se requiere Python 3.13.x / Python 3.13.x is required.
    echo.
    pause
    exit /b 1
)

echo [OK] Python 3.13 compatible detectado / Compatible Python 3.13 detected.

:: ========================================================
:: 2/8 - COMPROBAR VENV / CHECK VENV
:: ========================================================

echo.
echo [2/8] Comprobando modulo venv... / Checking venv module...

"%PYTHON_EXE%" -m venv --help >nul 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] El modulo venv de Python no esta disponible / Python venv module is not available.
    echo.
    pause
    exit /b 1
)

echo [OK] Modulo venv disponible / venv module available.

:: ========================================================
:: 3/8 - COMPROBAR GPU NVIDIA / CHECK NVIDIA GPU
:: ========================================================

echo.
echo [3/8] Comprobando compatibilidad NVIDIA... / Checking NVIDIA compatibility...
echo La deteccion de la GPU se realizara despues de instalar PyTorch / GPU detection will occur after PyTorch installation.
echo.
echo [OK] Continuando con la instalacion / Continuing installation.

echo.
echo [4/8] Preparando entorno virtual limpio... / Preparing clean virtual environment...

if exist "%BASE_DIR%venv" (
    echo.
    echo Entorno virtual anterior detectado. Eliminando... / Previous virtual environment detected. Removing...

    rmdir /s /q "%BASE_DIR%venv"

    if exist "%BASE_DIR%venv" (
        echo.
        echo [ERROR] No se pudo eliminar el venv anterior / Could not remove previous venv.
        echo Cierra cualquier programa en uso / Close any running program:
        echo     venv\Scripts\python.exe
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Creando nuevo entorno virtual... / Creating new virtual environment...

"%PYTHON_EXE%" -m venv "%BASE_DIR%venv"

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo crear el entorno virtual / Could not create virtual environment.
    pause
    exit /b 1
)

:: ========================================================
:: 5/8 - ACTIVAR VENV / ACTIVATE VENV
:: ========================================================

echo.
echo [5/8] Activando entorno virtual... / Activating virtual environment...

call "%BASE_DIR%venv\Scripts\activate.bat"

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo activar el entorno virtual / Could not activate virtual environment.
    pause
    exit /b 1
)

set "VENV_PYTHON=%BASE_DIR%venv\Scripts\python.exe"

echo.
echo Python del entorno virtual / Virtual environment Python:
echo %VENV_PYTHON%

"%VENV_PYTHON%" --version

:: ========================================================
:: 6/8 - ACTUALIZAR HERRAMIENTAS / UPGRADE TOOLS
:: ========================================================

echo.
echo [6/8] Actualizando pip, setuptools y wheel... / Upgrading pip, setuptools, and wheel...

"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudieron actualizar las herramientas / Could not upgrade Python tools.
    pause
    exit /b 1
)

:: ========================================================
:: 7/8 - INSTALAR PYTORCH / INSTALL PYTORCH
:: ========================================================

echo.
echo [7/8] Instalando PyTorch con soporte CUDA... / Installing PyTorch with CUDA support...
echo.
echo ========================================================
echo   IMPORTANTE / IMPORTANT
echo ========================================================
echo.
echo Se instalara PyTorch con CUDA 13.0 / PyTorch with CUDA 13.0 will be installed.
echo Para GPUs NVIDIA modernas (incluyendo RTX 50xx / Blackwell) / For modern NVIDIA GPUs (including RTX 50xx / Blackwell).
echo La instalacion puede tardar minutos / Installation may take minutes.
echo ========================================================
echo.

"%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo instalar PyTorch / Could not install PyTorch.
    pause
    exit /b 1
)

:: ========================================================
:: INSTALAR DEPENDENCIAS KREA-2 / INSTALL KREA-2 DEPENDENCIES
:: ========================================================

echo.
echo Instalando Diffusers, Transformers, PEFT, Accelerate, Safetensors y Hugging Face Hub...
echo Installing Diffusers, Transformers, PEFT, Accelerate, Safetensors, and Hugging Face Hub...

"%VENV_PYTHON%" -m pip install diffusers transformers peft accelerate safetensors huggingface_hub

if errorlevel 1 (
    echo.
    echo [ERROR] Error instalando dependencias / Error installing Diffusers dependencies.
    pause
    exit /b 1
)

echo.
echo Instalando BitsAndBytes y utilidades... / Installing BitsAndBytes and utilities...

"%VENV_PYTHON%" -m pip install bitsandbytes sentencepiece protobuf

if errorlevel 1 (
    echo.
    echo [ERROR] Error instalando BitsAndBytes o utilidades / Error installing BitsAndBytes or utilities.
    pause
    exit /b 1
)

:: ========================================================
:: 8/8 - COMPROBACION FINAL / FINAL CHECK
:: ========================================================

echo.
echo [8/8] Verificando instalacion completa... / Verifying installation...
echo.

:: --------------------------------------------------------
:: PYTORCH
:: --------------------------------------------------------

echo ========================================================
echo PyTorch
echo ========================================================

"%VENV_PYTHON%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA compilada:', torch.version.cuda); print('CUDA disponible:', torch.cuda.is_available()); print('GPUs detectadas:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA')"
if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch no se inicializo correctamente / PyTorch failed to initialize.
    pause
    exit /b 1
)

:: --------------------------------------------------------
:: DIFFUSERS
:: --------------------------------------------------------

echo.
echo ========================================================
echo Diffusers
echo ========================================================

"%VENV_PYTHON%" -c "import diffusers; print('Diffusers:', diffusers.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Diffusers no esta instalado correctamente / Diffusers is not installed properly.
    pause
    exit /b 1
)

echo [OK] Diffusers cargado correctamente / Diffusers loaded successfully.

:: --------------------------------------------------------
:: HUGGING FACE HUB
:: --------------------------------------------------------

echo.
echo ========================================================
echo Hugging Face Hub
echo ========================================================

"%VENV_PYTHON%" -c "import huggingface_hub; print('Hugging Face Hub:', huggingface_hub.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Hugging Face Hub no esta instalado correctamente / Hugging Face Hub is not installed properly.
    pause
    exit /b 1
)

echo [OK] Hugging Face Hub cargado correctamente / Hugging Face Hub loaded successfully.

:: --------------------------------------------------------
:: TRANSFORMERS
:: --------------------------------------------------------

echo.
echo ========================================================
echo Transformers
echo ========================================================

"%VENV_PYTHON%" -c "import transformers; print('Transformers:', transformers.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Transformers no esta instalado correctamente / Transformers is not installed properly.
    pause
    exit /b 1
)

echo [OK] Transformers cargado correctamente / Transformers loaded successfully.

:: --------------------------------------------------------
:: PEFT
:: --------------------------------------------------------

echo.
echo ========================================================
echo PEFT
echo ========================================================

"%VENV_PYTHON%" -c "import peft; print('PEFT:', peft.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] PEFT no esta instalado correctamente / PEFT is not installed properly.
    pause
    exit /b 1
)

echo [OK] PEFT cargado correctamente / PEFT loaded successfully.

:: --------------------------------------------------------
:: ACCELERATE
:: --------------------------------------------------------

echo.
echo ========================================================
echo Accelerate
echo ========================================================

"%VENV_PYTHON%" -c "import accelerate; print('Accelerate:', accelerate.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Accelerate no esta instalado correctamente / Accelerate is not installed properly.
    pause
    exit /b 1
)

echo [OK] Accelerate cargado correctamente / Accelerate loaded successfully.

:: --------------------------------------------------------
:: BITSANDBYTES
:: --------------------------------------------------------

echo.
echo ========================================================
echo BitsAndBytes
echo ========================================================

"%VENV_PYTHON%" -c "import bitsandbytes as bnb; print('BitsAndBytes:', bnb.__version__)"

if errorlevel 1 (
    echo.
    echo [ADVERTENCIA/WARNING] BitsAndBytes no se inicializo / BitsAndBytes failed to initialize.
    echo Esto puede afectar a optimizadores 8-bit / This may affect 8-bit optimizers.
    echo.
) else (
    echo [OK] BitsAndBytes cargado correctamente / BitsAndBytes loaded successfully.
)

:: --------------------------------------------------------
:: COMPROBAR CUDA / CHECK CUDA
:: --------------------------------------------------------

echo.
echo ========================================================
echo   RESULTADO DE LA COMPROBACION / CHECK RESULT
echo ========================================================
echo.

"%VENV_PYTHON%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

if errorlevel 1 (
    echo [ADVERTENCIA/WARNING] PyTorch NO detecta una GPU CUDA / PyTorch does NOT detect a CUDA GPU.
    echo.
    echo Posibles causas / Possible causes:
    echo - Driver NVIDIA antiguo / Outdated NVIDIA driver.
    echo - Instalacion incorrecta de PyTorch / Incorrect PyTorch installation.
    echo - Problema de GPU / GPU driver issue.
    echo.
    echo Ejecuta / Run: nvidia-smi
    echo.
) else (
    echo [OK] PyTorch detecta correctamente la GPU NVIDIA / PyTorch correctly detects NVIDIA GPU.
)

:: ========================================================
:: FINAL
:: ========================================================

echo.
echo ========================================================
echo   INSTALACION COMPLETADA / INSTALLATION COMPLETED
echo ========================================================
echo.
echo El entorno virtual "venv" ha sido creado / Virtual environment "venv" created.
echo.
echo Python utilizado / Python used:
echo %PYTHON_EXE%
echo.
echo Python del entorno virtual / Virtual env Python:
echo %VENV_PYTHON%
echo.
echo Version de Python / Python version:
"%VENV_PYTHON%" --version
echo.
echo GPU detectada por PyTorch / GPU detected by PyTorch:
"%VENV_PYTHON%" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DETECTADA / NOT DETECTED')"
echo.
echo Version CUDA de PyTorch / PyTorch CUDA version:
"%VENV_PYTHON%" -c "import torch; print(torch.version.cuda)"
echo.
echo Version Diffusers / Diffusers version:
"%VENV_PYTHON%" -c "import diffusers; print(diffusers.__version__)"
echo.
echo Version Hugging Face Hub / Hugging Face Hub version:
"%VENV_PYTHON%" -c "import huggingface_hub; print(huggingface_hub.__version__)"
echo.
echo ========================================================
echo.
echo El entorno esta listo para ejecutar Krea-2 Trainer.
echo Environment is ready to run Krea-2 Trainer.
echo.
pause

endlocal
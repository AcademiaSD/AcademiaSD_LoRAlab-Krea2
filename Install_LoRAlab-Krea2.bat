@echo off
setlocal EnableExtensions EnableDelayedExpansion

title Instalador Venv Krea-2 Trainer - NVIDIA

:: ========================================================
:: CONFIGURACION
:: ========================================================

set "BASE_DIR=%~dp0"
set "PYTHON_INSTALLER=%BASE_DIR%python-3.13.1-amd64.exe"
set "PYTHON_EXE="

echo ========================================================
echo   INSTALADOR KREA-2 LORA TRAINER
echo   Entorno Python 3.13.1 + PyTorch CUDA
echo   Compatible con GPUs NVIDIA modernas
echo ========================================================
echo.
echo Carpeta del instalador:
echo %BASE_DIR%
echo.

echo [1/8] Comprobando Python 3.13...

where python >nul 2>&1
if errorlevel 1 goto FIND_LOCAL_PYTHON

echo Python encontrado en PATH:
python --version
echo.
echo Comprobando version compatible...

python -c "import sys; exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1
if errorlevel 1 goto NOT_313_IN_PATH

echo [OK] Python 3.13 detectado.
set "PYTHON_EXE=python"
goto PYTHON_OK

:NOT_313_IN_PATH
echo [ADVERTENCIA] Python encontrado, pero no es Python 3.13.

:: --------------------------------------------------------
:: Buscar Python 3.13 en ubicaciones habituales
:: --------------------------------------------------------

:FIND_LOCAL_PYTHON
echo.
echo Buscando instalaciones existentes de Python 3.13...

if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
    echo [OK] Encontrado:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

if exist "%ProgramFiles%\Python313\python.exe" (
    set "PYTHON_EXE=%ProgramFiles%\Python313\python.exe"
    echo [OK] Encontrado:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

if exist "%ProgramFiles(x86)%\Python313\python.exe" (
    set "PYTHON_EXE=%ProgramFiles(x86)%\Python313\python.exe"
    echo [OK] Encontrado:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

:: ========================================================
:: INSTALAR PYTHON 3.13.1 (DESCARGA AUTOMÁTICA SI FALTA)
:: ========================================================

echo.
echo [INFO] Python 3.13 no esta disponible en el sistema.
echo.

if exist "%PYTHON_INSTALLER%" goto DO_INSTALL

echo ========================================================
echo   ANALIZANDO E INICIANDO DESCARGA AUTOMATICA
echo ========================================================
echo.
echo Analizando arquitectura del sistema...

rem PyTorch requiere un sistema operativo de 64 bits (AMD64 / x64)
set "ARCH=amd64"
if "%PROCESSOR_ARCHITECTURE%"=="x86" (
    if not defined PROCESSOR_ARCHITEW6432 (
        echo [ERROR] Sistema de 32 bits detectado.
        echo PyTorch y CUDA requieren obligatoriamente un sistema de 64 bits.
        pause
        exit /b 1
    )
)

echo Arquitectura compatible detectada: 64 bits %PROCESSOR_ARCHITECTURE%
echo Descargando Python 3.13.1 x64 desde el repositorio oficial...
echo.

rem Intento de descarga 1: curl (integrado en Windows 10/11)
curl -L "https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe" -o "%PYTHON_INSTALLER%"

if exist "%PYTHON_INSTALLER%" goto DOWNLOAD_OK

rem Intento de descarga 2: PowerShell (en caso de que curl no este o falle)
echo [INFO] curl no se encuentra o ha fallado. Intentando con PowerShell...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe' -OutFile '%PYTHON_INSTALLER%'" >nul 2>&1

if not exist "%PYTHON_INSTALLER%" (
    echo ========================================================
    echo [ERROR] No se pudo descargar el instalador de Python.
    echo ========================================================
    echo No ha sido posible realizar la descarga de manera automatica.
    echo Descarguelo manualmente desde su navegador:
    echo https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe
    echo guardelo con el nombre "python-3.13.1-amd64.exe" junto a este archivo BAT y vuelva a ejecutarlo.
    echo.
    pause
    exit /b 1
)

:DOWNLOAD_OK
echo [OK] Descargado correctamente: %PYTHON_INSTALLER%
echo.

:DO_INSTALL
echo ========================================================
echo   INSTALANDO PYTHON 3.13.1
echo ========================================================
echo.
echo El instalador de Python se ejecutara automaticamente de fondo.
echo Se instalara para el usuario actual añadiendose al PATH.
echo.
echo Esto puede tardar unos minutos. Por favor, espere...
echo.

"%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 Include_test=0

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo instalar Python 3.13.1.
    echo.
    pause
    exit /b 1
)

echo [OK] Instalacion de Python finalizada.
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

:: Actualizar PATH de la sesion actual
for /f "delims=" %%A in ('where python 2^>nul') do (
    set "PYTHON_EXE=%%A"
    goto PYTHON_OK
)

echo.
echo [ERROR] Python se ha instalado pero no se ha podido localizar en el sistema.
echo.
echo Reinicie Windows y vuelva a ejecutar este instalador.
echo.
pause
exit /b 1

:: ========================================================
:: PYTHON OK
:: ========================================================

:PYTHON_OK

echo.
echo ========================================================
echo   PYTHON DETECTADO
echo ========================================================
echo.
echo Ejecutable:
echo %PYTHON_EXE%
echo.

"%PYTHON_EXE%" --version

if errorlevel 1 (
    echo.
    echo [ERROR] Python no puede ejecutarse correctamente.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import sys; exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] La version de Python no es compatible.
    echo.
    echo Este instalador requiere Python 3.13.x.
    echo.
    pause
    exit /b 1
)

echo [OK] Python 3.13 compatible detectado.

:: ========================================================
:: 2/8 - COMPROBAR VENV
:: ========================================================

echo.
echo [2/8] Comprobando modulo venv...

"%PYTHON_EXE%" -m venv --help >nul 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] El modulo venv de Python no esta disponible.
    echo.
    pause
    exit /b 1
)

echo [OK] Modulo venv disponible.

:: ========================================================
:: 3/8 - COMPROBAR GPU NVIDIA
:: ========================================================

echo.
echo [3/8] Comprobando compatibilidad NVIDIA mediante PyTorch...

echo.
echo La deteccion definitiva de la GPU se realizara
echo despues de instalar PyTorch y CUDA.
echo.
echo [OK] El instalador continuara con la instalacion.

echo.
echo [4/8] Preparando entorno virtual limpio...

if exist "%BASE_DIR%venv" (
    echo.
    echo Se ha encontrado un entorno virtual anterior.
    echo Eliminando venv anterior...

    rmdir /s /q "%BASE_DIR%venv"

    if exist "%BASE_DIR%venv" (
        echo.
        echo [ERROR] No se pudo eliminar el entorno virtual anterior.
        echo.
        echo Cierra cualquier programa que este utilizando:
        echo     venv\Scripts\python.exe
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Creando nuevo entorno virtual...

"%PYTHON_EXE%" -m venv "%BASE_DIR%venv"

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)

:: ========================================================
:: 5/8 - ACTIVAR VENV
:: ========================================================

echo.
echo [5/8] Activando entorno virtual...

call "%BASE_DIR%venv\Scripts\activate.bat"

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo activar el entorno virtual.
    pause
    exit /b 1
)

set "VENV_PYTHON=%BASE_DIR%venv\Scripts\python.exe"

echo.
echo Python del entorno virtual:
echo %VENV_PYTHON%

"%VENV_PYTHON%" --version

:: ========================================================
:: 6/8 - ACTUALIZAR HERRAMIENTAS
:: ========================================================

echo.
echo [6/8] Actualizando pip, setuptools y wheel...

"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudieron actualizar las herramientas de Python.
    pause
    exit /b 1
)

:: ========================================================
:: 7/8 - INSTALAR PYTORCH
:: ========================================================

echo.
echo [7/8] Instalando PyTorch con soporte CUDA...
echo.
echo ========================================================
echo   IMPORTANTE
echo ========================================================
echo.
echo Se instalara PyTorch con CUDA 13.0.
echo.
echo Esta configuracion esta pensada para GPUs NVIDIA modernas,
echo incluyendo RTX 50xx / Blackwell.
echo.
echo La instalacion puede tardar varios minutos.
echo ========================================================
echo.

"%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo instalar PyTorch.
    pause
    exit /b 1
)

:: ========================================================
:: INSTALAR DEPENDENCIAS KREA-2
:: ========================================================

echo.
echo Instalando Diffusers, Transformers, PEFT, Accelerate,
echo Safetensors y Hugging Face Hub...

"%VENV_PYTHON%" -m pip install diffusers transformers peft accelerate safetensors huggingface_hub

if errorlevel 1 (
    echo.
    echo [ERROR] Error instalando las dependencias de Diffusers.
    pause
    exit /b 1
)

echo.
echo Instalando BitsAndBytes y utilidades...

"%VENV_PYTHON%" -m pip install bitsandbytes sentencepiece protobuf

if errorlevel 1 (
    echo.
    echo [ERROR] Error instalando BitsAndBytes o utilidades.
    pause
    exit /b 1
)

:: ========================================================
:: 8/8 - COMPROBACION FINAL
:: ========================================================

echo.
echo [8/8] Verificando instalacion completa...
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
    echo [ERROR] PyTorch no ha podido inicializarse correctamente.
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
    echo [ERROR] Diffusers no esta instalado correctamente.
    pause
    exit /b 1
)

echo [OK] Diffusers cargado correctamente.

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
    echo [ERROR] Hugging Face Hub no esta instalado correctamente.
    pause
    exit /b 1
)

echo [OK] Hugging Face Hub cargado correctamente.

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
    echo [ERROR] Transformers no esta instalado correctamente.
    pause
    exit /b 1
)

echo [OK] Transformers cargado correctamente.

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
    echo [ERROR] PEFT no esta instalado correctamente.
    pause
    exit /b 1
)

echo [OK] PEFT cargado correctamente.

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
    echo [ERROR] Accelerate no esta instalado correctamente.
    pause
    exit /b 1
)

echo [OK] Accelerate cargado correctamente.

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
    echo [ADVERTENCIA] BitsAndBytes no ha podido inicializarse.
    echo Esto puede afectar al entrenamiento con optimizadores 8-bit.
    echo.
) else (
    echo [OK] BitsAndBytes cargado correctamente.
)

:: --------------------------------------------------------
:: COMPROBAR CUDA
:: --------------------------------------------------------

echo.
echo ========================================================
echo   RESULTADO DE LA COMPROBACION
echo ========================================================
echo.

"%VENV_PYTHON%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

if errorlevel 1 (
    echo [ADVERTENCIA] PyTorch NO detecta una GPU CUDA.
    echo.
    echo Posibles causas:
    echo - Driver NVIDIA demasiado antiguo.
    echo - Instalacion incorrecta de PyTorch.
    echo - Problema con el driver de la GPU.
    echo.
    echo Ejecuta:
    echo     nvidia-smi
    echo.
) else (
    echo [OK] PyTorch detecta correctamente la GPU NVIDIA.
)

:: ========================================================
:: FINAL
:: ========================================================

echo.
echo ========================================================
echo   INSTALACION COMPLETADA
echo ========================================================
echo.
echo El entorno virtual "venv" ha sido creado desde cero.
echo.
echo Python utilizado:
echo %PYTHON_EXE%
echo.
echo Python del entorno virtual:
echo %VENV_PYTHON%
echo.
echo Version de Python:
"%VENV_PYTHON%" --version
echo.
echo GPU detectada por PyTorch:
"%VENV_PYTHON%" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DETECTADA')"
echo.
echo Version CUDA de PyTorch:
"%VENV_PYTHON%" -c "import torch; print(torch.version.cuda)"
echo.
echo Version Diffusers:
"%VENV_PYTHON%" -c "import diffusers; print(diffusers.__version__)"
echo.
echo Version Hugging Face Hub:
"%VENV_PYTHON%" -c "import huggingface_hub; print(huggingface_hub.__version__)"
echo.
echo ========================================================
echo.
echo El entorno esta listo para ejecutar el Krea-2 Trainer.
echo.
pause

endlocal
@echo off
setlocal EnableExtensions

title AcademiaSD - Krea-2 LoRA Trainer

cd /d "%~dp0"

set "BASE_DIR=%~dp0"
set "PYTHON_EXE="

echo.
echo ================================================================
echo        ACADEMIASD - KREA-2 LORA TRAINER
echo ================================================================
echo.
echo Carpeta del entrenador:
echo %BASE_DIR%
echo.

rem ================================================================
rem Buscar el entorno virtual existente
rem ================================================================

if exist "%BASE_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BASE_DIR%.venv\Scripts\python.exe"
    goto :python_found
)

if exist "%BASE_DIR%venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BASE_DIR%venv\Scripts\python.exe"
    goto :python_found
)

if exist "%BASE_DIR%env\Scripts\python.exe" (
    set "PYTHON_EXE=%BASE_DIR%env\Scripts\python.exe"
    goto :python_found
)

if exist "%BASE_DIR%..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BASE_DIR%..\venv\Scripts\python.exe"
    goto :python_found
)

if exist "%BASE_DIR%..\ .venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BASE_DIR%..\ .venv\Scripts\python.exe"
    goto :python_found
)

echo.
echo [ERROR] No se ha encontrado el entorno virtual.
echo.
echo Se han buscado:
echo   %BASE_DIR%.venv\Scripts\python.exe
echo   %BASE_DIR%venv\Scripts\python.exe
echo   %BASE_DIR%env\Scripts\python.exe
echo   %BASE_DIR%..\venv\Scripts\python.exe
echo.
pause
exit /b 1


:python_found

echo Entorno Python encontrado:
echo %PYTHON_EXE%
echo.

if not exist "%BASE_DIR%server.py" (
    echo [ERROR] No existe server.py
    echo.
    pause
    exit /b 1
)

if not exist "%BASE_DIR%trainer_ui.html" (
    echo [ERROR] No existe trainer_ui.html
    echo.
    pause
    exit /b 1
)

echo Comprobando Python...
"%PYTHON_EXE%" --version

if errorlevel 1 (
    echo.
    echo [ERROR] No se puede ejecutar Python.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Iniciando servidor web...
echo ================================================================
echo.
echo Abre en el navegador:
echo.
echo     http://127.0.0.1:5000
echo.
echo Cierra esta ventana para detener el servidor.
echo.

"%PYTHON_EXE%" "%BASE_DIR%server.py"

echo.
echo ================================================================
echo El servidor ha finalizado.
echo ================================================================
pause

endlocal
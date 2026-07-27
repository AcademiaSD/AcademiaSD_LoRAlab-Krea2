@echo off
setlocal EnableExtensions EnableDelayedExpansion

title AcademiaSD - Krea-2 LoRA Trainer Updater
color 0B

:: Asegurar que trabajamos en la carpeta donde esta guardado este archivo
cd /d "%~dp0"

echo ================================================================
echo   ACADEMIASD - KREA-2 LORA TRAINER UPDATER
echo   [EN] Repository Update Utility
echo   [ES] Utilidad de Actualizacion del Repositorio
echo ================================================================
echo.

rem 1. Comprobar si Git esta instalado
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not available in PATH.
    echo [ERROR] Git no esta instalado o no esta disponible en el PATH.
    echo.
    echo [EN] Please install Git from https://git-scm.com/
    echo [ES] Por favor instala Git desde https://git-scm.com/
    echo.
    pause
    exit /b 1
)

rem 2. Comprobar si existe la carpeta .git en este directorio
if exist ".git" goto :DO_PULL

rem 3. Si no existe .git (cuando fue descargado en ZIP)
echo [WARNING] No '.git' folder found in this directory.
echo [ADVERTENCIA] No se encontro la carpeta '.git' en este directorio.
echo.
echo [EN] Cause: You likely downloaded the repository as a ZIP file.
echo [ES] Causa: Es probable que hayas descargado el proyecto en formato ZIP.
echo.
echo [EN] Do you want to convert this folder into a Git repository and update now?
echo [ES] Deseas convertir esta carpeta en un repositorio Git y actualizar ahora?
echo.

choice /C YN /M "[Y] Yes / Si  -  [N] No"

if errorlevel 2 exit /b 0

echo.
echo [EN] Initializing Git repository and connecting to GitHub...
echo [ES] Inicializando repositorio Git y conectando a GitHub...
echo.

git init
git remote add origin https://github.com/AcademiaSD/AcademiaSD_LoRAlab-Krea2.git
git fetch origin

rem Crear/vincular la rama principal (main o master) con seguimiento de GitHub
git checkout -B main origin/main >nul 2>&1
if errorlevel 1 (
    git checkout -B master origin/master >nul 2>&1
)

git reset --hard origin/main >nul 2>&1
if errorlevel 1 (
    git reset --hard origin/master >nul 2>&1
)

echo.
echo [OK] Repository converted and synchronized with GitHub!
echo [OK] Repositorio convertido y sincronizado con GitHub!
goto :FINISH

:DO_PULL
echo [EN] Pulling latest updates from GitHub...
echo [ES] Descargando ultimas actualizaciones desde GitHub...
echo.

rem Intento 1: git pull estandar
git pull >nul 2>&1

if %errorlevel% neq 0 (
    rem Intento 2: Si no hay seguimiento asignado, especificar la rama origen explícitamente
    git pull origin main
    if errorlevel 1 (
        git pull origin master
    )
)

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to update repository via 'git pull'.
    echo [ERROR] No se pudo actualizar el repositorio mediante 'git pull'.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo [OK] Repository updated successfully! / Repositorio actualizado!
echo ================================================================

:FINISH
echo.
echo [EN] Update process completed.
echo [ES] Proceso de actualizacion completado.
echo.
pause
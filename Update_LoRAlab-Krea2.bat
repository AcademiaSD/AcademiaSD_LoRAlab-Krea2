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

rem 2. Si no existe .git (instalacion ZIP), inicializar
if not exist ".git" (
    echo [INFO] Initializing Git repository...
    echo [INFO] Inicializando repositorio Git...
    echo.
    git init >nul 2>&1
    git remote add origin https://github.com/AcademiaSD/AcademiaSD_LoRAlab-Krea2.git >nul 2>&1
)

rem 3. Asegurar la URL remota correcta
git remote set-url origin https://github.com/AcademiaSD/AcademiaSD_LoRAlab-Krea2.git >nul 2>&1

echo [EN] Syncing latest updates from GitHub...
echo [ES] Sincronizando ultimas actualizaciones desde GitHub...
echo.

rem 4. Descargar cambios de GitHub
git fetch origin main >nul 2>&1
if errorlevel 1 (
    git fetch origin master >nul 2>&1
)

rem 5. Forzar sincronizacion para evitar conflictos de archivos no rastreados
git checkout -f main >nul 2>&1
if errorlevel 1 (
    git checkout -f master >nul 2>&1
)

git reset --hard origin/main >nul 2>&1
if errorlevel 1 (
    git reset --hard origin/master >nul 2>&1
)

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to update repository from GitHub.
    echo [ERROR] No se pudo actualizar el repositorio desde GitHub.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo [OK] Repository updated successfully! / Repositorio actualizado!
echo ================================================================
echo.
echo [EN] Update process completed.
echo [ES] Proceso de actualizacion completado.
echo.
pause
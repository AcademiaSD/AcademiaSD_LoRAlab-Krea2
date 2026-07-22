@echo off
title Triton and SageAttention Installer by Academia SD
color 0A

:: Define the portable Python path for ComfyUI
set PYTHON_EXE=venv\scripts\python.exe

echo =======================================================
echo   Triton ^& SageAttention Installer 
echo   Created by: Academia SD
echo =======================================================
echo.

:: Check if we are in the correct folder
if not exist "%PYTHON_EXE%" (
    color 0C
    echo [ERROR] %PYTHON_EXE% not found.
    echo Please place this .bat file in the root folder.
    pause
    exit /b
)

echo =======================================================
echo    1. INSTALLING TRITON 3.4.0.post20
echo =======================================================
"%PYTHON_EXE%" -m pip install -U triton-windows==3.4.0.post20

echo.
echo =======================================================
echo    2. ANALYZING ENVIRONMENT FOR SAGEATTENTION 2.2.0
echo =======================================================

:: Create a temporary Python script for advanced logic
echo import sys, subprocess > temp_detector.py
echo try: >> temp_detector.py
echo     import torch >> temp_detector.py
echo except ImportError: >> temp_detector.py
echo     print("[ERROR] PyTorch is not installed. Cannot continue.") >> temp_detector.py
echo     sys.exit(1) >> temp_detector.py
echo. >> temp_detector.py
echo py_ver = f"{sys.version_info.major}.{sys.version_info.minor}" >> temp_detector.py
echo t_ver = torch.__version__ >> temp_detector.py
echo c_ver = torch.version.cuda or "None" >> temp_detector.py
echo print(f"[INFO] Detected Environment:") >> temp_detector.py
echo print(f"       - Python: {py_ver}") >> temp_detector.py
echo print(f"       - PyTorch: {t_ver}") >> temp_detector.py
echo print(f"       - CUDA: {c_ver}\n") >> temp_detector.py
echo. >> temp_detector.py
echo # Extract minor version of PyTorch (e.g., if 2.9.0, extract 9) >> temp_detector.py
echo t_minor = 0 >> temp_detector.py
echo if t_ver.startswith("2."): >> temp_detector.py
echo     try: t_minor = int(t_ver.split(".")[1]) >> temp_detector.py
echo     except: pass >> temp_detector.py
echo. >> temp_detector.py
echo url = "" >> temp_detector.py
echo # Logic based on Github releases: >> temp_detector.py
echo if t_minor ^>= 9 and "13." in c_ver: >> temp_detector.py
echo     print("[INFO] Profile: PyTorch 2.9+ and CUDA 13.x (Compatible via abi3)") >> temp_detector.py
echo     url = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl" >> temp_detector.py
echo elif t_minor ^>= 9 and "12.8" in c_ver: >> temp_detector.py
echo     print("[INFO] Profile: PyTorch 2.9+ and CUDA 12.8 (Compatible via abi3)") >> temp_detector.py
echo     url = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl" >> temp_detector.py
echo elif t_minor == 8 and "12.8" in c_ver and py_ver == "3.13": >> temp_detector.py
echo     print("[INFO] Profile: Python 3.13, PyTorch 2.8 and CUDA 12.8") >> temp_detector.py
echo     url = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows/sageattention-2.2.0+cu128torch2.8.0-cp313-cp313-win_amd64.whl" >> temp_detector.py
echo else: >> temp_detector.py
echo     print("[WARNING] Your environment does not have an exact matching .whl binary in the releases.") >> temp_detector.py
echo     print("Skipping SageAttention installation.") >> temp_detector.py
echo     sys.exit(0) >> temp_detector.py
echo. >> temp_detector.py
echo print(f"\n[INFO] Downloading and installing SageAttention from the assigned URL...") >> temp_detector.py
echo subprocess.check_call([sys.executable, "-m", "pip", "install", url]) >> temp_detector.py

:: Execute the temporary script
"%PYTHON_EXE%" temp_detector.py

:: Clean up by deleting the temp file
del temp_detector.py

echo.
echo =======================================================
echo    PROCESS COMPLETED
echo =======================================================
pause
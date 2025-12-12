@echo off
chcp 65001 >nul
title SAM3 AN - 创建便携式环境

:: 切换到脚本所在目录
cd /d "%~dp0"

echo ============================================
echo   SAM3 AN - 创建便携式环境 (推荐方案)
echo ============================================
echo.
echo 此方案将创建一个独立的 Python 环境，
echo 包含所有依赖，可直接复制到其他电脑使用。
echo.
echo 优点:
echo   - 完美支持 CUDA/GPU 加速
echo   - 文件较小 (约 5-8GB)
echo   - 兼容性好
echo   - 易于更新和调试
echo.

set PORTABLE_DIR=SAM3_AN_Portable
set PYTHON_VERSION=3.10.11

:: 检查是否已存在
if not exist "%PORTABLE_DIR%" goto create_dirs

echo [警告] 目录 %PORTABLE_DIR% 已存在
set /p overwrite=是否删除并重新创建? (y/n):
if /i "%overwrite%"=="y" (
    echo [信息] 删除旧目录...
    rmdir /s /q "%PORTABLE_DIR%"
    goto create_dirs
)
echo [信息] 保留现有目录，跳过创建步骤
goto copy_files

:create_dirs
echo.
echo [步骤 1/5] 创建目录结构...
mkdir "%PORTABLE_DIR%"
mkdir "%PORTABLE_DIR%\python"
mkdir "%PORTABLE_DIR%\app"

:copy_files
echo.
echo [步骤 2/5] 复制应用文件...
xcopy /E /I /Y "templates" "%PORTABLE_DIR%\app\templates" >nul
xcopy /E /I /Y "static" "%PORTABLE_DIR%\app\static" >nul
xcopy /E /I /Y "services" "%PORTABLE_DIR%\app\services" >nul
xcopy /E /I /Y "exports" "%PORTABLE_DIR%\app\exports" >nul
xcopy /E /I /Y "SAM_src" "%PORTABLE_DIR%\app\SAM_src" >nul
copy /Y "app.py" "%PORTABLE_DIR%\app\" >nul
copy /Y "requirements.txt" "%PORTABLE_DIR%\app\" >nul
echo [完成] 应用文件已复制

:: 创建启动脚本
echo.
echo [步骤 3/5] 创建启动脚本...

echo @echo off > "%PORTABLE_DIR%\启动SAM3AN.bat"
echo chcp 65001 ^>nul >> "%PORTABLE_DIR%\启动SAM3AN.bat"
echo title SAM3 AN - 数据标注工具 >> "%PORTABLE_DIR%\启动SAM3AN.bat"
echo cd /d "%%~dp0" >> "%PORTABLE_DIR%\启动SAM3AN.bat"
echo set PYTHON_HOME=%%~dp0python >> "%PORTABLE_DIR%\启动SAM3AN.bat"
echo set PATH=%%PYTHON_HOME%%;%%PYTHON_HOME%%\Scripts;%%PATH%% >> "%PORTABLE_DIR%\启动SAM3AN.bat"
echo cd app >> "%PORTABLE_DIR%\启动SAM3AN.bat"
echo python app.py >> "%PORTABLE_DIR%\启动SAM3AN.bat"
echo pause >> "%PORTABLE_DIR%\启动SAM3AN.bat"

:: 创建安装依赖脚本
echo.
echo [步骤 4/5] 创建安装依赖脚本...

echo @echo off > "%PORTABLE_DIR%\安装依赖.bat"
echo chcp 65001 ^>nul >> "%PORTABLE_DIR%\安装依赖.bat"
echo title 安装依赖 >> "%PORTABLE_DIR%\安装依赖.bat"
echo cd /d "%%~dp0" >> "%PORTABLE_DIR%\安装依赖.bat"
echo set PYTHON_HOME=%%~dp0python >> "%PORTABLE_DIR%\安装依赖.bat"
echo set PATH=%%PYTHON_HOME%%;%%PYTHON_HOME%%\Scripts;%%PATH%% >> "%PORTABLE_DIR%\安装依赖.bat"
echo echo 正在安装依赖... >> "%PORTABLE_DIR%\安装依赖.bat"
echo pip install -r app\requirements.txt >> "%PORTABLE_DIR%\安装依赖.bat"
echo echo. >> "%PORTABLE_DIR%\安装依赖.bat"
echo echo 安装 PyTorch CUDA 12.6... >> "%PORTABLE_DIR%\安装依赖.bat"
echo pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 >> "%PORTABLE_DIR%\安装依赖.bat"
echo echo 完成! >> "%PORTABLE_DIR%\安装依赖.bat"
echo pause >> "%PORTABLE_DIR%\安装依赖.bat"

echo.
echo [步骤 5/5] 完成！
echo.
echo ============================================
echo 便携式环境已创建: %PORTABLE_DIR%\
echo.
echo 后续步骤:
echo   1. 复制你的 Python 环境到 %PORTABLE_DIR%\python\
echo      (conda环境或venv的整个文件夹)
echo   2. 或下载嵌入式Python:
echo      https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip
echo   3. 运行 "安装依赖.bat" 安装所需库
echo   4. 运行 "启动SAM3AN.bat" 启动程序
echo ============================================
echo.
pause

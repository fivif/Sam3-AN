@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

title SAM3 AN - 构建真正便携版

echo ╔════════════════════════════════════════════════════════════╗
echo ║         SAM3 AN - 构建真正便携版                           ║
echo ║         Python Embedded + 预装依赖                         ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

:: 设置变量
set "SCRIPT_DIR=%~dp0"
set "OUTPUT_DIR=%SCRIPT_DIR%SAM3_AN_Portable_Full"
set "PYTHON_VERSION=3.10.11"
set "PYTHON_EMBED_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
set "GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py"

:: 检查是否有 curl
where curl >nul 2>&1
if errorlevel 1 (
    echo [错误] 需要 curl 来下载文件
    echo 请确保 Windows 10 1803+ 或手动安装 curl
    pause
    exit /b 1
)

:: 创建输出目录
echo [步骤 1/8] 创建目录结构...
if exist "%OUTPUT_DIR%" (
    echo [警告] 目录已存在，是否删除重建？ (Y/N)
    set /p confirm=
    if /i "!confirm!"=="Y" (
        rmdir /s /q "%OUTPUT_DIR%"
    ) else (
        echo [取消] 用户取消操作
        pause
        exit /b 0
    )
)

mkdir "%OUTPUT_DIR%"
mkdir "%OUTPUT_DIR%\python"
mkdir "%OUTPUT_DIR%\app"

:: 下载 Python Embedded
echo.
echo [步骤 2/8] 下载 Python %PYTHON_VERSION% Embedded...
set "PYTHON_ZIP=%OUTPUT_DIR%\python-embed.zip"
curl -L -o "%PYTHON_ZIP%" "%PYTHON_EMBED_URL%"
if errorlevel 1 (
    echo [错误] 下载 Python Embedded 失败
    pause
    exit /b 1
)

:: 解压 Python
echo.
echo [步骤 3/8] 解压 Python Embedded...
powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%OUTPUT_DIR%\python' -Force"
del "%PYTHON_ZIP%"

:: 修改 python310._pth 以启用 site-packages
echo.
echo [步骤 4/8] 配置 Python 环境...
set "PTH_FILE=%OUTPUT_DIR%\python\python310._pth"
(
    echo python310.zip
    echo .
    echo Lib\site-packages
    echo import site
) > "%PTH_FILE%"

:: 创建 Lib\site-packages 目录
mkdir "%OUTPUT_DIR%\python\Lib\site-packages"

:: 下载并安装 pip
echo.
echo [步骤 5/8] 安装 pip...
curl -L -o "%OUTPUT_DIR%\python\get-pip.py" "%GET_PIP_URL%"
"%OUTPUT_DIR%\python\python.exe" "%OUTPUT_DIR%\python\get-pip.py" --no-warn-script-location
del "%OUTPUT_DIR%\python\get-pip.py"

:: 安装依赖
echo.
echo [步骤 6/8] 安装依赖 (这可能需要较长时间)...
echo.
echo [6.1] 安装 PyTorch CUDA 12.1...
"%OUTPUT_DIR%\python\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --no-warn-script-location

echo.
echo [6.2] 安装其他依赖...
"%OUTPUT_DIR%\python\python.exe" -m pip install flask flask-cors Werkzeug Pillow opencv-python "numpy==1.26" requests PyYAML pycocotools tqdm "timm>=1.0.17" "ftfy==6.1.1" regex "iopath>=0.1.10" typing_extensions huggingface_hub --no-warn-script-location

echo.
echo [6.3] 安装 triton-windows...
"%OUTPUT_DIR%\python\python.exe" -m pip install triton-windows --no-warn-script-location

:: 复制应用文件
echo.
echo [步骤 7/8] 复制应用文件...
xcopy /E /I /Y "%SCRIPT_DIR%SAM_src" "%OUTPUT_DIR%\app\SAM_src"
xcopy /E /I /Y "%SCRIPT_DIR%services" "%OUTPUT_DIR%\app\services"
xcopy /E /I /Y "%SCRIPT_DIR%exports" "%OUTPUT_DIR%\app\exports"
xcopy /E /I /Y "%SCRIPT_DIR%templates" "%OUTPUT_DIR%\app\templates"
xcopy /E /I /Y "%SCRIPT_DIR%static" "%OUTPUT_DIR%\app\static"
xcopy /E /I /Y "%SCRIPT_DIR%utils" "%OUTPUT_DIR%\app\utils" 2>nul
mkdir "%OUTPUT_DIR%\app\data"
mkdir "%OUTPUT_DIR%\app\uploads"
copy /Y "%SCRIPT_DIR%app.py" "%OUTPUT_DIR%\app\"
copy /Y "%SCRIPT_DIR%requirements.txt" "%OUTPUT_DIR%\app\"

:: 复制模型文件
echo.
echo [步骤 7.5] 复制模型文件 (约3.2GB，请耐心等待)...
if exist "%SCRIPT_DIR%sam3.pt" (
    copy /Y "%SCRIPT_DIR%sam3.pt" "%OUTPUT_DIR%\app\"
) else (
    echo [警告] 未找到 sam3.pt 模型文件，请手动复制
)

:: 创建启动脚本
echo.
echo [步骤 8/8] 创建启动脚本...

:: 主启动脚本
(
echo @echo off
echo chcp 65001 ^>nul
echo title SAM3 AN - 数据标注工具
echo.
echo cd /d "%%~dp0"
echo.
echo echo ╔════════════════════════════════════════════════════════════╗
echo echo ║              SAM3 AN - 智能数据标注工具                    ║
echo echo ║              正在启动，请稍候...                           ║
echo echo ╚════════════════════════════════════════════════════════════╝
echo echo.
echo.
echo :: 设置 Python 路径
echo set "PYTHON_HOME=%%~dp0python"
echo set "PATH=%%PYTHON_HOME%%;%%PYTHON_HOME%%\Scripts;%%PATH%%"
echo.
echo :: 启动应用
echo cd app
echo "%%PYTHON_HOME%%\python.exe" app.py
echo.
echo if errorlevel 1 ^(
echo     echo.
echo     echo [错误] 程序异常退出
echo     pause
echo ^)
) > "%OUTPUT_DIR%\启动SAM3AN.bat"

:: 创建 README
(
echo # SAM3 AN 便携版
echo.
echo ## 使用方法
echo.
echo 1. 双击 `启动SAM3AN.bat` 即可运行
echo 2. 浏览器会自动打开 http://localhost:5000
echo.
echo ## 系统要求
echo.
echo - Windows 10/11 64位
echo - NVIDIA GPU ^(推荐 8GB+ 显存^)
echo - CUDA 12.1 驱动 ^(需要安装 NVIDIA 驱动^)
echo.
echo ## 注意事项
echo.
echo - 首次启动需要加载模型，请耐心等待
echo - 如果没有 NVIDIA GPU，程序会使用 CPU 运行^(较慢^)
echo - 确保 sam3.pt 模型文件在 app 目录中
echo.
echo ## 目录结构
echo.
echo ```
echo SAM3_AN_Portable_Full/
echo ├── python/          # Python 运行环境 ^(已预装所有依赖^)
echo ├── app/             # 应用程序
echo │   ├── sam3.pt      # SAM3 模型文件
echo │   ├── SAM_src/     # SAM3 源码
echo │   └── ...
echo └── 启动SAM3AN.bat   # 启动脚本
echo ```
) > "%OUTPUT_DIR%\README.md"

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║                    构建完成！                              ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo 便携版位置: %OUTPUT_DIR%
echo.
echo 你可以将整个 SAM3_AN_Portable_Full 文件夹复制到任何
echo Windows 电脑上，双击 启动SAM3AN.bat 即可运行！
echo.
echo [提示] 目标电脑需要安装 NVIDIA 显卡驱动 (CUDA 12.1+)
echo.
pause

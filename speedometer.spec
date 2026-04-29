# speedometer.spec — PyInstaller build spec
# Usage: pyinstaller speedometer.spec --noconfirm

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Bundle ultralytics YAML configs and other package data
ul_datas = collect_data_files("ultralytics")
ul_hidden = collect_submodules("ultralytics")

# pywebview — collect all platform backends so the frozen app can import them
wv_datas = collect_data_files("webview")
wv_hidden = collect_submodules("webview")

# Include model files only if they exist in the project root
bundled_models = []
for model in ("yolo12s.pt", "yolov8n.pt", "yolo11s.pt"):
    if os.path.exists(model):
        bundled_models.append((model, "."))

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("templates", "templates"),
        ("static", "static"),
        ("speedcam", "speedcam"),
    ] + bundled_models + ul_datas + wv_datas,
    hiddenimports=ul_hidden + wv_hidden + [
        "flask",
        "jinja2",
        "werkzeug",
        "werkzeug.serving",
        "werkzeug.debug",
        "cv2",
        "numpy",
        "pandas",
        "PIL",
        "PIL.Image",
        "ultralytics",
        "torch",
        "torchvision",
        "torchvision.ops",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "IPython", "notebook", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Speedometer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # No console window — the UI is in the browser
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Speedometer",
)

# Mac: wrap the collected folder in a proper .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Speedometer.app",
        icon=None,
        bundle_identifier="com.speedometer.app",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSUIElement": False,
            "CFBundleShortVersionString": os.environ.get("APP_VERSION", "1.0.0"),
            "CFBundleName": "Speedometer",
            "CFBundleDisplayName": "Speedometer",
        },
    )

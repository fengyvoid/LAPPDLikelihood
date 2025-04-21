import os
import ast
import sys
import subprocess
from stdlib_list import stdlib_list

# 设置入口脚本路径
ENTRY_FILE = "mainMC_exp.py"
CODE_DIR = os.path.dirname(os.path.abspath(ENTRY_FILE))

# 获取当前 Python 的标准库模块列表
version = f"{sys.version_info.major}.{sys.version_info.minor}"
stdlib = set(stdlib_list(version))

# 所有需要分析的文件：入口文件 + 同目录下被 import 的自定义模块
files_to_scan = set()
scanned = set()

def find_imported_files(file):
    if file in scanned:
        return
    scanned.add(file)

    filepath = os.path.join(CODE_DIR, file)
    if not os.path.exists(filepath):
        return

    with open(filepath, "r") as f:
        tree = ast.parse(f.read(), filename=file)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split('.')[0]
                pyfile = f"{name}.py"
                if os.path.exists(os.path.join(CODE_DIR, pyfile)):
                    files_to_scan.add(pyfile)
                    find_imported_files(pyfile)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                name = node.module.split('.')[0]
                pyfile = f"{name}.py"
                if os.path.exists(os.path.join(CODE_DIR, pyfile)):
                    files_to_scan.add(pyfile)
                    find_imported_files(pyfile)

files_to_scan.add(os.path.basename(ENTRY_FILE))
find_imported_files(os.path.basename(ENTRY_FILE))

# 解析所有 import，收集非标准库模块
modules = set()
for file in files_to_scan:
    with open(os.path.join(CODE_DIR, file), "r") as f:
        tree = ast.parse(f.read(), filename=file)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split('.')[0])

# 排除标准库和你自己的脚本
external_modules = [m for m in modules if m not in stdlib and not os.path.exists(os.path.join(CODE_DIR, f"{m}.py"))]

# 安装所有第三方模块
print("📦 Installing the following modules:", external_modules)
for mod in external_modules:
    subprocess.call([sys.executable, "-m", "pip", "install", mod])


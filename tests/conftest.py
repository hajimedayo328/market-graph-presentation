import sys
from pathlib import Path

# scripts/lib と scripts を sys.path に追加
lib_path = Path(__file__).parent.parent / "scripts" / "lib"
scripts_path = Path(__file__).parent.parent / "scripts"

if str(lib_path) not in sys.path:
    sys.path.insert(0, str(lib_path))
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

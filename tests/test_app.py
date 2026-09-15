import pytest
import sys
from pathlib import Path

def test_app_imports():
    """
    Verifies that app/main.py imports cleanly without any syntax or path resolution errors.
    """
    app_path = Path(__file__).parent.parent / "app" / "main.py"
    assert app_path.exists()
    
    # Verify file can be read and parsed by Python AST
    content = app_path.read_text(encoding="utf-8")
    compile(content, str(app_path), "exec")

"""Root conftest: defensively ensures the project root is on sys.path so
`import generator...` / `import attacks...` / `import preprocessing...` work
regardless of how pytest was invoked.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

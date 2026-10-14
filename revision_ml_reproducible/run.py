"""Portable command-line entry point; independent of the current working directory."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT / "hd_separation_ml"))
sys.path.insert(0,str(ROOT / "workflow"))

if __name__=="__main__":
    from main import main
    main()

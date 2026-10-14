"""Copy the package outside the checkout and prohibit reads from the original repo."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
GUARD=r'''
import os, runpy, sys
entry, forbidden, *arguments=sys.argv[1:]
forbidden=os.path.normcase(os.path.realpath(forbidden))
def audit(event, values):
    if event != 'open' or not values or not isinstance(values[0],(str,bytes,os.PathLike)):
        return
    path=os.path.normcase(os.path.realpath(os.fsdecode(values[0])))
    try:
        inside=os.path.commonpath([path,forbidden])==forbidden
    except ValueError:
        inside=False
    if inside:
        raise RuntimeError('Attempt to read original repository: '+path)
sys.addaudithook(audit)
sys.argv=[entry,*arguments]
runpy.run_path(entry,run_name='__main__')
'''


def main():
    results=[]
    with tempfile.TemporaryDirectory(prefix="hd_ml_relocated_") as temporary:
        bundle=Path(temporary) / "standalone package"
        shutil.copytree(ROOT,bundle,ignore=shutil.ignore_patterns("runs","__pycache__",".pytest_cache",".venv"))
        commands=[("audit",["audit","--regenerate-splits","--output","runs/relocated_audit"]),
                  ("refit",["refit","--pool","33","--limit","1","--no-shap","--jobs","1","--output","runs/relocated_refit"])]
        for name,arguments in commands:
            call=subprocess.run([sys.executable,"-I","-c",GUARD,str(bundle / "run.py"),str(ROOT.parent),*arguments],
                cwd=temporary,capture_output=True,text=True,encoding="utf-8",errors="replace")
            print(call.stdout)
            if call.returncode:
                raise RuntimeError(call.stderr+"\n"+call.stdout)
            status=json.loads((bundle / "runs" / ("relocated_"+name) / "verification.json").read_text(encoding="utf-8"))
            results.append(dict(command=name,returncode=call.returncode,verification=status))
    output=ROOT / "validation"; output.mkdir(exist_ok=True)
    (output / "relocation.json").write_text(json.dumps(dict(passed=True,original_repository_reads_forbidden=True,
        isolated_python=True,working_directory_outside_package=True,package_path_contains_spaces=True,checks=results),indent=2)+"\n",encoding="utf-8")
    print("Relocation and original-repository isolation passed")


if __name__=="__main__":
    main()

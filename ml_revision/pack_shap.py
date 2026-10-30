"""Losslessly compress the generated SHAP table for repository distribution."""
import gzip
import hashlib
import shutil
from workflow.data import ROOT
from model_core import save

def main():
    source=ROOT/'results/shap.csv'
    if not source.exists():return
    target=source.with_suffix('.csv.gz')
    with source.open('rb') as i,target.open('wb') as o:
        with gzip.GzipFile(filename='',mode='wb',fileobj=o,mtime=0) as g:shutil.copyfileobj(i,g)
    with source.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    with gzip.open(target,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha
    save(ROOT/'checks/compression.json',dict(file='results/shap.csv.gz',uncompressed_sha256=sha,
        uncompressed_bytes=source.stat().st_size,compressed_bytes=target.stat().st_size))
    assert source.resolve()==(ROOT.resolve()/'results/shap.csv')
    source.unlink()

if __name__=='__main__':main()

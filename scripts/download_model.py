"""Download only the frozen, hash-verified model files used in this project."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'analysis/model_manifest.json').read_text())
    for item in manifest['files']:
        target=ROOT/'tmp/roberta'/item['name']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']:
            print('Verified cache:',item['name']);continue
        url=f"https://huggingface.co/{manifest['model']}/resolve/{manifest['revision']}/{item['name']}"
        target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_suffix(target.suffix+'.part');digest=hashlib.sha256()
        request=urllib.request.Request(url,headers={'User-Agent':'Academic sentiment replication'})
        with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as handle:
            while chunk:=response.read(1024*1024):handle.write(chunk);digest.update(chunk)
        if digest.hexdigest()!=item['sha256']:raise ValueError('Model hash mismatch: '+item['name'])
        partial.replace(target);print('Downloaded and verified:',item['name'])

if __name__=='__main__':main()

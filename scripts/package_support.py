"""Package reproducible results without redistributing full X post bodies."""
from pathlib import Path
import sys,json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/python_deps'))
import pandas as pd

def main():
    destination=ROOT/'output/AI_Risk_Sentiment_2026_Support.zip'
    hashes=[]
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        def put(name,content):
            if isinstance(content,str):content=content.encode('utf-8')
            name=str(name).replace('\\','/')
            archive.writestr(name,content)
            hashes.append({'file':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
        for name in ['README.md','requirements.txt','.gitignore']:
            put(name,(ROOT/name).read_bytes())
        for path in sorted((ROOT/'scripts').glob('*')):
            if path.suffix in ['.py','.ps1']:put(path.relative_to(ROOT),path.read_bytes())
        paths=list((ROOT/'analysis').glob('*.csv'))+[
            ROOT/'data/prices.csv',ROOT/'data/news_candidates.csv',ROOT/'data/x/stratified_posts.csv',
            ROOT/'data/x/exposure_filter_audit.csv']
        for path in sorted(paths):
            frame=pd.read_csv(path,dtype={'id':str})
            # Normalized text is also a copy of comment bodies and must be removed.
            remove=[c for c in frame.columns if c in ['text','normalized','normalized_text','comment_text']]
            frame=frame.drop(columns=remove)
            put(path.relative_to(ROOT),frame.to_csv(index=False))
        for path in sorted((ROOT/'analysis').glob('*.json')):
            put(path.relative_to(ROOT),path.read_bytes())
        for name in ['retrieval_manifest.json','prices_errors.json','news_errors.json']:
            put('data/'+name,(ROOT/'data'/name).read_bytes())
        for path in sorted((ROOT/'data/x').glob('*.json')):
            put(path.relative_to(ROOT),path.read_bytes())
        anchor=json.loads((ROOT/'data/anchor_verification.json').read_text(encoding='utf-8'))
        put('data/anchor_metadata.json',json.dumps({
            'id':anchor.get('id_str'),'created_at_utc':anchor['created_at'],
            'verified_ny_time':anchor['verified_ny_time'],
            'url':'https://x.com/EvanHub/status/2097497037956891126'},indent=2))
        for path in sorted((ROOT/'data/raw/opinion_lexicon').glob('*.txt')):
            put(path.relative_to(ROOT),path.read_bytes())
        for path in sorted((ROOT/'output/figures').glob('*.png')):
            put(path.relative_to(ROOT),path.read_bytes())
        put('PACKAGE_NOTES.txt',
            'News/prices collected October 1, 2026; X retrieved October 2. The report is supplied separately.\n'
            'Text and normalized-text columns have been removed from distributed CSV files.\n'
            'Headlines, IDs, URLs, metadata and computed scores remain available.\n'
            'Raw article HTML, X responses, post bodies, API keys, libraries and model weights are excluded.\n'
            'X exposure thresholds: at least 100 followers AND at least 100 views at retrieval; missing counts excluded.\n'
            'Primary X sample uses fixed five-minute windows with verified since_id/max_id boundaries.\n'
            'Failed epoch queries and diagnostic costs are disclosed but excluded from the primary sample.\n'
            'Refetch news/prices with collect_data.py and X with the explicitly budget-capped collect_x.py.\n'
            'The supplied scores allow numerical refitting without paid recollection; live archives may change.\n'
            'Hu-Liu lexicon files retain their original license/comment headers.\n'
            'Model files can be downloaded and hash-verified with download_model.py.\n')
        archive.writestr('PACKAGE_MANIFEST.json',json.dumps(hashes,indent=2))
    print('PACKAGED',destination,'files',len(hashes)+1,'bytes',destination.stat().st_size)

if __name__=='__main__':main()

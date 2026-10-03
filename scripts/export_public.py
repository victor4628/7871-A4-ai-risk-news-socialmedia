"""Publish social metadata and scores with all full-text copies removed."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/python_deps'))
import pandas as pd

def main():
    public=ROOT/'data/public';public.mkdir(exist_ok=True)
    paths={'social_candidates.csv':ROOT/'data/x/stratified_posts.csv',
        **{name:ROOT/'analysis'/name for name in ['social_scored.csv','social_roberta.csv',
           'social_unfiltered_scored.csv','social_unfiltered_roberta.csv','review_sample.csv']}}
    for name,path in paths.items():
        frame=pd.read_csv(path,dtype={'id':str,'author_id':str,'conversation_id':str,'story_id':str})
        frame=frame.drop(columns=[c for c in frame if c in ['text','normalized','normalized_text','comment_text']])
        frame.to_csv(public/name,index=False,encoding='utf-8-sig')
    manifest_path=ROOT/'data/retrieval_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    (ROOT/'tmp/hn_previous').mkdir(parents=True,exist_ok=True)
    (ROOT/'tmp/hn_previous/retrieval_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manifest=[entry for entry in manifest if '/social/' not in entry['file'].replace('\\','/')]
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Exported six body-free social metadata/score files; removed legacy social retrieval entries from the distributed manifest.')

if __name__=='__main__':main()

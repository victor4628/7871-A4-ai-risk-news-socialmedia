"""Audit an explicit X exposure filter using retrieval-time metadata.

The raw time-stratified sample is preserved. Thresholds do not change search
queries or trigger extra paid requests. Missing counts are excluded from the
exposure-qualified sample and disclosed separately. These measurements are
post-period selection variables, not historically available predictors.
"""
from pathlib import Path
import argparse,csv,json,sys,html,re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/python_deps'))
import pandas as pd
from collect_x import backfill_cached_metadata

def clean_for_analysis(posts):
    """Use the same relevance/text rules before comparing exposure groups."""
    posts=posts.copy();stages={'unique_sampled':len(posts)}
    posts['text']=posts.text.fillna('').map(html.unescape)
    relevant=posts.text.str.contains(r'(?<!\w)(?:AI|artificial intelligence|superintelligence)(?!\w)',case=False,regex=True)
    posts=posts[relevant].copy();stages['explicit_ai_term']=len(posts)
    words=posts.text.map(lambda text:len(re.sub(r'https?://\S+|(?<!\w)@\w+',' ',text).split()))
    posts=posts[words>=5].copy();stages['at_least_five_non_url_non_mention_tokens']=len(posts)
    posts['normalized']=posts.text.str.lower().str.replace(r'\s+',' ',regex=True).str.strip()
    posts=posts.drop_duplicates(['author_id','normalized']).copy();stages['unique_author_text']=len(posts)
    stages['exposure_qualified_after_common_cleaning']=int(posts.exposure_eligible.sum())
    stages['limitations']='Keyword relevance excludes compound hashtags/aliases without an explicit AI term; token count is not a semantic relevance guarantee.'
    return posts,stages

def read_posts():
    manifest=json.loads((ROOT/'data/x/stratified_manifest.json').read_text(encoding='utf-8'))
    with (ROOT/'data/x/stratified_posts.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows={row['id']:row for row in csv.DictReader(handle)}
    backfill_cached_metadata(rows,manifest)
    return pd.DataFrame(rows.values()),manifest

def apply_filter(posts,min_followers=100,min_views=100):
    if min_followers<0 or min_views<0:raise ValueError('Thresholds must be nonnegative.')
    posts=posts.copy()
    for col in ['author_followers_at_retrieval','views_at_retrieval']:
        posts[col]=pd.to_numeric(posts[col],errors='coerce')
        if (posts[col].dropna()<0).any():raise ValueError('Negative exposure count.')
    followers=posts.author_followers_at_retrieval
    views=posts.views_at_retrieval
    missing=followers.isna()|views.isna()
    low_followers=followers.lt(min_followers)
    low_views=views.lt(min_views)
    reason=pd.Series('retained',index=posts.index)
    reason.loc[low_followers&~low_views]='low_followers'
    reason.loc[low_views&~low_followers]='low_views'
    reason.loc[low_views&low_followers]='low_followers_and_views'
    reason.loc[missing]='missing_exposure_metadata'
    posts['exposure_filter_reason']=reason
    posts['exposure_eligible']=reason.eq('retained')
    daily=posts.groupby('date').agg(sampled=('id','size'),retained=('exposure_eligible','sum'))
    audit={'min_followers':min_followers,'min_views':min_views,
        'threshold_logic':'Both thresholds must be met; equality passes.',
        'metadata_timing':'Values recorded at API retrieval, not at posting or the historical trading close.',
        'missing_policy':'Excluded from exposure-qualified sample; retained in raw sample and separately counted.',
        'sampled':len(posts),'retained':int(posts.exposure_eligible.sum()),
        'reason_counts':reason.value_counts().to_dict(),
        'daily':daily.reset_index().to_dict('records'),
        'limitation':'Exposure filtering changes the target population and may favor large accounts, promotional content, or viral warnings. Compare with the unfiltered sample.'}
    return posts,audit

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-followers',type=int,default=100)
    parser.add_argument('--min-views',type=int,default=100)
    args=parser.parse_args()
    posts,manifest=read_posts()
    posts,audit=apply_filter(posts,args.min_followers,args.min_views)
    audit['collection_completed']=manifest['completed']
    out=ROOT/'data/x'
    audit_cols=['id','date','author_id','author_followers_at_retrieval','views_at_retrieval',
                'metrics_retrieved_utc','sample_window','exposure_eligible','exposure_filter_reason']
    posts[audit_cols].to_csv(out/'exposure_filter_audit.csv',index=False,encoding='utf-8-sig')
    (out/'exposure_filter_summary.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='daily'},indent=2))

if __name__=='__main__':main()

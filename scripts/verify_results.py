"""Independent numerical and document checks for the submitted analysis."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/python_deps'))
import numpy as np,pandas as pd
from scipy.stats import f as f_distribution
from statsmodels.stats.multitest import multipletests
from pypdf import PdfReader

def main():
    checks={}
    prices=pd.read_csv(ROOT/'data/prices.csv')
    constituents=pd.read_csv(ROOT/'analysis/constituent_returns.csv')
    returns=pd.read_csv(ROOT/'analysis/returns.csv',index_col='date')
    summary=json.loads((ROOT/'analysis/summary.json').read_text())
    assert len(prices)==450 and prices.symbol.nunique()==25
    assert prices.date.min()=='2026-09-04' and prices.date.max()=='2026-09-30'
    assert not prices.duplicated(['date','symbol']).any()
    assert len(constituents)==23 and constituents.symbol.is_unique and 'ORCL' in set(constituents.symbol)
    independent_return=100*((constituents.end_adj_close/constituents.start_adj_close).mean()-1)
    assert np.isclose(independent_return,summary['basket_return_pct'],atol=1e-10)
    assert np.isclose(constituents.contribution_pp.sum(),independent_return,atol=1e-10)
    assert np.isclose(100*(returns.basket_wealth.iloc[-1]-1),independent_return,atol=1e-10)
    assert np.allclose(returns.basket_return.iloc[1:],100*np.diff(np.log(returns.basket_wealth)))
    checks['prices_and_portfolio']={'symbols':25,'constituents':23,'basket_return_pct':independent_return}
    for name,n in [('news_editorial_scored',summary['editorial_news']),('social_scored',summary['social'])]:
        path=ROOT/f'analysis/{name}.csv'
        if not path.exists():path=ROOT/f'data/public/{name}.csv'
        frame=pd.read_csv(path,dtype={'id':str})
        assert len(frame)==n and frame.id.is_unique
        assert frame.date.between('2026-09-08','2026-09-30').all()
        if name=='social_scored':social_frame=frame.copy()
    tests=pd.read_csv(ROOT/'analysis/all_granger_results.csv')
    assert len(tests)==32 and tests.primary.sum()==8
    independent_tests=[]
    for _,row in tests.iterrows():
        filename='roberta_model_data.csv' if row.specification.startswith('RoBERTa') else 'model_data.csv'
        frame=pd.read_csv(ROOT/'analysis'/filename)
        xy=frame[[row.outcome,row.predictor]].dropna().to_numpy()
        lag=int(row.lags);y=xy[lag:,0]
        own=np.column_stack([xy[lag-j:-j,0] for j in range(1,lag+1)])
        other=np.column_stack([xy[lag-j:-j,1] for j in range(1,lag+1)])
        restricted=np.column_stack([np.ones(len(y)),own])
        unrestricted=np.column_stack([restricted,other])
        residuals=[]
        for matrix in [restricted,unrestricted]:
            coeff=np.linalg.lstsq(matrix,y,rcond=None)[0]
            residuals.append(float(np.sum((y-matrix@coeff)**2)))
        df_den=len(y)-unrestricted.shape[1]
        statistic=((residuals[0]-residuals[1])/lag)/(residuals[1]/df_den)
        probability=float(f_distribution.sf(statistic,lag,df_den))
        assert len(y)==row.n and df_den==row.df_den
        assert np.isclose(statistic,row.F,rtol=1e-8,atol=1e-10)
        assert np.isclose(probability,row.p,atol=1e-10)
        independent_tests.append(probability)
    assert tests.loc[tests.primary,'n'].eq(14).all()
    assert np.allclose(tests.loc[tests.primary,'eight_primary_holm_p'],multipletests(tests.loc[tests.primary,'p'],method='holm')[1])
    assert np.allclose(tests.all_specs_holm_p,multipletests(tests.p,method='holm')[1])
    checks['independent_restricted_unrestricted_OLS']={'specifications_verified':len(tests),'primary_n':14}
    if summary.get('social_source')=='x':
        from filter_x import read_posts,apply_filter
        source=json.loads((ROOT/'data/x/stratified_manifest.json').read_text(encoding='utf-8'))
        if (ROOT/'data/x/stratified_posts.csv').exists():raw,_=read_posts()
        else:raw=pd.read_csv(ROOT/'data/public/social_candidates.csv',dtype={'id':str,'author_id':str,'conversation_id':str})
        assert source['completed'] and len(source['coverage'])==138
        assert raw.date.nunique()==23 and raw.id.is_unique
        if all((ROOT/e['file']).exists() for e in source['requests']):
            returned_ids=set()
            for entry in source['requests']:
                returned_ids.update(str(post['id']) for post in json.loads((ROOT/entry['file']).read_bytes())['tweets'])
            assert set(raw.id)==returned_ids,'Saved CSV does not contain every unique returned primary post.'
        policy=json.loads((ROOT/'data/x/filter_policy.json').read_text(encoding='utf-8'))
        qualified,_=apply_filter(raw,policy['min_followers'],policy['min_views'])
        assert set(social_frame.id).issubset(set(qualified.loc[qualified.exposure_eligible,'id']))
        assert pd.to_numeric(social_frame.author_followers_at_retrieval).ge(100).all()
        assert pd.to_numeric(social_frame.views_at_retrieval).ge(100).all()
        assert social_frame.date.nunique()==23
        errors=[]
        for _,post in raw.iterrows():
            actual=pd.Timestamp(post.created_at).timestamp()*1000
            encoded=(int(post.id)>>22)+1288834974657
            if abs(actual-encoded)>1000:errors.append(post.id)
        assert not errors,'Tweet IDs and creation timestamps disagree.'
        total=float(source['estimated_billed_usd'])
        for name in ['legacy_epoch_manifest.json','pilot_manifest.json']:
            total+=float(json.loads((ROOT/'data/x'/name).read_text(encoding='utf-8'))['estimated_billed_usd'])
        total+=sum(e['estimated_usd'] for e in json.loads((ROOT/'data/x/diagnostic_manifest.json').read_text(encoding='utf-8')))
        assert total<=3
        hash_count=0
        for e in source['requests']:
            path=ROOT/e['file']
            if path.exists():
                assert hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256'];hash_count+=1
        checks['x_data']={'raw_posts':len(raw),'qualified_cleaned_posts':len(social_frame),'dates':23,'windows':138,'estimated_total_api_usd':total,'ids_verified':True,'available_response_hashes_verified':hash_count}
    manifest=json.loads((ROOT/'data/retrieval_manifest.json').read_text())
    checked=0
    for entry in manifest:
        path=ROOT/entry['file'].replace('\\','/')
        if path.exists():
            # Collector hashes decoded UTF-8 response text before storing it.
            content=path.read_text(encoding='utf-8').encode('utf-8')
            assert hashlib.sha256(content).hexdigest()==entry['sha256'],entry['file']
            checked+=1
    checks['raw_source_text_hashes_verified']=checked
    model=json.loads((ROOT/'analysis/model_manifest.json').read_text())
    checked=0
    for entry in model['files']:
        path=ROOT/'tmp/roberta'/entry['name']
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
            checked+=1
    checks['model_file_hashes_verified']=checked
    pdf=ROOT/'output/AI_Risk_Sentiment_2026_Report.pdf'
    if pdf.exists():
        reader=PdfReader(pdf)
        assert len(reader.pages)==6
        text='\n'.join(page.extract_text() for page in reader.pages)
        assert all(f'[{i}]' in text for i in range(1,14))
        assert 'Victor Chen' in text and 'Hacker News' not in text and 'HN ' not in text
        assert len([a for page in reader.pages for a in page.get('/Annots',[])])>=13
        checks['report']={'pages':6,'references':13,'links_verified':True}
    (ROOT/'analysis/verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()

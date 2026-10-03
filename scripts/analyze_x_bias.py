"""Compare exposure-selected X sentiment with the preserved unfiltered sample."""
from pathlib import Path
import json,sys,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/python_deps'))
import numpy as np,pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from analyze import bin_to_close,test_pair,NY,topic
OUT=ROOT/'analysis'
PROMOTION=re.compile(r'\b(?:airdrop|presale|giveaway|affiliate|promo code|buy now|free trial|sign[ -]?up|subscribe|join (?:our|the)|dm (?:me|us))\b',re.I)

def main():
    frames={name:pd.read_csv(OUT/f'{name}_roberta.csv',dtype={'id':str,'author_id':str,'conversation_id':str})
            for name in ['social','social_unfiltered']}
    rows=[]
    for name,df in frames.items():
        df['rb_negative']=df.roberta_tone.eq('negative').astype(int)
        df['promotion_cue']=df.text.str.contains(PROMOTION).astype(int)
        df['automated_flag']=df.author_automated_flag.astype(str).str.lower().eq('true')
        df['phase']=np.select([df.date<='2026-09-14',df.date<='2026-09-22'],['Sep 8-14','Sep 15-22'],default='Sep 23-30')
        q=df.groupby('phase').agg(n=('id','size'),vader_negative=('negative','mean'),roberta_negative=('rb_negative','mean'),
            mean_vader=('compound','mean'),mean_roberta=('roberta_score','mean'),risk_share=('risk_theme','mean'),
            existential_share=('existential_theme','mean'),promotion_cues=('promotion_cue','mean'))
        q['sample']=name;rows.append(q.reset_index())
    pd.concat(rows).to_csv(OUT/'x_filter_phase_comparison.csv',index=False)
    full=frames['social_unfiltered'];selected=frames['social']
    full['qualified']=full.id.isin(selected.id).astype(int)
    full['topic']=full.text.map(topic);full['log_words']=np.log1p(full.words)
    regressions=[]
    for model,y in [('VADER','negative'),('RoBERTa','rb_negative')]:
        for variable in ['qualified','promotion_cue']:
            fit=smf.ols(f'{y} ~ {variable} + C(date) + C(topic) + log_words + I(log_words**2)',full).fit(
                cov_type='cluster',cov_kwds={'groups':full.author_id.fillna(full.author)})
            ci=fit.conf_int().loc[variable]
            regressions.append({'model':model,'comparison':variable,'coefficient':fit.params[variable],
                'low':ci.iloc[0],'high':ci.iloc[1],'p':fit.pvalues[variable],'n':int(fit.nobs)})
    pd.DataFrame(regressions).to_csv(OUT/'x_selection_regressions.csv',index=False)
    coverage=json.loads((ROOT/'data/x/stratified_manifest.json').read_text(encoding='utf-8'))
    author_counts=selected.author.value_counts();thread_counts=selected.story_id.value_counts()
    summary={'unfiltered_cleaned_posts':len(full),'qualified_cleaned_posts':len(selected),
        'retention_rate_after_common_cleaning':len(selected)/len(full),
        'authors':int(selected.author.nunique()),'conversations':int(selected.story_id.nunique()),
        'top10_author_share':float(author_counts.head(10).sum()/len(selected)),
        'top10_conversation_share':float(thread_counts.head(10).sum()/len(selected)),
        'primary_windows':len(coverage['coverage']),'capped_windows':sum(r['capped'] for r in coverage['coverage']),
        'time_coverage_fraction':30/1440,
        'qualified_automated_flag_share':float(selected.automated_flag.mean()),
        'qualified_promotion_cue_share':float(selected.promotion_cue.mean()),
        'unfiltered_promotion_cue_share':float(full.promotion_cue.mean()),
        'vader_negative_qualified':float(selected.negative.mean()),'vader_negative_unfiltered':float(full.negative.mean()),
        'roberta_negative_qualified':float(selected.rb_negative.mean()),'roberta_negative_unfiltered':float(full.rb_negative.mean()),
        'daily_qualified_min':int(selected.groupby('date').size().min()),
        'daily_qualified_max':int(selected.groupby('date').size().max()),
        'promotion_rule':PROMOTION.pattern,
        'promotion_limitation':'Lexical commercial cues are a proxy, not verified advertising, spam or bot labels.',
        'exposure_limitation':'Exposure counts are retrospective selection variables; primary X results do not establish feasible real-time prediction.'}
    aligned=pd.read_csv(OUT/'model_data.csv',index_col='date')
    rb_aligned=pd.read_csv(OUT/'roberta_model_data.csv',index_col='date')
    returns=pd.read_csv(OUT/'returns.csv',index_col='date')
    closes=pd.DatetimeIndex([pd.Timestamp(d+' 16:00',tz=NY) for d in returns.index if d>='2026-09-08'])
    extra=[]
    for model,col,target in [('VADER','compound',aligned),('RoBERTa','roberta_score',rb_aligned)]:
        for name,df in [('unfiltered',full),('automated_excluded',selected[~selected.automated_flag])]:
            df=df.copy();df['close_bin']=bin_to_close(df.created_at,closes)
            signal=df.groupby('close_bin')[col].mean().reindex(target.index)
            x=f'delta_x_{name}_{model.lower()}'
            target[x]=signal.diff()
            extra.append(test_pair(target,'basket_return',x,1,f'{model} X {name} change -> basket'))
            if name=='unfiltered':extra.append(test_pair(target,x,'basket_return',1,f'{model} basket -> X unfiltered change'))
    aligned.to_csv(OUT/'model_data.csv');rb_aligned.to_csv(OUT/'roberta_model_data.csv')
    tests=pd.read_csv(OUT/'all_granger_results.csv')
    tests=tests[~tests.specification.str.contains('X unfiltered|X automated_excluded')]
    tests=pd.concat([tests,pd.DataFrame(extra)],ignore_index=True)
    tests['all_specs_holm_p']=multipletests(tests.p,method='holm')[1]
    tests['eight_primary_holm_p']=np.nan
    tests.loc[tests.primary,'eight_primary_holm_p']=multipletests(tests.loc[tests.primary,'p'],method='holm')[1]
    tests.to_csv(OUT/'all_granger_results.csv',index=False)
    summary['all_test_specifications']=len(tests)
    summary['sensitivity_tests']=pd.DataFrame(extra).to_dict('records')
    (OUT/'x_selection_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));print(pd.DataFrame(regressions).to_string(index=False))

if __name__=='__main__':main()

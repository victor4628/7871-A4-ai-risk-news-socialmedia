"""Write repository documentation from the completed X study results."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/python_deps'))
import pandas as pd

def main():
    summary=json.loads((ROOT/'analysis/summary.json').read_text())
    x=json.loads((ROOT/'analysis/x_selection_summary.json').read_text())
    checks=json.loads((ROOT/'analysis/verification.json').read_text())
    assert summary['social_source']=='x' and checks['report']['pages']==6
    tests=pd.read_csv(ROOT/'analysis/all_granger_results.csv')
    text=f'''# AI risk coverage and equity returns

**Victor Chen** | [GitHub repository](https://github.com/victor4628/7871-A4-ai-risk-news-socialmedia)

This completed study analyzes September 8–30, 2026 in America/New_York time using news headlines and **Twitter/X posts**. The English report is six pages including references, with an editable Word version. Its forest-green theme is Roc Investments `#254C45`. Data, scripts and all fitted specifications are supplied separately. The fixed basket contains 23 US-listed AI equities, including Oracle, plus SPY and QQQ benchmarks.

## Data and findings

- News: {summary['editorial_news']:,} deduplicated editorial headlines from {summary['news_domains']} domains, plus {summary['corporate_news']} company/research releases used only as a framing comparator.
- Twitter/X: {summary['raw_social']:,} unique sampled posts from 138 fixed search windows. The confirmed exposure thresholds require **followers ≥100 AND views ≥100**; short-text and duplicate-author/text cleaning leave **{summary['social']:,} posts from {summary['social_authors']:,} authors**, with {x['daily_qualified_min']}–{x['daily_qualified_max']} posts per date. All 23 dates are represented.
- The AI basket returns **{summary['basket_return_pct']:+.2f}%**, SPY {summary['SPY_return_pct']:+.2f}% and QQQ {summary['QQQ_return_pct']:+.2f}%. These are buy-and-hold total returns from the prior close on September 4 through September 30.
- Eight baseline Granger tests cover both directions, two channels and two sentiment models. All **{len(tests)} specifications**, including sensitivities, are disclosed with Holm corrections. Consult the report and `analysis/all_granger_results.csv` for conclusions and uncertainty; Granger prediction does not establish structural causation.
- X filtering, commercial-language cues and news-source tone differences are measured explicitly. Associations do not prove unjustified exaggeration of AI risk.

## Sources and sampling

**News.** All entries in DataAIHub's 23 daily archives were collected and deduplicated by URL and normalized title. This curated archive is a convenience sample rather than a media census. First archive appearance is the availability date, not a guaranteed original publication timestamp. Headline tone is analyzed; article bodies are not scored. Company and research releases are excluded from the primary news signal.

**X.** [TwitterAPI.io advanced search](https://docs.twitterapi.io/api-reference/endpoint/tweet_advanced_search) uses `(AI OR "artificial intelligence" OR superintelligence) lang:en -filter:retweets`, without fear-only or engagement query conditions. Seed 7871 selects one five-minute interval within each four-hour stratum, six intervals per date. Each window is capped at five pages of 20 posts. All {x['capped_windows']} of 138 windows are capped; 30 minutes per day are planned search intervals, not a claim that every post within those minutes was captured. The sample favors posts returned most recently within each selected interval and excludes deleted/unindexed material.

The provider's documented `since_time`/`until_time` queries produced false empty responses in diagnostics. Verified `since_id`/`max_id` boundaries derived from X's timestamp-encoded IDs replaced that run, using the **same preselected windows**. Every primary post's creation time is checked against its window and encoded ID. The original epoch run and diagnosis ledgers are preserved for provenance and cost reconciliation; their posts do not enter the primary series.

Common text cleaning requires an explicit whole term `AI`, `artificial intelligence` or `superintelligence` in the post, at least five whitespace tokens after removing URLs and @mentions, and a unique author-ID/normalized-text pair. HTML entities are decoded. Both exposure groups use exactly this same cleaned universe before applying the exposure gate. Brand-name substring hits and compound-hashtag-only posts are excluded; this improves keyword precision but does not constitute human verification of AI stance. Every cleaning-stage count is disclosed in `analysis/x_cleaning_audit.json`.

Exposure filtering is local and uses counts **at retrieval**, not counts at historical posting or trading closes. Equality passes; missing counts are separately recorded and excluded from the qualified sample. The raw sample is preserved, scored under both models and tested separately. This retrospective selection can favor large accounts, viral concerns and commercial promotion. Older posts have had longer to accumulate views, potentially changing the selection rate across dates. It cannot establish a historically implementable trading strategy. Automated-account flags and commercial-cue regexes are diagnostic proxies, not verified spam labels.

**Prices and initiating event.** Yahoo Finance adjusted daily closes cover 25 symbols and 18 dates, including September 4 as the prior-session baseline. Nvidia, Oracle and Microsoft endpoints were independently checked against ChartExchange. Hubinger's initiating [X post](https://x.com/EvanHub/status/2097497037956891126) was verified through X syndication and an API pilot: September 8 at 21:27:18 EDT, after the equity close. It is event context, not an added observation in the seeded sample.

News/prices were retrieved October 1, 2026, and X on October 2. Estimated total X API usage, including the replaced run, pilot and diagnostics, is **${checks['x_data']['estimated_total_api_usd']:.4f}**, below the authorized $3 usage cap. The separate $10 account recharge is prepaid credit, not the study's API consumption.

## Timing and methods

The portfolio receives 1/23 of capital per constituent at the September 4 close and is held without rebalancing. Daily portfolio wealth is the mean of adjusted-price relatives; daily basket returns are its log differences multiplied by 100. SPY and QQQ excess log returns are supplementary outcomes. The constituents remain fixed; the basket is not the average of constituent log returns.

VADER is the lexicon baseline; frozen quantized Twitter RoBERTa supplies a contextual cross-check. RoBERTa's signed score is P(positive) minus P(negative); classes use maximum probability. Hu–Liu negative-word shares and risk/existential keyword flags provide additional descriptions. These measure language tone and themes, not AI-target stance or objective risk. The fixed review of 20 news headlines and 20 X posts is **AI-assisted**, with labels assigned before viewing scores; it is not independent human validation. Model hashes and truncation counts are disclosed.

News is conservatively assigned 23:59:59 New York time on its first archive date. X uses creation timestamps. Text enters the next observed 16:00 trading-close bin, and only the preceding bin predicts a return; weekend text is pooled. September 8's incomplete bin is excluded, sentiment is differenced, and one-lag models have 14 regression observations with F(1,11). Missing values are not filled as neutral. September 30 news cannot predict returns within this window. This text timing prevents using later text, but later exposure-based selection remains a retrospective limitation.

The eight baseline tests use one lag; two lags, benchmark adjustments, author/conversation weighting, unfiltered X and excluding accounts flagged automated are disclosed sensitivities. Holm corrections are reported across the eight baselines and all {len(tests)} specifications. ADF and residual tests have low power in the short window. COVID-style news comparisons control for archive date, topic and length with source-domain clustering; X qualification/commercial-cue comparisons use date/topic/length controls and author clustering. Source composition and selection associations do not identify unfair reporting or the COVID paper's audience-demand mechanism.

## Reproduce without new collection charges

Use Python 3.12+ and `requirements.txt`. Full post bodies, raw API/HTML responses, credentials and model weights stay local and are ignored by Git. Distributed body-free records in `data/public/` contain source IDs, URLs, metadata and model scores. Prices, daily model inputs and all test results are under `data/` and `analysis/`.

Run `python scripts/verify_results.py` to check portfolio arithmetic and every Granger F statistic/p-value through an independent restricted/unrestricted OLS calculation, plus Holm corrections and report pagination. A distributed checkout verifies available metadata and scores; full response hash checks require local raw caches.

## Full text collection and scoring

1. `python scripts/collect_data.py` collects news and prices. The legacy Hacker News exploration requires an explicit `--include-hn` and is excluded from this submission.
2. `python scripts/collect_x.py --dry-run` displays the fixed X sampling plan without network calls. Live collection requires an API key through `TWITTERAPI_IO_KEY` or ignored `tmp/twitterapi_io.key` and an explicit usage cap. A new collection can cost money: after a completed/distributed run, use `--refetch --max-usd YOUR_AUTHORIZED_CAP`. The collector preserves the previous ledger and any local text, checks response hashes on resume, and never prints the key. Do not rerun paid collection merely to inspect results.
3. Run `python scripts/filter_x.py --min-followers 100 --min-views 100`, then `python scripts/analyze.py --social-source x --min-followers 100 --min-views 100`. All 23 dates must have at least 20 qualified cleaned posts; incomplete coverage stops the full-period analysis. Raw and unfiltered cleaned text remain local.
4. Before examining model scores, label the fixed 20-record X review sample in `analysis/x_review_labels.json`. The news review is also keyed by IDs; changed source samples require new blind labels. Review labels are diagnostic and do not train either model.
5. Run `python scripts/download_model.py` to verify/download the frozen model, then `python scripts/score_roberta.py`. Unfiltered X predictions are reused for the exact qualified subset.
6. Run `python scripts/audit_and_charts.py`, `python scripts/analyze_roberta.py`, `python scripts/analyze_x_bias.py`, then `python scripts/build_report.py`.
7. Export using Microsoft Word on Windows: `powershell -File scripts/export_word_pdf.ps1`. On other systems use a compatible office renderer and check all six pages visually.
8. Run `python scripts/verify_results.py`, `python scripts/export_public.py`, `python scripts/write_readme.py`, and `python scripts/package_support.py`.

The Hu–Liu lexicon retains original license headers. Live archives and engagement counts can change, so newly collected samples may differ. The supplied scores, return series, model inputs and fitted specifications allow numerical checks without paid recollection.

## Deliverables

- `output/AI_Risk_Sentiment_2026_Report.pdf`: six-page submission, with Victor Chen byline and clickable GitHub link at the beginning.
- `output/AI_Risk_Sentiment_2026_Report.docx`: editable counterpart.
- `output/AI_Risk_Sentiment_2026_Support.zip`: scripts, source metadata, all fitted results, figures and model provenance. Full post bodies, credentials, raw responses and model weights are omitted.
- `Resources/`: the supplied course readings and presentations.
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')
    print('Updated README for completed X analysis.')

if __name__=='__main__':main()

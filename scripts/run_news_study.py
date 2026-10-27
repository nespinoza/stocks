"""Does LLM-extracted news predict returns or volatility? Post-cutoff evaluation.

Signals (per stock and session, point in time, see stock_api.news.aggregate):
  llm      Σ relevance × magnitude × direction from the cached LLM extractions
  lexicon  Σ Loughran–McDonald tone over the same documents (baseline)
Outcomes, all measured from closes:
  same-day reaction  close d−1 → d   (validates extraction; not tradable)
  next day           close d → d+1   (tradable only by acting at the close of d)
  h-day forward      close d+1 → d+1+h, h = 1, 5, 21  (what the lag-1 engine can earn)
IC = cross-sectional Spearman among stocks with news that session (≥ 5),
Newey–West t-stats; also after neutralizing trailing 5- and 21-day returns, and
without items the LLM labels 'market_move'. The volatility channel regresses
log(next-5-day realized variance / trailing 63-day variance) on event intensity.

    .conda-env/bin/python scripts/run_news_study.py --model haiku
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stock_api.evaluation import ic_summary, information_coefficient, neutralize
from stock_api.news import ExtractionCache, daily_panel, decay, etf_signal, item_scores
from stock_api.news.aggregate import assign_sessions

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 5, 21]


def load_prices(folder, tickers):
    frames = {t: pd.read_csv(folder/f'{t}.csv', parse_dates=['Date'], index_col='Date')['Adj Close'] for t in tickers
              if (folder/f'{t}.csv').exists()}
    return pd.DataFrame(frames).sort_index()


def outcome_panels(prices):
    logs = np.log(prices)
    out = {'same_day': logs-logs.shift(1), 'next_day': logs.shift(-1)-logs}
    for h in HORIZONS:
        out[f'fwd_{h}d'] = logs.shift(-(1+h))-logs.shift(-1)
    out['trail_5d'], out['trail_21d'] = logs-logs.shift(5), logs-logs.shift(21)
    return out


def long_frame(signals, outcomes, mask):
    data = {name: panel.where(mask).stack() for name, panel in signals.items()}
    data.update({name: panel.stack() for name, panel in outcomes.items()})
    frame = pd.DataFrame(data).rename_axis(['date', 'asset']).reset_index()
    return frame.dropna(subset=list(signals))


def ic_table(frame, signals, outcomes, label):
    rows = []
    for signal in signals:
        for outcome in outcomes:
            ic = information_coefficient(frame, signal, outcome, min_assets=5)
            if len(ic) < 5:
                continue
            horizon = int(outcome.split('_')[1][:-1]) if outcome.startswith('fwd') else 1
            # Overlapping h-day outcomes over a few months make HAC t-stats unreliable (spurious |t| of 3-6
            # on synthetic noise), so inference uses non-overlapping dates only; the mean uses all dates.
            independent = ic.iloc[::horizon]
            test = ic_summary(independent, 0) if len(independent) >= 3 else dict(t=np.nan, p=np.nan)
            rows.append(dict(sample=label, signal=signal, outcome=outcome, mean_ic=float(ic.mean()),
                             hit_rate=float((ic > 0).mean()), dates=len(ic), independent_dates=len(independent),
                             t=test['t'], p=test['p']))
    return rows


def volatility_channel(items, sessions, prices):
    logs = np.log(prices).diff()
    future = (logs**2).rolling(5).sum().shift(-6)  # close d+1 → d+6, matching the lag-1 holding
    baseline = (logs**2).rolling(63).mean().shift(1)*5  # through d−1, so the event day's own move is excluded
    ratio = np.log(future/baseline)
    intensity = daily_panel(items, sessions, value='vol_intensity', tickers=list(prices.columns)).reindex(prices.index).fillna(0.)
    any_news = daily_panel(items.assign(one=1.), sessions, value='one', tickers=list(prices.columns)).reindex(prices.index).fillna(0.)
    data = pd.DataFrame({'y': ratio.stack(), 'intensity': intensity.stack(), 'news': (any_news > 0).astype(float).stack()}).dropna()
    rows = []
    for spec in (['news'], ['intensity'], ['news', 'intensity']):
        beta, t, r2 = clustered_ols(data.y.to_numpy(), data[spec].to_numpy(), data.index.get_level_values(0))
        rows.append(dict(spec='+'.join(spec), n=len(data), r2=r2, **{f'b_{c}': b for c, b in zip(spec, beta[1:])},
                         **{f't_{c}': v for c, v in zip(spec, t[1:])}))
    return pd.DataFrame(rows)


def clustered_ols(y, x, clusters):
    """OLS with standard errors clustered by date (stocks on one day share shocks)."""
    x = np.column_stack([np.ones(len(y)), x])
    bread = np.linalg.inv(x.T@x)
    beta = bread@x.T@y
    residual = y-x@beta
    scores = pd.DataFrame(x*residual[:, None]).groupby(np.asarray(clusters)).sum().to_numpy()
    se = np.sqrt(np.diag(bread@(scores.T@scores)@bread))
    return beta, beta/se, float(1-residual@residual/np.sum((y-y.mean())**2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='haiku', help='model alias used when scoring')
    parser.add_argument('--since', default='2026-07-01')
    parser.add_argument('--output', default=str(ROOT/'reports/news-pipeline-2026-10-27'))
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    news = Path(os.environ.get('NEWS_DIR', ROOT/'data_store/news'))
    config = json.loads((ROOT/'configs/news_universe.json').read_text())
    tickers = list(config['stocks'])
    prices = load_prices(ROOT/'data_store/stocks', tickers)
    sessions = prices.index
    extractions = ExtractionCache(news/'extractions.jsonl').frame(args.model)
    extractions = extractions[extractions.published_utc >= args.since]
    lexicon = pd.read_csv(news/'lexicon_scores.csv')
    lexicon = lexicon.merge(extractions[['doc_id', 'ticker']], on=['doc_id', 'ticker'])  # same documents as the LLM
    items = item_scores(extractions)
    window = sessions[(sessions >= args.since)]
    counts = dict(llm_items=len(items), relevant_items=int((items.relevance >= .5).sum()),
                  sessions=len(window), first=str(window[0].date()), last=str(window[-1].date()),
                  by_source=items.source.value_counts().to_dict(), by_event=items.event_type.value_counts().to_dict())

    llm = daily_panel(items, sessions, tickers=tickers)
    lex = daily_panel(lexicon.rename(columns={'lm_tone': 'score'}), sessions, tickers=tickers)
    no_moves = daily_panel(items[items.event_type != 'market_move'], sessions, tickers=tickers)
    has_news = daily_panel(items.assign(one=1.)[items.relevance >= .5], sessions, value='one', tickers=tickers) > 0
    signals = {'llm': llm, 'lexicon': lex, 'llm_ex_market_move': no_moves,
               'llm_decay5': decay(llm, 5), 'lexicon_decay5': decay(lex, 5)}
    outcomes = outcome_panels(prices)
    frame = long_frame(signals, outcomes, has_news)
    frame = frame[frame.date >= args.since]
    outcome_names = ['same_day', 'next_day', *[f'fwd_{h}d' for h in HORIZONS]]
    rows = ic_table(frame, list(signals), outcome_names, 'raw')
    for name in ('llm', 'lexicon'):
        frame[f'{name}_neutral'] = neutralize(frame, name, ['trail_5d', 'trail_21d'])
    rows += ic_table(frame, ['llm_neutral', 'lexicon_neutral'], outcome_names[1:], 'neutralized vs trailing 5d/21d')
    ics = pd.DataFrame(rows)
    ics.to_csv(out/'ic.csv', index=False)

    # Agreement between the LLM and the dictionary on the same items.
    merged = items.merge(lexicon[['doc_id', 'ticker', 'lm_tone']], on=['doc_id', 'ticker'])
    relevant = merged[merged.relevance >= .5]
    agreement = dict(spearman_direction_vs_lm=float(relevant.direction.corr(relevant.lm_tone, method='spearman')),
                     share_lm_zero=float((relevant.lm_tone == 0).mean()), items=len(relevant))

    vol = volatility_channel(items, sessions, prices)
    vol.to_csv(out/'volatility_channel.csv', index=False)

    # Sector-ETF roll-up for the Black–Litterman layer.
    etf_prices = load_prices(ROOT/'data_store/etf', config['sector_etfs'])
    etf_z = etf_signal(decay(llm, 5), {t: s['etf'] for t, s in config['stocks'].items()})
    etf_out = outcome_panels(etf_prices.reindex(sessions))
    etf_frame = long_frame({'etf_llm_decay5': etf_z}, {k: etf_out[k] for k in ('fwd_5d', 'fwd_21d')},
                           pd.DataFrame(True, index=etf_z.index, columns=etf_z.columns))
    etf_frame = etf_frame[etf_frame.date >= args.since]
    etf_rows = ic_table(etf_frame, ['etf_llm_decay5'], ['fwd_5d', 'fwd_21d'], 'sector ETFs (10)')
    pd.DataFrame(etf_rows).to_csv(out/'etf_ic.csv', index=False)

    (out/'run_info.json').write_text(json.dumps(dict(model_alias=args.model, since=args.since, counts=counts,
                                                     agreement=agreement, models=sorted(extractions.model.dropna().unique()),
                                                     prompt_versions=sorted(extractions.prompt_version.unique())),
                                                indent=2, default=str))
    pd.set_option('display.width', 250)
    print(json.dumps(counts, indent=1, default=str))
    print(json.dumps(agreement, indent=1))
    print(ics.round(3).to_string())
    print(vol.round(3).to_string())
    print(pd.DataFrame(etf_rows).round(3).to_string())


if __name__ == '__main__':
    main()

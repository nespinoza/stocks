"""Build the single-origin numeric report and a browsable plot gallery."""
from pathlib import Path
import json
import html
import hashlib
import sys
import importlib.metadata
import re
from urllib.parse import unquote
import numpy as np
import pandas as pd
from stock_api.diagnostics.sampling import weighted_quantile
from stock_api.diagnostics.sampling import mcmc_diagnostics


def verify_saved_fit(folder, chosen):
    """Check saved samples and derived values without constructing/fitting a model."""
    meta = json.loads((folder/'config.json').read_text())
    assert hashlib.sha256((folder/'training.csv').read_bytes()).hexdigest() == meta['training_sha256']
    train = pd.read_csv(folder/'training.csv', index_col=0, float_precision='round_trip')
    test = pd.read_csv(folder/'held-out.csv', index_col=0, float_precision='round_trip')
    assert train.index.tolist() == meta['training_dates']
    assert test.index.tolist() == meta['forecast_dates']
    assert max(train.index) < meta['origin'] <= min(test.index)
    from stock_api import ValidationResult
    source = ValidationResult.load(f'validation_runs/return-gp-2026-09-01-benchmarks/lookback-{folder.parent.name.split("_")[-1]}.json')
    source._check_identity()
    assert source.experiment_id == meta['source_experiment_id']
    snapshot = pd.DataFrame(source.snapshots['prices']).set_index('date')
    np.testing.assert_allclose(train, snapshot.loc[train.index, train.columns], rtol=1e-14)
    np.testing.assert_allclose(test, snapshot.loc[test.index, test.columns], rtol=1e-14)
    posterior = pd.read_csv(chosen/'posterior.csv', float_precision='round_trip')
    summary = pd.read_csv(chosen/'summary.csv', float_precision='round_trip')
    info = json.loads((chosen/'inference.json').read_text())
    with np.load(chosen/'posterior.npz') as saved:
        weights = saved['weights']
        assert np.isfinite(weights).all() and (weights >= 0).all()
        np.testing.assert_allclose(weights.sum(), 1)
        np.testing.assert_allclose(weights, posterior.weight, rtol=1e-12)
        coordinates = saved['coordinates']
        assert np.isfinite(coordinates).all()
        physical = coordinates.copy()
        logs = np.log(train.to_numpy())
        if 'A_sigma' in posterior:
            positive = [i for i, name in enumerate(summary.parameter) if name != 'm_g']
            physical[:, positive] = np.exp(physical[:, positive])
            scale = max(float(np.sqrt(np.mean(np.diff(logs[:, 0])**2))), 1e-6)
            physical[:, 2] += 2*np.log(scale)
            if 'A_mu' in posterior: physical[:, 3] *= scale
            chains = saved['chains']
            np.testing.assert_array_equal(coordinates, chains.reshape(coordinates.shape))
            assert saved['latent_z'].shape[0] == len(weights)
            assert np.isfinite(saved['latent_z']).all()
            checks = mcmc_diagnostics(chains)
            for key, value in checks.items(): np.testing.assert_allclose(value, info[key])
            assert info['converged'] == (max(checks['split_rhat']) < 1.05 and min(checks['autocorrelation_ess']) > 100)
        else:
            physical = np.exp(coordinates)
            tasks = (physical.shape[1]-1)//2
            scale = np.maximum(logs[:, :tasks].std(axis=0), 1e-6)
            physical[:, 1:1+tasks] *= scale
            physical[:, 1+tasks:] *= scale
            np.testing.assert_allclose(info['weighted_ess'], 1/np.sum(weights**2))
            assert info['converged'] == (info['likelihood_calls'] < meta['config']['nested_maxcall'])
        np.testing.assert_allclose(physical, posterior[summary.parameter], rtol=1e-12, atol=1e-14)
        for row in summary.itertuples():
            np.testing.assert_allclose(weighted_quantile(posterior[row.parameter], weights),
                [row.lower95, row.lower68, row.median, row.upper68, row.upper95], rtol=1e-10, atol=1e-13)
    with np.load(chosen/'predictive.npz') as pred:
        paths = pred['paths']
        assert paths.shape[1] == len(test)+1 and np.isfinite(paths).all() and (paths > 0).all()
        np.testing.assert_allclose(paths[:, 0], train.iloc[-1, 0])
        q = np.quantile(paths[:, -1], [.025, .5, .975])
        p = info['predictive']
        np.testing.assert_allclose(q, [p['terminal_price_lower95'], p['terminal_price_median'], p['terminal_price_upper95']])
        terminal = np.log(paths[:, -1]/paths[:, 0])
        np.testing.assert_allclose([terminal.std(), (terminal > 0).mean()], [p['terminal_log_return_sigma'], p['probability_up']])
        for key, stored in [('median', 'old_median'), ('lower95', 'old_lower95'), ('upper95', 'old_upper95')]:
            np.testing.assert_allclose(p['old_terminal_price_'+key], pred[stored][-1])
    return {'folder': str(chosen), 'posterior_draws': len(weights), 'predictive_paths': len(paths),
            'training_prices': len(train), 'held_out_prices': len(test), 'verified': True}


def selected_folder(folder):
    for name in ['refined_long','refined']:
        if (folder/name/'inference.json').exists():return folder/name
    return folder


def generate(root='diagnostics/AMZN/2025-11-01'):
    root=Path(root);summaries=[];findings=[];predictions=[];sampling=[];gallery=[]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(root.glob('lookback_*/*/**/*')) if p.is_file()}
    verification = []
    for days in [30,60,90,180]:
        for model in ['gp','multitask_gp','volatility_gp_returns','heteroskedastic_gp_returns']:
            folder=root/f'lookback_{days}'/model;chosen=selected_folder(folder)
            verification.append(verify_saved_fit(folder, chosen))
            info=json.loads((chosen/'inference.json').read_text())
            summary=pd.read_csv(chosen/'summary.csv');summaries.append(summary.assign(lookback_days=days,model=model))
            posterior=pd.read_csv(chosen/'posterior.csv');w=posterior.weight.to_numpy();w/=w.sum()
            prior=json.loads((folder/'config.json').read_text())['config']['priors']
            training=pd.read_csv(folder/'training.csv',index_col=0)
            profiles=pd.read_csv(folder/'profiles.csv');starts=pd.read_csv(folder/'multistart.csv')
            finding={'lookback_days':days,'model':model,'selected_sampling':str(chosen.relative_to(root)),
                     'failed_profile_points':int((~profiles.success).sum()),'sampling_converged':info['converged']}
            if model in ['gp', 'multitask_gp']:
                for ticker in (['AMZN'] if model == 'gp' else training.columns):
                    scale = max(float(np.log(training[ticker]).std(ddof=0)), 1e-6)
                    mass = float(w[posterior['A_'+ticker] < prior['inactive_amplitude_ratio']*scale].sum())
                    finding['inactive_probability_'+ticker] = mass
            for kind,g in starts.groupby('objective_kind'):
                prefix='old' if kind=='old_regularized' else 'broad'
                finding[prefix+'_starts_successful']=int(g.success.sum())
                finding[prefix+'_starts_within_0.1_best']=int((g.objective<g.objective.min()+.1).sum())
                finding[prefix+'_objective_range']=float(g.objective.max()-g.objective.min())
            for name in [n for n in summary.parameter if n.startswith('ell')]:
                d=profiles[profiles.parameter==name];s=summary[summary.parameter==name].iloc[0]
                finding[name+'_profile_grid_best']=float(d.loc[d.objective.idxmin(),'value'])
                finding[name+'_profile_loss_lower_edge']=float(-d.iloc[0].relative_log_objective)
                finding[name+'_profile_loss_upper_edge']=float(-d.iloc[-1].relative_log_objective)
                finding[name+'_profile_range']=float(-d.relative_log_objective.min())
                finding[name+'_prob_below_1']=float(w[posterior[name]<1].sum())
                finding[name+'_prob_below_2']=float(w[posterior[name]<2].sum())
                finding[name+'_prob_above_100']=float(w[posterior[name]>100].sum())
                histogram,_=np.histogram(np.log(posterior[name]),bins=np.linspace(*np.log(prior['length']),21),weights=w)
                positive=histogram>0
                finding[name+'_marginal_KL_vs_log_uniform_nats']=float(np.sum(histogram[positive]*np.log(histogram[positive]*20)))
            if 'A_mu' in posterior:
                scatter=np.diff(np.log(training.iloc[:,0])).std()
                active=posterior.A_mu.to_numpy()>=prior['inactive_amplitude_ratio']*scatter
                finding['mean_inactive_probability']=float(w[~active].sum())
                finding['mean_process_active']=finding['mean_inactive_probability']<.05
                q=weighted_quantile(posterior.loc[active,'ell_mu'].to_numpy(),w[active]) if active.any() else [np.nan]*5
                for key,value in zip(['lower95','lower68','median','upper68','upper95'],q):finding['ell_mu_conditional_active_'+key]=float(value)
            if 'A_sigma' in posterior:
                finding['volatility_inactive_probability']=float(w[posterior.A_sigma<prior.get('log_variance_inactive_amplitude',.1)].sum())
                finding['volatility_process_active']=finding['volatility_inactive_probability']<.05
                extended=pd.read_csv(folder/'old-regularized-extended-profile.csv')
                for name,g in extended.groupby('parameter'):
                    finding[name+'_old_penalties_extended_best']=float(g.loc[g.objective.idxmin(),'value'])
            findings.append(finding)
            prediction={'lookback_days':days,'model':model,**info['predictive']}
            prediction['interval_width_ratio']=(prediction['terminal_price_upper95']-prediction['terminal_price_lower95'])/(prediction['old_terminal_price_upper95']-prediction['old_terminal_price_lower95'])
            predictions.append(prediction)
            sampling.append({'lookback_days':days,'model':model,'method':info['method'],'converged':info['converged'],
                             'max_split_rhat':max(info.get('split_rhat',[np.nan])),
                             'min_autocorrelation_ess':min(info.get('autocorrelation_ess',[np.nan])),
                             'weighted_ess':info.get('weighted_ess'), 'logZ':info.get('logZ'),'logZ_error':info.get('logZ_error'),
                             'draws_per_chain':info.get('draws_per_chain'), 'selected_sampling':str(chosen.relative_to(root))})
            links=[]
            for name in ['profiles.png','surface-ell_sigma-A_sigma.png','surface-ell_mu-A_mu.png',
                         'surface-ell_mu-ell_sigma.png','surface-A_mu-A_sigma.png','surface-ell-A_AMZN.png',
                         'corner.png','multistart.png','chains.png','latents.png','forecast.png','bound-sensitivity.png']:
                path=chosen/name if (chosen/name).exists() else folder/name
                if path.exists():
                    relative=path.relative_to(root).as_posix()
                    links.append(f'<a href="{relative}"><img loading="lazy" src="{relative}" alt="{name}"><span>{name}</span></a>')
            gallery.append(f'<section id="{days}-{model}"><h2>{days} days — {model}</h2><p>Sampling check: {info["converged"]}; <a href="{chosen.relative_to(root).as_posix()}/summary.csv">all credible intervals</a></p><div class="grid">'+''.join(links)+'</div></section>')
    summary=pd.concat(summaries,ignore_index=True);finding=pd.DataFrame(findings);prediction=pd.DataFrame(predictions);sample=pd.DataFrame(sampling)
    summary.to_csv(root/'hyperparameter-summary.csv',index=False);finding.to_csv(root/'identifiability.csv',index=False)
    prediction.to_csv(root/'predictive-comparison.csv',index=False);sample.to_csv(root/'sampling-checks.csv',index=False)
    (root/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>AMZN GP inference diagnostics — 2025-11-01</title>
<style>body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 20px;background:#f6f7f9;color:#17202a}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}.grid a{background:white;padding:12px;border:1px solid #ddd;text-decoration:none}.grid img{width:100%}section{margin:60px 0}span{display:block}a{color:#1655a2}</style>
<h1>AMZN GP inference diagnostics</h1><p>One origin: 2025-11-01. Trading-day kernel coordinates. Red: old optimizer; orange star: best profile; green/cyan: posterior. Return-GP surfaces are conditional ELBO slices; samples use joint latent-variable MCMC.</p><p><a href="REPORT.md">Written report</a> · <a href="hyperparameter-summary.csv">All 68/95% intervals</a> · <a href="sampling-checks.csv">Sampling checks</a></p>'''+''.join(gallery))
    runtime={'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','dynesty','matplotlib']},
             'scope':'Report generation environment; not a claim about the original sampling environment.',
             'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('stock_api/diagnostics').glob('*.py')}}
    (root/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    for path, digest in source_hashes.items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == digest, path
    (root/'verification.json').write_text(json.dumps({'fits':verification,
        'source_artifacts_unchanged':True, 'source_sha256':source_hashes,
        'checks':'snapshot identity and values, training checksum/date separation, CSV/NPZ agreement, weighted credible intervals, chain diagnostics or nested ESS/budget, predictive quantiles/probabilities'}, indent=2)+'\n')
    from gp_diagnostic_writeup import write_report
    write_report(root, summary, finding, prediction, sample)
    # Check generated links, including the previously missing written report.
    targets = re.findall(r'(?:href|src)="([^"]+)"', (root/'index.html').read_text())
    targets += re.findall(r'\]\(([^)]+)\)', (root/'REPORT.md').read_text())
    for target in targets:
        path, _, anchor = target.partition('#')
        if not path: continue
        destination = root/unquote(path)
        assert destination.is_file(), f'Missing report link: {target}'
        if anchor and destination.suffix == '.html':
            assert f'id="{anchor}"' in destination.read_text(), target
    audit = json.loads((root/'verification.json').read_text())
    audit['resolved_local_links'] = len(targets)
    (root/'verification.json').write_text(json.dumps(audit, indent=2)+'\n')
    return summary,finding,prediction,sample


if __name__=='__main__':
    _,findings,predictions,sampling=generate()
    print(sampling.to_string(index=False))
    print(findings[[c for c in ['lookback_days','model','ell_sigma_profile_grid_best','ell_sigma_prob_below_2','ell_mu_marginal_KL_vs_log_uniform_nats','mean_inactive_probability','ell_mu_conditional_active_lower95','ell_mu_conditional_active_upper95'] if c in findings]].to_string(index=False))
    print(predictions[['lookback_days','model','interval_width_ratio']].to_string(index=False))

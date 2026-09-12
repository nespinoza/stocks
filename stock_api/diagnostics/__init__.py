"""Opt-in diagnostics for a selected saved validation fold; never runs a backtest."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .config import DiagnosticConfig,PriorConfig
from .problems import GaussianProblem,ReturnProblem,component_activity
from .profiles import multistart,profile_1d,surfaces
from .sampling import sample_gaussian,sample_latent,weighted_quantile,mcmc_diagnostics
from .predictive import draw_predictive
from .plotting import all_plots


DIAGNOSTIC_MODELS={'gp':GaussianProblem,'multitask_gp':GaussianProblem,
                   'volatility_gp_returns':ReturnProblem,'heteroskedastic_gp_returns':ReturnProblem}


def diagnose_fold(run,origin,*,models=None,config=None,output='diagnostics',progress=print,resume=False):
    """Analyze ONE fold of a ValidationResult's immutable snapshot.

    No validate() call or new prices. Model adapters receive training prices only.
    The held-out table is passed only to the plotting/output layer.
    Custom adapters can register in DIAGNOSTIC_MODELS without modifying models.
    """
    config=config or DiagnosticConfig()
    if not config.enabled:return None
    run._check_identity()
    fold=next((f for f in run.folds if f['origin']==str(origin)),None)
    if fold is None:raise ValueError(f'Origin {origin} is not in the saved validation folds')
    frame=pd.DataFrame(run.snapshots['prices']).set_index('date');frame.index=pd.to_datetime(frame.index)
    tickers=[run.config['ticker'],*run.config['related_tickers']]
    train=frame.loc[pd.to_datetime(fold['train_dates']),tickers].copy()
    test=frame.loc[pd.to_datetime(fold['test_dates']),tickers].copy()
    if train.index.max()>=test.index.min():raise ValueError('Training/forecast dates overlap')
    names=list(DIAGNOSTIC_MODELS) if models is None else list(models)
    if any(name not in DIAGNOSTIC_MODELS for name in names):raise ValueError('Unsupported diagnostic model')
    root=Path(output)/tickers[0]/str(origin)/f"lookback_{run.config['train_days']}"
    root.mkdir(parents=True,exist_ok=True)
    metadata={'config':config.model_dump(),'source_experiment_id':run.experiment_id,
              'origin':origin,'training_dates':fold['train_dates'],'forecast_dates':fold['test_dates'],
              'training_sha256':hashlib.sha256(train.to_csv().encode()).hexdigest(),
              'time_unit':'trading_days','held_out_usage':'plots and predictive comparison only'}
    metadata=json.loads(json.dumps(metadata))
    for name in names:
        directory=root/name
        if directory.exists():
            if not resume:raise FileExistsError(f'{directory} exists; use resume=True to verify and resume')
            saved=json.loads((directory/'config.json').read_text())
            saved['config']=DiagnosticConfig.model_validate(saved['config']).model_dump(mode='json')
            if saved != metadata:raise ValueError('Cannot resume different data/configuration')
            if (directory/'complete.json').exists():
                progress(f'Already complete: {directory}');continue
        directory.mkdir(exist_ok=resume)
        (directory/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
        train.to_csv(directory/'training.csv');test.to_csv(directory/'held-out.csv')
        progress(f'{name}, {len(train)} training prices, origin {origin}')
        problem=DIAGNOSTIC_MODELS[name](train,name,config.priors)
        old={'parameters':dict(zip(problem.names,problem.physical(problem.old))), 'diagnostics':problem.old_info}
        if isinstance(problem,GaussianProblem):
            from stock_api.models import MODELS
            _,_,old['legacy_calendar_day_fit']=MODELS[name](train.to_numpy(),len(test),
                   train_times=(train.index-train.index[0]).days.to_numpy(),future_times=(test.index-train.index[0]).days.to_numpy())
        (directory/'old-optimizer.json').write_text(json.dumps(old,indent=2)+'\n')
        if (directory/'best-profile-coordinates.npy').exists():
            starts=pd.read_csv(directory/'multistart.csv') if config.multistart else None
            best=np.load(directory/'best-profile-coordinates.npy')
        elif config.multistart:
            starts,best=multistart(problem,config);starts.to_csv(directory/'multistart.csv',index=False)
        else:
            starts=None;best=problem.optimize(problem.old,maxiter=config.profile_maxiter)[0]
        progress('  multistart complete')
        profiles=surface=None
        if config.profile_surfaces:
            if (directory/'surfaces.csv').exists():
                profiles=pd.read_csv(directory/'profiles.csv');surface=pd.read_csv(directory/'surfaces.csv')
            else:
                profiles,best=profile_1d(problem,config,best,progress)
                profiles.to_csv(directory/'profiles.csv',index=False)
                surface=surfaces(problem,config,best);surface.to_csv(directory/'surfaces.csv',index=False)
        np.save(directory/'best-profile-coordinates.npy',best)
        samples=weights=latents=chains=predictive=None;summary=[];inference={}
        if config.posterior_sampling:
            progress('  posterior sampling')
            if (directory/'posterior.npz').exists():
                saved=np.load(directory/'posterior.npz')
                samples,weights=saved['coordinates'],saved['weights']
                if 'latent_z' in saved:
                    latents,chains=saved['latent_z'],saved['chains']
                    inference={'method':'joint_latent_gaussian_mcmc_collapsed_f','logZ':None,
                               **mcmc_diagnostics(chains),'warmup':config.warmup,'draws_per_chain':config.draws,
                               'chains':config.chains,'thin':config.thin,'interweave':config.interweave}
                    inference['converged']=bool(max(inference['split_rhat'])<1.05 and min(inference['autocorrelation_ess'])>100)
                else:
                    path=directory/'sampling.json' if (directory/'sampling.json').exists() else directory/'inference.json'
                    inference=json.loads(path.read_text())
            else:
                samples,weights,latents,chains,inference=(sample_gaussian(problem,config) if isinstance(problem,GaussianProblem) else sample_latent(problem,config,progress))
            (directory/'sampling.json').write_text(json.dumps(inference,indent=2)+'\n')
            np.savez_compressed(directory/'posterior.npz',coordinates=samples,weights=weights,
                                **({'latent_z':latents,'chains':chains} if latents is not None else {}))
            values=problem.physical(samples)
            pd.DataFrame(values,columns=problem.names).assign(weight=weights).to_csv(directory/'posterior.csv',index=False)
            for j,param in enumerate(problem.names):
                q=weighted_quantile(values[:,j],weights)
                low,high=problem.bounds[j];coordinate=samples[:,j]
                summary.append({'parameter':param,'old':problem.physical(problem.old)[j],
                                'best_profile':problem.physical(best[-len(problem.names):])[j],
                                **dict(zip(['lower95','lower68','median','upper68','upper95'],q)),
                                'mass_lower_5pct_prior_coordinate':float(weights[coordinate<low+.05*(high-low)].sum()),
                                'mass_upper_5pct_prior_coordinate':float(weights[coordinate>high-.05*(high-low)].sum())})
            pd.DataFrame(summary).to_csv(directory/'summary.csv',index=False)
            inference.update(component_activity(problem,values,weights))
            predictive=dict(np.load(directory/'predictive.npz')) if (directory/'predictive.npz').exists() else draw_predictive(problem,samples,weights,latents,len(test),config)
            np.savez_compressed(directory/'predictive.npz',**predictive)
            terminal=np.log(predictive['paths'][:,-1]/predictive['paths'][:,0]);q=np.quantile(predictive['paths'][:,-1],[.025,.5,.975])
            inference['predictive']={'terminal_price_lower95':float(q[0]),'terminal_price_median':float(q[1]),'terminal_price_upper95':float(q[2]),
                    'old_terminal_price_median':float(predictive['old_median'][-1]),
                    'old_terminal_price_lower95':float(predictive['old_lower95'][-1]),
                    'old_terminal_price_upper95':float(predictive['old_upper95'][-1]),
                    'terminal_log_return_sigma':float(terminal.std()),'probability_up':float(np.mean(terminal>0))}
            (directory/'inference.json').write_text(json.dumps(inference,indent=2)+'\n')
        all_plots(directory,problem,config,profiles,surface,starts,samples,weights,chains,best,predictive,test)
        (directory/'complete.json').write_text(json.dumps({'complete':True})+'\n')
        progress(f'  saved {directory}; sampling convergence={inference.get("converged")}')
    return root


__all__=['DiagnosticConfig','PriorConfig','diagnose_fold','DIAGNOSTIC_MODELS']

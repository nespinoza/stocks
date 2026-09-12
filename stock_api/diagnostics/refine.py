"""Refine a return-GP posterior without repeating profiles or touching held-out fits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import DiagnosticConfig
from .problems import ReturnProblem,component_activity
from .sampling import sample_latent,weighted_quantile,mcmc_diagnostics
from .predictive import draw_predictive
from .plotting import all_plots


def refine_posterior(directory,*,warmup=1500,draws=5000,interweave=True,progress=print,output_name="refined",continue_from=None):
    directory=Path(directory);destination=directory/output_name
    destination.mkdir(exist_ok=False)
    meta=json.loads((directory/'config.json').read_text())
    config=DiagnosticConfig.model_validate(meta['config']).model_copy(update={'warmup':warmup,'draws':draws,'interweave':interweave,'seed':143})
    (destination/'config.json').write_text(config.model_dump_json(indent=2)+'\n')
    train=pd.read_csv(directory/'training.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    test=pd.read_csv(directory/'held-out.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    problem=ReturnProblem(train,directory.name,config.priors)
    if continue_from is None:
        samples,weights,latents,chains,info=sample_latent(problem,config,progress)
    else:
        previous=np.load(directory/continue_from/'posterior.npz')
        oldchains=previous['chains'];oldz=previous['latent_z'].reshape(config.chains,oldchains.shape[1],problem.n)
        continuation=config.model_copy(update={'warmup':0,'seed':config.seed+1})
        samples,weights,latents,chains,info=sample_latent(problem,continuation,progress,initial=(oldchains[:,-1],oldz[:,-1]))
        chains=np.concatenate([oldchains,chains],axis=1)
        latents=np.concatenate([oldz,latents.reshape(config.chains,config.draws,problem.n)],axis=1).reshape(-1,problem.n)
        samples=chains.reshape(-1,len(problem.names));weights=np.full(len(samples),1/len(samples))
        info.update(mcmc_diagnostics(chains));info['draws_per_chain']=chains.shape[1]
        info['converged']=bool(max(info['split_rhat'])<1.05 and min(info['autocorrelation_ess'])>100)
        info['continued_from']=str(directory/continue_from);info['warmup']=config.warmup
        (destination/'continuation.json').write_text(continuation.model_dump_json(indent=2)+'\n')
    np.savez_compressed(destination/'posterior.npz',coordinates=samples,weights=weights,latent_z=latents,chains=chains)
    values=problem.physical(samples)
    pd.DataFrame(values,columns=problem.names).assign(weight=weights).to_csv(destination/'posterior.csv',index=False)
    best=np.load(directory/'best-profile-coordinates.npy')
    summary=[]
    for j,name in enumerate(problem.names):
        q=weighted_quantile(values[:,j],weights)
        lo,hi=problem.bounds[j]
        summary.append({'parameter':name,'old':problem.physical(problem.old)[j],
                        'best_profile':problem.physical(best[-len(problem.names):])[j],
                        **dict(zip(['lower95','lower68','median','upper68','upper95'],q)),
                        'mass_lower_5pct_prior_coordinate':float(weights[samples[:,j]<lo+.05*(hi-lo)].sum()),
                        'mass_upper_5pct_prior_coordinate':float(weights[samples[:,j]>hi-.05*(hi-lo)].sum())})
    pd.DataFrame(summary).to_csv(destination/'summary.csv',index=False)
    info.update(component_activity(problem,values,weights))
    pred=draw_predictive(problem,samples,weights,latents,len(test),config)
    np.savez_compressed(destination/'predictive.npz',**pred)
    terminal=np.log(pred['paths'][:,-1]/pred['paths'][:,0]);q=np.quantile(pred['paths'][:,-1],[.025,.5,.975])
    info['predictive']={'terminal_price_lower95':float(q[0]),'terminal_price_median':float(q[1]),'terminal_price_upper95':float(q[2]),
                    'old_terminal_price_median':float(pred['old_median'][-1]),'old_terminal_price_lower95':float(pred['old_lower95'][-1]),
                    'old_terminal_price_upper95':float(pred['old_upper95'][-1]),'terminal_log_return_sigma':float(terminal.std()),'probability_up':float(np.mean(terminal>0))}
    (destination/'inference.json').write_text(json.dumps(info,indent=2)+'\n')
    all_plots(destination,problem,config,pd.read_csv(directory/'profiles.csv'),pd.read_csv(directory/'surfaces.csv'),
              pd.read_csv(directory/'multistart.csv'),samples,weights,chains,best,pred,test)
    progress(f'Refined {directory}: {info["converged"]}; Rhat={max(info["split_rhat"]):.3f}, ESS={min(info["autocorrelation_ess"]):.0f}')
    return destination

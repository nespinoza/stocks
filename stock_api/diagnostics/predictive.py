"""Posterior predictive draws; held-out observations are never an input."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from stock_api.heteroskedastic import _kernel,_root,JITTER,paths_from_returns
from stock_api.models import MODELS


def draw_predictive(problem,samples,weights,latents,steps,config):
    rng=np.random.default_rng(config.seed+1)
    selected=rng.choice(len(samples),size=config.predictive_draws,p=weights)
    paths=[];fs=[];sigmas=[]
    for index in selected:
        p=samples[index]
        if latents is None:
            mean,cov=problem.predict(p,np.arange(problem.n,problem.n+steps,dtype=float))
            logprice=mean+_root(cov)@rng.normal(size=steps)
            paths.append(np.r_[float(problem.prices.iloc[-1,0]),np.exp(logprice)])
            continue
        state=problem.conditional(p,latents[index],state=True)
        future=np.arange(problem.n,problem.n+steps,dtype=float)
        cross,_=_kernel(problem.x,future,np.exp(p[0]),np.exp(p[1]))
        kfuture,_=_kernel(future,future,np.exp(p[0]),np.exp(p[1]))
        gmean=p[2]+cross.T@cho_solve((state['lg'],True),state['g']-p[2])
        gcov=kfuture+JITTER*np.eye(steps)-cross.T@cho_solve((state['lg'],True),cross)
        g=np.r_[state['g'],gmean+_root(gcov)@rng.normal(size=steps)]
        if problem.mean_gp:
            times=np.arange(problem.n+steps,dtype=float)
            cross,_=_kernel(problem.x,times,np.exp(p[3]),np.exp(p[4]))
            # Match the fitted white numerical jitter at training coordinates.
            cross[:,:problem.n]+=JITTER*np.eye(problem.n)
            kfull,_=_kernel(times,times,np.exp(p[3]),np.exp(p[4]))
            mean=cross.T@state['alpha'];cov=kfull+JITTER*np.eye(len(times))-cross.T@cho_solve(state['factor'],cross)
            f=mean+_root(cov)@rng.normal(size=len(times))
        else:f=np.zeros(len(g))
        sigma=problem.scale*np.exp(g/2)
        r=problem.scale*f[-steps:]+sigma[-steps:]*rng.normal(size=steps)
        paths.append(paths_from_returns(float(problem.prices.iloc[-1,0]),r[None,:])[0])
        fs.append(problem.scale*f);sigmas.append(sigma)
    paths=np.asarray(paths)
    if not np.isfinite(paths).all():raise ValueError('Posterior price paths overflowed')
    _,_,old=MODELS[problem.name](problem.prices.to_numpy(),steps,seed=config.seed,n_paths=10000) if latents is not None else (None,None,None)
    if old is None:
        means,variances,_=MODELS[problem.name](problem.prices.to_numpy(),steps)
        oldmedian=np.exp(means);oldlo=np.exp(means-1.96*np.sqrt(variances));oldhi=np.exp(means+1.96*np.sqrt(variances))
    else:
        oldmedian=old['_statistics']['median_price'];oldlo=old['_statistics']['lower_95'];oldhi=old['_statistics']['upper_95']
    return {'paths':paths,'f':np.asarray(fs),'sigma':np.asarray(sigmas),
            'old_median':oldmedian,'old_lower95':oldlo,'old_upper95':oldhi}

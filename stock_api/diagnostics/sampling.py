"""Exact Gaussian nested likelihoods and joint latent-Gaussian MCMC.

Return GPs: integrate f analytically conditional on g, sample whitened g via
elliptical slice sampling and covariance hyperparameters via bounded slice
sampling. The ELBO is never used as a likelihood for these posterior samples.
"""
import numpy as np
from scipy.special import logsumexp


def elliptical_slice(z,loglike,rng):
    current=loglike(z);threshold=current+np.log(rng.random())
    direction=rng.normal(size=z.shape);angle=rng.uniform(0,2*np.pi)
    lower,upper=angle-2*np.pi,angle
    for _ in range(1000):
        proposal=z*np.cos(angle)+direction*np.sin(angle)
        if loglike(proposal)>=threshold:return proposal
        if angle<0:lower=angle
        else:upper=angle
        angle=rng.uniform(lower,upper)
    raise RuntimeError('Elliptical slice failed to bracket an acceptable state')


def bounded_slice(x,logdensity,lower,upper,rng):
    threshold=logdensity(x)+np.log(rng.random());left,right=lower,upper
    for _ in range(1000):
        candidate=rng.uniform(left,right)
        if logdensity(candidate)>=threshold:return candidate
        if candidate<x:left=candidate
        else:right=candidate
    raise RuntimeError('Hyperparameter slice failed to find an acceptable state')


def mcmc_diagnostics(chains):
    """Split R-hat and autocorrelation ESS; also inspect traces, not just numbers."""
    m,n,d=chains.shape;n2=n//2
    split=np.concatenate([chains[:,:n2],chains[:,-n2:]],axis=0)
    within=split.var(axis=1,ddof=1).mean(axis=0)
    between=n2*split.mean(axis=1).var(axis=0,ddof=1)
    variance=(n2-1)/n2*within+between/n2
    rhat=np.sqrt(variance/np.maximum(within,1e-30))
    centered=chains-chains.mean(axis=1,keepdims=True)
    fft=np.fft.rfft(centered,n=2*n,axis=1)
    acov=np.fft.irfft(fft*np.conjugate(fft),axis=1)[:,:n]/np.arange(n,0,-1)[None,:,None]
    rho=1-(within-acov.mean(axis=0))/np.maximum(variance,1e-30)
    ess=[]
    for j in range(d):
        total=0.;previous=np.inf
        for k in range(1,n-1,2):
            pair=rho[k,j]+rho[k+1,j]
            if pair<0:break
            previous=min(previous,pair);total+=previous
        ess.append(min(m*n,m*n/(1+2*total)))
    return {'split_rhat':rhat.tolist(),'autocorrelation_ess':[float(e) for e in ess]}


def sample_latent(problem,config,progress=print,initial=None):
    rng=np.random.default_rng(config.seed);chains=[];latents=[];likelihoods=[]
    for c in range(config.chains):
        p=problem.transform(rng.uniform(.05,.95,len(problem.names))) if initial is None else initial[0][c].copy()
        z=rng.normal(size=problem.n)*.1 if initial is None else initial[1][c].copy()
        chain=[];zs=[];lls=[]
        for i in range(config.warmup+config.draws*config.thin):
            z=elliptical_slice(z,lambda zz:problem.conditional(p,zz),rng)
            for j in rng.permutation(len(p)):
                def target(value):
                    candidate=p.copy();candidate[j]=value
                    return problem.conditional(candidate,z)+problem.logprior_signed(candidate)
                p[j]=bounded_slice(p[j],target,*problem.bounds[j],rng)
            if config.interweave:
                # Centered/noncentered interweaving: retain the physical g state
                # while updating covariance parameters, then whiten it again.
                # The Gaussian density/Jacobian is required in centered coordinates.
                g=problem.conditional(p,z,state=True)['g']
                for j in rng.permutation(len(p)):
                    def centered_target(value):
                        from scipy.linalg import solve_triangular
                        candidate=p.copy();candidate[j]=value
                        lg,_=problem.factors(tuple(candidate))
                        zz=solve_triangular(lg,g-candidate[2],lower=True)
                        return (problem.conditional(candidate,zz)+problem.logprior_signed(candidate)
                                -.5*zz@zz-np.log(np.diag(lg)).sum())
                    p[j]=bounded_slice(p[j],centered_target,*problem.bounds[j],rng)
                from scipy.linalg import solve_triangular
                z=solve_triangular(problem.factors(tuple(p))[0],g-p[2],lower=True)
            if i>=config.warmup and (i-config.warmup)%config.thin==0:
                chain.append(p.copy());zs.append(z.copy());lls.append(problem.conditional(p,z))
            if (i+1)%500==0:progress(f'  chain {c+1}/{config.chains}: {i+1} iterations')
        chains.append(chain);latents.extend(zs);likelihoods.extend(lls)
    chains=np.asarray(chains);samples=chains.reshape(-1,len(problem.names))
    info={'method':'joint_latent_gaussian_mcmc_collapsed_f','logZ':None,
          **mcmc_diagnostics(chains),'warmup':config.warmup,'draws_per_chain':config.draws,
          'chains':config.chains,'thin':config.thin,'interweave':config.interweave}
    info['converged']=bool(max(info['split_rhat'])<1.05 and min(info['autocorrelation_ess'])>100)
    return samples,np.full(len(samples),1/len(samples)),np.asarray(latents),chains,info


def sample_gaussian(problem,config):
    try:from dynesty import NestedSampler
    except ImportError as exc:raise ImportError('Install the diagnostics extra: pip install -e ".[diagnostics]"') from exc
    sampler=NestedSampler(problem.loglike,problem.transform,len(problem.names),nlive=config.nlive,
                          sample='rwalk',rstate=np.random.default_rng(config.seed))
    sampler.run_nested(dlogz=config.nested_dlogz,maxcall=config.nested_maxcall,print_progress=False)
    r=sampler.results;weights=np.exp(r.logwt-logsumexp(r.logwt))
    calls=int(r.ncall.sum())
    info={'method':'dynesty_exact_gaussian_marginal_likelihood','logZ':float(r.logz[-1]),
          'logZ_error':float(r.logzerr[-1]),'likelihood_calls':calls,'weighted_ess':float(1/np.sum(weights**2)),
          'converged':calls<config.nested_maxcall,'nlive':config.nlive,
          'evidence_convention':'density of observed log prices, conditional on training-derived centering/scaling'}
    return r.samples,weights,None,None,info


def weighted_quantile(values,weights,probabilities=(.025,.16,.5,.84,.975)):
    values=np.asarray(values);weights=np.asarray(weights);order=np.argsort(values)
    x=values[order];w=weights[order];cdf=(np.cumsum(w)-.5*w)/w.sum()
    return np.interp(probabilities,cdf,x)

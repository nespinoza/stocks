"""Training-only adapters. Coordinates are log-positive parameters plus signed m_g."""
from functools import lru_cache
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm
from stock_api.models import MODELS
from stock_api.heteroskedastic import _kernel, _objective, fit_latent_returns, JITTER


class GaussianProblem:
    quantity = 'log marginal likelihood'
    def __init__(self, prices, name, priors):
        self.name,self.priors=name,priors
        self.prices=prices.iloc[:,:1] if name=='gp' else prices
        logs=np.log(self.prices.to_numpy())
        self.center=logs.mean(axis=0);self.scale=np.maximum(logs.std(axis=0),1e-6)
        self.y=(logs-self.center)/self.scale
        self.n,self.tasks=self.y.shape;self.x=np.arange(self.n,dtype=float)
        self.names=['ell']+[f'A_{t}' for t in self.prices.columns]+[f'noise_{t}' for t in self.prices.columns]
        self.bounds=np.log([priors.length]+[priors.price_amplitude_ratio]*self.tasks+[priors.price_noise_ratio]*self.tasks)
        self.old_bounds=np.log([[1,500]]+[[.01,10]]*self.tasks+[[.001,2]]*self.tasks)
        _,_,self.old_info=MODELS[name](prices.to_numpy(),5)
        self.old=np.log([self.old_info['length_scale'],*self.old_info['standardized_amplitudes'],*self.old_info['standardized_noises']])
        self.positive=np.ones(len(self.names),bool)

    def transform(self,u):
        return self.bounds[:,0]+u*(self.bounds[:,1]-self.bounds[:,0])

    def physical(self,p):
        result=np.exp(np.asarray(p)).copy()
        result[...,1:1+self.tasks]*=self.scale
        result[...,1+self.tasks:]*=self.scale
        return result

    def objective(self,p,old=False):
        ell=np.exp(p[0]);amps=np.exp(p[1:1+self.tasks]);noises=np.exp(p[1+self.tasks:])
        value=0.;grad=np.zeros_like(p);eye=np.eye(self.n)
        base,dl=_kernel(self.x,self.x,1.,ell)
        for j in range(self.tasks):
            k=amps[j]**2*base;fac=cho_factor(k+(noises[j]**2+1e-8)*eye,lower=True)
            alpha=cho_solve(fac,self.y[:,j]);q=cho_solve(fac,eye)-np.outer(alpha,alpha)
            value+=.5*self.y[:,j]@alpha+np.log(np.diag(fac[0])).sum()+self.n*(.5*np.log(2*np.pi)+np.log(self.scale[j]))
            grad[0]+=.5*np.sum(q*amps[j]**2*dl)
            grad[1+j]=np.sum(q*k);grad[1+self.tasks+j]=noises[j]**2*np.trace(q)
        return float(value),grad

    def loglike(self,p):
        # No inverse/derivatives needed for nested likelihood evaluation.
        ell=np.exp(p[0]);value=0.;base,_=_kernel(self.x,self.x,1.,ell)
        for j in range(self.tasks):
            amp,noise=np.exp(p[1+j]),np.exp(p[1+self.tasks+j])
            fac=cho_factor(amp**2*base+(noise**2+1e-8)*np.eye(self.n),lower=True)
            value-=.5*self.y[:,j]@cho_solve(fac,self.y[:,j])+np.log(np.diag(fac[0])).sum()+self.n*(.5*np.log(2*np.pi)+np.log(self.scale[j]))
        return float(value)

    def optimize(self,start, fixed=None,old=False,maxiter=600):
        fixed=fixed or {};bounds=self.old_bounds if old else self.bounds
        free=[i for i in range(len(start)) if i not in fixed]
        full=np.clip(start,*bounds.T).copy()
        for i,v in fixed.items():full[i]=v
        def objective(v):
            p=full.copy();p[free]=v;val,grad=self.objective(p,old)
            return val,grad[free]
        fit=minimize(objective,full[free],jac=True,bounds=bounds[free],method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-9})
        full[free]=fit.x
        return full,float(fit.fun),bool(fit.success),str(fit.message)

    def predict(self,p,times):
        amp,noise=np.exp(p[1]),np.exp(p[1+self.tasks]);ell=np.exp(p[0])
        k,_=_kernel(self.x,self.x,amp,ell);cross,_=_kernel(self.x,times,amp,ell);future,_=_kernel(times,times,amp,ell)
        fac=cho_factor(k+(noise**2+1e-8)*np.eye(self.n),lower=True)
        mean=self.center[0]+self.scale[0]*cross.T@cho_solve(fac,self.y[:,0])
        cov=self.scale[0]**2*(future+noise**2*np.eye(len(times))-cross.T@cho_solve(fac,cross))
        return mean,cov


class ReturnProblem:
    quantity = 'unregularized variational ELBO (approximate)'
    def __init__(self,prices,name,priors):
        self.name,self.priors=name,priors;self.prices=prices.iloc[:,:1]
        self.mean_gp=name=='heteroskedastic_gp_returns'
        self.returns=np.diff(np.log(prices.iloc[:,0].to_numpy()))
        self.scale=max(float(np.sqrt(np.mean(self.returns**2))),1e-6)
        self.scatter=max(float(self.returns.std()),1e-6)
        self.y=self.returns/self.scale;self.n=len(self.y);self.x=np.arange(self.n,dtype=float)
        self.level=float(np.log(max(np.var(self.y),1e-6)))
        self.names=['A_sigma','ell_sigma','m_g']+(['A_mu','ell_mu'] if self.mean_gp else [])
        self.positive=np.array([True,True,False]+([True,True] if self.mean_gp else []))
        b=[np.log(priors.log_variance_amplitude),np.log(priors.length),[self.level-priors.log_variance_mean_truncation*priors.log_variance_mean_sd,self.level+priors.log_variance_mean_truncation*priors.log_variance_mean_sd]]
        if self.mean_gp:b += [np.log(np.array(priors.return_mean_amplitude_ratio)*self.scatter/self.scale),np.log(priors.length)]
        self.bounds=np.array(b)
        self.old_bounds=np.array([np.log([.03,3]),np.log([2,252]),[-6,3]]+([np.log([.01,2]),np.log([2,252])] if self.mean_gp else []))
        old=fit_latent_returns(self.returns,mean_gp=self.mean_gp)
        self.old=np.array([np.log(old['ag']),np.log(old['ellg']),old['mg']]+([np.log(old['af']),np.log(old['ellf'])] if self.mean_gp else []))
        self.old_info={k:v for k,v in old.items() if isinstance(v,(float,int,str,bool))}
        self.latent_initial=np.r_[np.zeros(self.n),np.full(self.n,np.log(.5))]
        self.latent_bounds=np.array([[-8,8]]*self.n+[[-12,6]]*self.n)

    def transform(self,u):
        p=self.bounds[:,0]+u*(self.bounds[:,1]-self.bounds[:,0])
        p[2]=self.level+self.priors.log_variance_mean_sd*norm.ppf(norm.cdf(-self.priors.log_variance_mean_truncation)+u[2]*(norm.cdf(self.priors.log_variance_mean_truncation)-norm.cdf(-self.priors.log_variance_mean_truncation)))
        return p

    def physical(self,p):
        out=np.asarray(p).copy();out[...,self.positive]=np.exp(out[...,self.positive])
        out[...,2]+=2*np.log(self.scale)
        if self.mean_gp:out[...,3]*=self.scale
        return out

    def objective(self,p,old=False):
        value,gradient=_objective(p,self.y,self.mean_gp)
        if not old:
            h=2*self.n;ag,ellg=np.exp(p[h:h+2]);mg=p[h+2]
            value-=.5*ag**2+.5*((np.log(ellg/20))/1.5)**2+.5*(mg/2)**2
            gradient[h]-=ag**2;gradient[h+1]-=np.log(ellg/20)/1.5**2;gradient[h+2]-=mg/4
            if self.mean_gp:
                af,ellf=np.exp(p[h+3:h+5]);value-=.5*(af/.3)**2+.5*(np.log(ellf/20)/1.5)**2
                gradient[h+3]-=(af/.3)**2;gradient[h+4]-=np.log(ellf/20)/1.5**2
        value+=self.n*(.5*np.log(2*np.pi)+np.log(self.scale))
        return float(value),gradient

    def optimize(self,start,fixed=None,old=False,maxiter=600):
        fixed=fixed or {};h=2*self.n
        full=np.r_[self.latent_initial,start] if len(start)==len(self.names) else start.copy()
        bounds=np.vstack([self.latent_bounds,self.old_bounds if old else self.bounds]);full=np.clip(full,*bounds.T)
        for i,v in fixed.items():full[h+i]=v
        free=[i for i in range(len(full)) if i-h not in fixed]
        def obj(v):
            p=full.copy();p[free]=v;value,gradient=self.objective(p,old)
            return value,gradient[free]
        fit=minimize(obj,full[free],jac=True,method='L-BFGS-B',bounds=bounds[free],options={'maxiter':maxiter,'maxls':40,'maxcor':30,'ftol':1e-8})
        full[free]=fit.x
        return full,float(fit.fun),bool(fit.success),str(fit.message)

    def hyper(self,p):return p[-len(self.names):]

    @lru_cache(maxsize=32)
    def factors(self,parameters):
        p=np.array(parameters)
        kg,_=_kernel(self.x,self.x,np.exp(p[0]),np.exp(p[1]))
        lg=np.linalg.cholesky(kg+JITTER*np.eye(self.n))
        kf=_kernel(self.x,self.x,np.exp(p[3]),np.exp(p[4]))[0] if self.mean_gp else None
        return lg,kf

    def conditional(self,p,z, state=False):
        lg,kf=self.factors(tuple(p));g=p[2]+lg@z
        noise=np.exp(g)
        if self.mean_gp:
            fac=cho_factor(kf+np.diag(noise)+JITTER*np.eye(self.n),lower=True)
            alpha=cho_solve(fac,self.y)
            ll=-.5*self.y@alpha-np.log(np.diag(fac[0])).sum()
        else:
            kf=None;ll=-.5*np.sum(g+self.y**2/noise)
        ll-=self.n*(.5*np.log(2*np.pi)+np.log(self.scale))
        if state:return {'g':g,'lg':lg,'kf':kf,**({'factor':fac,'alpha':alpha} if self.mean_gp else {})}
        return float(ll)

    def logprior_signed(self,p):
        return -.5*((p[2]-self.level)/self.priors.log_variance_mean_sd)**2


def component_activity(problem,physical,weights):
    """Probability of a negligible component; never identify its length by its MAP alone."""
    cutoff=problem.priors.inactive_amplitude_ratio
    if isinstance(problem,GaussianProblem):
        probabilities={str(t):float(weights[physical[:,1+j]/problem.scale[j]<cutoff].sum()) for j,t in enumerate(problem.prices.columns)}
        return {'amplitude_inactive_probability':probabilities,
                'process_active':{t:p<.05 for t,p in probabilities.items()}}
    mass=float(weights[physical[:,0]<problem.priors.log_variance_inactive_amplitude].sum())
    result={'volatility_inactive_probability':mass,'volatility_process_active':mass<.05}
    if problem.mean_gp:
        mass=float(weights[physical[:,3]/problem.scatter<cutoff].sum())
        result.update(mean_inactive_probability=mass,mean_process_active=mass<.05,
                      mean_identifiability='mean amplitude consistent with zero; mean length scale unconstrained' if mass>=.05 else 'non-negligible mean amplitude; inspect length posterior')
    return result

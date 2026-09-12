"""Reusable log-grid profile optimization and explicitly conditional surfaces."""
import numpy as np
import pandas as pd


def hyper(problem,p):
    return p[-len(problem.names):]


def multistart(problem,config):
    rng=np.random.default_rng(config.seed);records=[];best=None
    for i in range(config.starts):
        raw=problem.transform(rng.uniform(.001,.999,len(problem.names)))
        for old in (True,False):
            p,value,success,message=problem.optimize(raw,old=old,maxiter=config.profile_maxiter)
            record={'start':i,'objective_kind':'old_regularized' if old else problem.quantity,
                    'objective':value,'success':success,'message':message}
            for j,name in enumerate(problem.names):
                record['start_'+name]=problem.physical(raw)[j]
                record['initial_feasible_'+name]=problem.physical(np.clip(raw,*(problem.old_bounds if old else problem.bounds).T))[j]
                record['final_'+name]=problem.physical(hyper(problem,p))[j]
            records.append(record)
            if not old and success and (best is None or value<best[1]):best=(p,value)
    if best is None:raise RuntimeError('No converged broad-domain optimizer start')
    return pd.DataFrame(records),best[0]


def profile_1d(problem,config,start,progress=print):
    records=[];best=start.copy();bestvalue=problem.objective(best)[0]
    for j,name in enumerate(problem.names):
        # Include signed level too, with linear physical coordinates.
        grid=np.linspace(*problem.bounds[j],config.profile_points)
        warm=best.copy()
        for value in grid:
            candidates=[problem.optimize(warm,{j:value},maxiter=config.profile_maxiter),
                        problem.optimize(best,{j:value},maxiter=config.profile_maxiter)]
            valid=[c for c in candidates if c[2]]
            fit=min(valid or candidates,key=lambda c:c[1]);warm=fit[0]
            if fit[2] and fit[1]<bestvalue:best,bestvalue=fit[0].copy(),fit[1]
            physical=problem.physical(hyper(problem,fit[0]))
            records.append({'parameter':name,'coordinate':value,'value':physical[j],
                            'objective':fit[1],'success':fit[2],'message':fit[3],
                            **{'optimized_'+n:physical[k] for k,n in enumerate(problem.names)}})
        progress(f'  profiled {name}')
    frame=pd.DataFrame(records)
    frame['relative_log_objective']=frame.groupby('parameter').objective.transform('min')-frame.objective
    return frame,best


def surface_pairs(problem):
    names=problem.names
    pairs=[('ell_sigma','A_sigma')]
    if 'ell_mu' in names:pairs += [('ell_mu','A_mu'),('ell_mu','ell_sigma'),('A_mu','A_sigma')]
    if 'ell' in names:pairs=[('ell',names[1])]
    return [(names.index(a),names.index(b)) for a,b in pairs]


def surfaces(problem,config,best):
    records=[]
    for i,j in surface_pairs(problem):
        for x in np.linspace(*problem.bounds[i],config.surface_points):
            for y in np.linspace(*problem.bounds[j],config.surface_points):
                p=best.copy();p[len(p)-len(problem.names)+i]=x;p[len(p)-len(problem.names)+j]=y
                value=problem.objective(p)[0];physical=problem.physical(hyper(problem,p))
                records.append({'x_parameter':problem.names[i],'y_parameter':problem.names[j],
                                'x':physical[i],'y':physical[j],'objective':value,
                                'kind':'conditional slice: all other coordinates including variational state fixed'})
    return pd.DataFrame(records)

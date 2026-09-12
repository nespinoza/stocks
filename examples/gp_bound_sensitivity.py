"""Isolate the old length bound: extend only the fixed length grid, retain old penalties."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from stock_api.diagnostics import DiagnosticConfig
from stock_api.diagnostics.problems import ReturnProblem


def run(root='diagnostics/AMZN/2025-11-01'):
    import matplotlib.pyplot as plt
    for days in [30,60,90,180]:
        for name in ['volatility_gp_returns','heteroskedastic_gp_returns']:
            folder=Path(root)/f'lookback_{days}'/name
            config=DiagnosticConfig.model_validate(json.loads((folder/'config.json').read_text())['config'])
            train=pd.read_csv(folder/'training.csv',index_col=0,parse_dates=True,float_precision='round_trip')
            problem=ReturnProblem(train,name,config.priors);rows=[]
            for param in ['ell_sigma']+(['ell_mu'] if problem.mean_gp else []):
                j=problem.names.index(param);warm=problem.old.copy()
                for v in np.linspace(*np.log(config.priors.length),config.profile_points):
                    fit=problem.optimize(warm,{j:v},old=True,maxiter=config.profile_maxiter)
                    warm=fit[0]
                    rows.append({'parameter':param,'value':float(np.exp(v)),'objective':fit[1],'success':fit[2]})
            result=pd.DataFrame(rows)
            result['relative_log_objective']=result.groupby('parameter').objective.transform('min')-result.objective
            result.to_csv(folder/'old-regularized-extended-profile.csv',index=False)
            original=pd.read_csv(folder/'profiles.csv')
            fig,axes=plt.subplots(len(result.parameter.unique()),1,squeeze=False,figsize=(9,4*len(result.parameter.unique())))
            for ax,param in zip(axes[:,0],result.parameter.unique()):
                d=result[result.parameter==param];other=original[original.parameter==param]
                ax.plot(d.value,d.relative_log_objective,label='old regularized ELBO, extended length only')
                ax.plot(other.value,other.relative_log_objective,label='unregularized ELBO, full prior domain')
                ax.axvline(problem.physical(problem.old)[problem.names.index(param)],c='red',ls='--',label='old solution')
                ax.axvline(2,c='gray',ls=':',label='old lower length bound')
                ax.set(xscale='log',xlabel=param+' (trading days)',ylabel='Relative objective (each curve normalized)',ylim=(-10,.2));ax.legend(fontsize=8)
            fig.tight_layout();fig.savefig(folder/'bound-sensitivity.png',dpi=130,bbox_inches='tight');plt.close(fig)
            print(days,name,result.loc[result.groupby('parameter').objective.idxmin(),['parameter','value']].to_dict('records'),flush=True)


if __name__=='__main__':run()

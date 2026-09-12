"""Matplotlib diagnostic artifacts in physical parameter units."""
import numpy as np
from .sampling import weighted_quantile


def save(fig,path):
    import matplotlib.pyplot as plt
    fig.savefig(path,dpi=130,bbox_inches='tight');plt.close(fig)


def limits(problem,j):
    a=problem.bounds[:,0].copy();b=problem.bounds[:,1].copy()
    return problem.physical(a)[j],problem.physical(b)[j]


def markers(ax,old,best,median,i,j):
    ax.scatter(old[i],old[j],marker='x',c='red',s=65,label='old optimizer',zorder=6)
    ax.scatter(best[i],best[j],marker='*',c='orange',edgecolor='black',s=100,label='best profile',zorder=6)
    if median is not None:ax.scatter(median[i],median[j],marker='+',c='cyan',s=75,label='posterior median',zorder=6)


def contour_posterior(ax,values,weights,i,j,problem):
    positive=problem.positive
    transform=lambda x,k:np.log(x) if positive[k] else x
    h,x,y=np.histogram2d(transform(values[:,i],i),transform(values[:,j],j),bins=24,weights=weights)
    descending=np.sort(h.ravel())[::-1];cumulative=np.cumsum(descending)/descending.sum()
    levels=sorted(set(descending[min(np.searchsorted(cumulative,p),len(descending)-1)] for p in [.95,.68]))
    levels=[v for v in levels if 0<v<h.max()]
    xc=(x[:-1]+x[1:])/2;yc=(y[:-1]+y[1:])/2
    if positive[i]:xc=np.exp(xc)
    if positive[j]:yc=np.exp(yc)
    if levels:ax.contour(xc,yc,h.T,levels=levels,colors='black',linewidths=.8)


def all_plots(root,problem,config,profiles,surfaces,starts,samples,weights,chains,best,predictive,test):
    import matplotlib.pyplot as plt
    old=problem.physical(problem.old);best=problem.physical(best[-len(problem.names):])
    values=problem.physical(samples) if samples is not None else None
    quantiles=np.array([weighted_quantile(values[:,j],weights) for j in range(len(problem.names))]) if values is not None else None
    median=quantiles[:,2] if quantiles is not None else None
    if profiles is not None:
        fig,axes=plt.subplots(len(problem.names),1,figsize=(9,3*len(problem.names)),squeeze=False)
        for j,name in enumerate(problem.names):
            ax=axes[j,0];d=profiles[profiles.parameter==name]
            ax.plot(d.value,d.relative_log_objective,'o-',ms=3)
            failed=d[~d.success];ax.scatter(failed.value,failed.relative_log_objective,c='red',marker='x',label='optimizer not converged')
            ax.axvline(old[j],color='red',ls='--',label='old optimizer')
            ax.axvline(best[j],color='orange',ls=':',label='best profile')
            if median is not None:
                ax.axvspan(quantiles[j,0],quantiles[j,4],alpha=.15,color='green',label='posterior 95%')
                ax.axvline(median[j],color='green',label='posterior median')
            if problem.positive[j]:ax.set_xscale('log')
            ax.set(xlabel=name,ylabel='Δ '+problem.quantity,ylim=(-30,1));ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);fig.suptitle(problem.name+' — reoptimized 1D profiles');fig.tight_layout()
        save(fig,root/'profiles.png')
    if surfaces is not None:
        for (xname,yname),d in surfaces.groupby(['x_parameter','y_parameter'],sort=False):
            i,j=problem.names.index(xname),problem.names.index(yname)
            grid=d.pivot(index='y',columns='x',values='objective');delta=grid.to_numpy().min()-grid.to_numpy()
            fig,ax=plt.subplots(figsize=(8,6))
            im=ax.pcolormesh(grid.columns,grid.index,np.maximum(delta,-30),shading='auto',vmin=-30,vmax=0,cmap='viridis')
            if values is not None:contour_posterior(ax,values,weights,i,j,problem)
            markers(ax,old,best,median,i,j)
            ax.set(xscale='log',yscale='log',xlabel=xname,ylabel=yname,title='Conditional slice: '+problem.quantity)
            ax.legend(fontsize=8);fig.colorbar(im,ax=ax,label='Δ objective (higher is better; color clipped at −30)')
            save(fig,root/f'surface-{xname}-{yname}.png')
    if starts is not None:
        fig,axes=plt.subplots(len(problem.names),1,figsize=(9,3*len(problem.names)),squeeze=False)
        for j,name in enumerate(problem.names):
            ax=axes[j,0]
            for kind,d in starts.groupby('objective_kind',sort=False):
                ax.scatter(d['final_'+name],d.objective-d.objective.min(),label=kind,alpha=.65)
                failed=d[~d.success];ax.scatter(failed['final_'+name],failed.objective-d.objective.min(),marker='x',c='red')
            if problem.positive[j]:ax.set_xscale('log')
            ax.set(xlabel=name,ylabel='Objective above best in group',ylim=(-.1,30))
        axes[0,0].legend(fontsize=8);fig.suptitle('Multistart endpoints (red x: nonconverged)');fig.tight_layout()
        save(fig,root/'multistart.png')
    if values is not None and config.corner_plots:
        n=len(problem.names);fig,axes=plt.subplots(n,n,figsize=(3*n,3*n))
        rng=np.random.default_rng(config.seed);selected=rng.choice(len(values),min(1500,len(values)),p=weights)
        for j in range(n):
            for i in range(n):
                ax=axes[j,i]
                if i>j:ax.axis('off');continue
                if i==j:
                    bins=np.geomspace(*limits(problem,i),40) if problem.positive[i] else np.linspace(*limits(problem,i),40)
                    ax.hist(values[:,i],bins=bins,weights=weights,color='steelblue')
                    q=quantiles[i];ax.axvspan(q[0],q[4],color='green',alpha=.1);ax.axvspan(q[1],q[3],color='green',alpha=.2)
                    ax.axvline(q[2],color='green');ax.axvline(old[i],color='red',ls='--')
                    ax.set_title(f'{problem.names[i]}: {q[2]:.3g}\n68% [{q[1]:.3g}, {q[3]:.3g}]\n95% [{q[0]:.3g}, {q[4]:.3g}]',fontsize=9)
                else:
                    ax.scatter(values[selected,i],values[selected,j],s=2,alpha=.12)
                    contour_posterior(ax,values,weights,i,j,problem);markers(ax,old,best,median,i,j)
                    if problem.positive[j]:ax.set_yscale('log')
                if problem.positive[i]:ax.set_xscale('log')
                ax.set_xlim(*limits(problem,i))
                if j==n-1:ax.set_xlabel(problem.names[i])
                if i==0 and j>0:ax.set_ylabel(problem.names[j])
        fig.suptitle(problem.name+' — posterior; black contours: 68/95% mass in log-coordinate bins',y=1.01)
        fig.tight_layout();save(fig,root/'corner.png')
    if chains is not None:
        fig,axes=plt.subplots(len(problem.names),1,figsize=(11,2.2*len(problem.names)))
        for j,ax in enumerate(axes):
            for chain in chains:ax.plot(problem.physical(chain)[:,j],alpha=.55,lw=.6)
            ax.set_ylabel(problem.names[j])
            if problem.positive[j]:ax.set_yscale('log')
        axes[-1].set_xlabel('Retained iteration per chain');fig.tight_layout();save(fig,root/'chains.png')
    if predictive is not None:
        dates=test.index;paths=predictive['paths'];q=np.quantile(paths[:,1:],[.025,.16,.5,.84,.975],axis=0)
        fig,ax=plt.subplots(figsize=(10,5))
        ax.plot(problem.prices.index,problem.prices.iloc[:,0],color='black',label='training prices')
        ax.fill_between(dates,q[0],q[4],alpha=.2,label='marginalized 95%');ax.fill_between(dates,q[1],q[3],alpha=.2)
        ax.plot(dates,q[2],label='marginalized median')
        ax.plot(dates,predictive['old_median'],'--',color='orange',label='old point fit')
        ax.fill_between(dates,predictive['old_lower95'],predictive['old_upper95'],color='orange',alpha=.12,label='old 95%')
        ax.plot(dates,test.iloc[:,0],'o-',color='red',label='held-out prices (display only)')
        ax.axvline(problem.prices.index[-1],color='gray',ls=':');ax.set(ylabel='Adjusted price ($)',title=problem.name);ax.legend(fontsize=8)
        save(fig,root/'forecast.png')
        if predictive['f'].size and config.latent_plots:
            dates=problem.prices.index[1:].append(test.index);origin=problem.prices.index[-1]
            fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
            axes[0].plot(problem.prices.index[1:],100*problem.returns,'.',label='training log returns')
            actual=np.diff(np.log(np.r_[problem.prices.iloc[-1,0],test.iloc[:,0]]))
            axes[0].plot(test.index,100*actual,'rx',label='held-out returns (display only)');axes[0].set_ylabel('Daily log return × 100');axes[0].legend()
            for ax,key,label in [(axes[1],'f','Latent mean × 100'),(axes[2],'sigma','Latent volatility × 100')]:
                q=100*np.quantile(predictive[key],[.025,.16,.5,.84,.975],axis=0)
                ax.fill_between(dates,q[0],q[4],alpha=.2);ax.fill_between(dates,q[1],q[3],alpha=.3);ax.plot(dates,q[2]);ax.set_ylabel(label)
            for ax in axes:ax.axvspan(origin,test.index[-1],color='gray',alpha=.1);ax.axvline(origin,color='gray',ls=':')
            fig.suptitle(problem.name+' — joint latent posterior (68/95% intervals)');fig.tight_layout();save(fig,root/'latents.png')

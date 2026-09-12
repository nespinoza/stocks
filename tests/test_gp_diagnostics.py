import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative
from stock_api.diagnostics import DiagnosticConfig,PriorConfig
from stock_api.diagnostics.problems import GaussianProblem,ReturnProblem
from stock_api.diagnostics.sampling import elliptical_slice,bounded_slice,weighted_quantile,mcmc_diagnostics,sample_gaussian,sample_latent


def prices(n=22):
    x=100*np.exp(np.cumsum(np.random.default_rng(12).normal(0,.02,(n,3)),axis=0))
    return pd.DataFrame(x,index=pd.bdate_range('2025-01-01',periods=n),columns=['AMZN','GOOGL','AAPL'])


def test_prior_transforms_and_units():
    p=GaussianProblem(prices(),'gp',PriorConfig())
    np.testing.assert_allclose(np.exp(p.transform(np.array([0.,.5,1.]))),[.5,np.sqrt(.001*100),10.])
    np.testing.assert_allclose(p.physical(p.transform(np.zeros(3)))[1],.001*p.scale[0])
    r=ReturnProblem(prices(),'heteroskedastic_gp_returns',PriorConfig())
    np.testing.assert_allclose(r.physical(r.transform(np.zeros(5)))[3],.001*r.scatter)
    assert r.physical(r.transform(np.zeros(5)))[1]==pytest.approx(.5)
    assert r.physical(r.transform(np.ones(5)))[4]==pytest.approx(300)
    with pytest.raises(ValueError):PriorConfig(length=(0,300))


@pytest.mark.parametrize('name',['gp','multitask_gp','volatility_gp_returns','heteroskedastic_gp_returns'])
def test_diagnostic_objective_gradient(name):
    p=(GaussianProblem if name in ('gp','multitask_gp') else ReturnProblem)(prices(),name,PriorConfig())
    x=p.old if isinstance(p,GaussianProblem) else np.r_[p.latent_initial,p.old]
    numerical=approx_derivative(lambda x:p.objective(x)[0],x).ravel()
    np.testing.assert_allclose(p.objective(x)[1],numerical,atol=2e-5,rtol=1e-3)
    if isinstance(p,GaussianProblem):assert -p.objective(x)[0]==pytest.approx(p.loglike(x))


def test_slice_samplers_known_targets():
    rng=np.random.default_rng(3);z=np.zeros(2);draws=[]
    # Unit Gaussian prior times N(observation=1 | z, noise variance=1).
    for i in range(6000):
        z=elliptical_slice(z,lambda v:-.5*np.sum((v-1)**2),rng)
        if i>1000:draws.append(z)
    assert np.asarray(draws).mean()==pytest.approx(.5,abs=.05)
    assert np.asarray(draws).var()==pytest.approx(.5,abs=.06)
    x=0.;draws=[]
    for i in range(4000):
        x=bounded_slice(x,lambda v:-v*v/2,-8,8,rng)
        draws.append(x)
    assert np.mean(draws)==pytest.approx(0,abs=.06)
    assert np.std(draws)==pytest.approx(1,abs=.06)


def test_mcmc_metrics_detect_unmixed_chains():
    rng=np.random.default_rng(2);x=rng.normal(size=(4,500,2))
    assert max(mcmc_diagnostics(x)['split_rhat'])<1.02
    x[0,:,0]+=4
    assert mcmc_diagnostics(x)['split_rhat'][0]>1.2


def test_nested_reproducibility_and_weights():
    p=GaussianProblem(prices(12),'gp',PriorConfig())
    c=DiagnosticConfig(nlive=20,nested_dlogz=1.,nested_maxcall=10000)
    a=sample_gaussian(p,c);b=sample_gaussian(p,c)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    assert a[1].sum()==pytest.approx(1)
    assert np.isfinite(a[4]['logZ'])
    assert a[4]['method']=='dynesty_exact_gaussian_marginal_likelihood'


def test_latent_sampler_reproducible_and_not_elbo(monkeypatch):
    p=ReturnProblem(prices(12),'heteroskedastic_gp_returns',PriorConfig())
    def fail(*args,**kwargs):raise AssertionError('ELBO must not enter posterior sampling')
    monkeypatch.setattr(p,'objective',fail)
    c=DiagnosticConfig(chains=2,warmup=10,draws=20)
    a=sample_latent(p,c,lambda msg:None);b=sample_latent(p,c,lambda msg:None)
    np.testing.assert_array_equal(a[0],b[0]);assert a[4]['logZ'] is None
    assert np.isfinite(a[0]).all()


def test_weighted_quantiles():
    q=weighted_quantile([1,2,100],[.49,.49,.02])
    assert 1<q[2]<3


def test_saved_fold_diagnostics_no_leakage_and_resume(tmp_path):
    from stock_api import ValidationConfig,validate
    from stock_api.data import calendar
    from stock_api.diagnostics import diagnose_fold
    index=calendar().sessions_in_range('2025-01-01','2025-02-28').tz_localize(None)
    frame=prices(len(index));frame.index=index
    run=validate('AMZN',prices=frame,models=['last_price'],config=ValidationConfig(reference_date='2025-03-01',months=1))
    c=DiagnosticConfig(profile_surfaces=False,multistart=False,corner_plots=False,nlive=20,
                       nested_dlogz=1.,predictive_draws=100)
    a=diagnose_fold(run,'2025-02-01',models=['gp'],config=c,output=tmp_path/'a',progress=lambda m:None)
    assert (a/'gp/complete.json').exists()
    assert run.diagnose('2025-02-01',models=['gp'],config=c,output=tmp_path/'a',resume=True,progress=lambda m:None)==a
    with pytest.raises(FileExistsError):diagnose_fold(run,'2025-02-01',models=['gp'],config=c,output=tmp_path/'a')
    with pytest.raises(ValueError,match='different data/configuration'):
        diagnose_fold(run,'2025-02-01',models=['gp'],config=c.model_copy(update={'seed':99}),output=tmp_path/'a',resume=True)
    changed=frame.copy();changed.loc[changed.index>='2025-02-01']*=2
    other=validate('AMZN',prices=changed,models=['last_price'],config=ValidationConfig(reference_date='2025-03-01',months=1))
    b=diagnose_fold(other,'2025-02-01',models=['gp'],config=c,output=tmp_path/'b',progress=lambda m:None)
    np.testing.assert_array_equal(np.load(a/'gp/posterior.npz')['coordinates'],np.load(b/'gp/posterior.npz')['coordinates'])
    np.testing.assert_array_equal(np.load(a/'gp/predictive.npz')['paths'],np.load(b/'gp/predictive.npz')['paths'])


def test_interweaving_preserves_prior_when_likelihood_is_flat(monkeypatch):
    p=ReturnProblem(prices(12),'volatility_gp_returns',PriorConfig())
    original=p.conditional
    monkeypatch.setattr(p,'conditional',lambda par,z,state=False:original(par,z,state=True) if state else 0.)
    c=DiagnosticConfig(chains=2,warmup=100,draws=500,interweave=True)
    samples,_,_,_,info=sample_latent(p,c,lambda m:None)
    # Log-uniform priors are uniform in sampler coordinates.
    np.testing.assert_allclose(samples[:,:2].mean(axis=0),p.bounds[:2].mean(axis=1),atol=.25)
    assert abs(samples[:,2].mean()-p.level)<.3
    assert info['interweave'] is True


def test_inactive_component_does_not_claim_identified_length():
    from stock_api.diagnostics.problems import component_activity
    p=ReturnProblem(prices(12),'heteroskedastic_gp_returns',PriorConfig())
    physical=p.physical(p.transform(np.full(5,.5)))[None,:]
    physical[:,0]=.01;physical[:,3]=p.scatter*.001
    d=component_activity(p,physical,np.array([1.]))
    assert d['mean_process_active'] is False
    assert d['volatility_process_active'] is False
    assert 'unconstrained' in d['mean_identifiability']

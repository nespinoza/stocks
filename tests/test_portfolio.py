import unittest
import numpy as np
import pandas as pd
from stock_api.portfolio import optimize_weights, trade, shrunk_covariance, joint_gp, check_chronology


class PortfolioTests(unittest.TestCase):
    def test_constraints_and_mean_response(self):
        mu=np.linspace(-.02,.02,22); cov=np.eye(22)*.01
        w,_=optimize_weights(mu,cov,np.zeros(22))
        self.assertAlmostEqual(w.sum(),1,places=9)
        self.assertGreaterEqual(w.min(),-1e-10)
        self.assertLessEqual(w.max(),.2+1e-10)
        self.assertGreater(w[-1],w[0])

    def test_exact_cost_budget(self):
        w=np.full(22,1/22)
        holdings,fee,buy,sell=trade(2000,np.zeros(22),w,.001)
        self.assertAlmostEqual(fee,2000*.001/1.001,places=9)
        self.assertAlmostEqual(holdings.sum()+fee,2000,places=9)
        old=np.zeros(22); old[0]=2000
        holdings,fee,buy,sell=trade(2000,old,w,.0025)
        self.assertAlmostEqual(fee,.0025*(buy+sell),places=9)
        self.assertAlmostEqual(holdings.sum()+fee,2000,places=9)
        self.assertAlmostEqual(buy-sell+fee,0,places=9)
        _,fee,_,_=trade(2000,old,w,0)
        self.assertEqual(fee,0)

    def test_psd_and_gp_marginals(self):
        rng=np.random.default_rng(44); r=rng.normal(0,.01,(30,3)); r[:,1]+=.8*r[:,0]
        cov,corr=shrunk_covariance(r)
        self.assertGreater(np.linalg.eigvalsh(cov).min(),0)
        np.testing.assert_allclose(np.diag(corr),1)
        result=joint_gp(r,5,n_paths=1000,seed=42)
        self.assertGreater(np.linalg.eigvalsh(result['covariance']).min(),0)
        np.testing.assert_allclose(result['samples'].mean(axis=0),result['mu'],atol=1e-12)
        np.testing.assert_allclose(np.cov(result['samples'],rowvar=False),result['covariance'])
        from stock_api.heteroskedastic import fit_latent_returns,_future
        for i in range(3):
            p=fit_latent_returns(r[:,i],mean_gp=True)
            fm,fc,gm,gc=_future(p,5)
            expected=p['scale']**2*(fc.sum()+np.exp(gm+.5*np.diag(gc)).sum())
            self.assertAlmostEqual(result['covariance'][i,i],expected,places=12)

    def test_chronology(self):
        dates=pd.date_range('2026-03-01',periods=30)
        check_chronology(dates,pd.Timestamp('2026-03-30'),pd.Timestamp('2026-03-31'),pd.date_range('2026-03-31',periods=21))
        with self.assertRaises(ValueError):
            check_chronology(dates,pd.Timestamp('2026-03-29'),pd.Timestamp('2026-03-31'),pd.date_range('2026-03-31',periods=21))
        with self.assertRaises(ValueError):
            check_chronology(dates,pd.Timestamp('2026-03-30'),pd.Timestamp('2026-03-30'),pd.date_range('2026-03-31',periods=21))

if __name__=='__main__': unittest.main()

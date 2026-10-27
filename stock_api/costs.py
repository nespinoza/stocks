"""Trading-cost model: half-spread plus square-root market impact.

cost / wealth = Σᵢ |Δwᵢ| (sᵢ + k σᵢ √(|Δwᵢ| AUM / ADVᵢ))

where sᵢ is the half-spread, σᵢ the daily volatility and ADVᵢ the average daily
dollar volume (the square-root law; k ≈ 1 is a conservative choice, see
Tóth et al. 2011 and Frazzini, Israel & Moskowitz 2018).
"""
import numpy as np

# Half-spread assumptions (bp) by average daily dollar volume, for US-listed ETFs.
# These are assumptions, not measured quotes; double them for a stress case.
SPREAD_TIERS = ((1e9, .5), (2e8, 1.5), (5e7, 3.), (0., 6.))


def half_spread_bps(adv_dollars, tiers=SPREAD_TIERS):
    adv = np.asarray(adv_dollars, float)
    out = np.full(adv.shape, tiers[-1][1])
    for threshold, spread in reversed(tiers):
        out = np.where(adv >= threshold, spread, out)
    return out


def spread_impact_cost(trade, aum, adv_dollars, daily_vol, spread_bps=None, impact=1.):
    """Cost of a signed weight change `trade` as a fraction of wealth."""
    trade = np.abs(np.asarray(trade, float))
    adv = np.asarray(adv_dollars, float)
    spread = half_spread_bps(adv) if spread_bps is None else np.asarray(spread_bps, float)
    participation = trade*aum/np.maximum(adv, 1.)
    return float(np.sum(trade*(spread/1e4+impact*np.asarray(daily_vol, float)*np.sqrt(participation))))

"""Validation of public inference selection; no fitting or data access."""
from typing import Literal

InferenceName = Literal['lbfgsb', 'dynesty', 'mcmc']
GAUSSIAN_GP = {'gp', 'multitask_gp'}
RETURN_GP = {'volatility_gp_returns', 'heteroskedastic_gp_returns'}


def inference_settings(model, inference, setup, *, seed=0, n_paths=10000):
    """Validate model-specific options and return existing sampler configuration."""
    if setup is not None and not isinstance(setup, dict):
        raise ValueError('inference_setup must contain dictionaries or None')
    if inference is None:
        if setup:
            raise ValueError('inference_setup requires an explicit inference')
        return None
    supported = {'lbfgsb': GAUSSIAN_GP | RETURN_GP,
                 'dynesty': GAUSSIAN_GP, 'mcmc': RETURN_GP}
    if inference not in supported or model not in supported[inference]:
        raise ValueError(f'Inference {inference!r} is not supported for model {model!r}')
    allowed = {'lbfgsb': set(),
               'dynesty': {'nlive', 'nested_dlogz', 'nested_maxcall', 'priors'},
               'mcmc': {'chains', 'warmup', 'draws', 'thin', 'interweave', 'priors'}}
    setup = setup or {}
    unknown = set(setup) - allowed[inference]
    if unknown:
        raise ValueError(f'Unsupported {inference} inference_setup keys: {sorted(unknown)}')
    if inference == 'lbfgsb':
        return None
    from stock_api.diagnostics.config import DiagnosticConfig, PriorConfig
    if 'priors' in setup:
        if not isinstance(setup['priors'], dict):
            raise ValueError('priors must be a dictionary')
        unknown_priors = set(setup['priors']) - set(PriorConfig.model_fields)
        if unknown_priors:
            raise ValueError(f'Unknown prior settings: {sorted(unknown_priors)}')
    return DiagnosticConfig(profile_surfaces=False, multistart=False,
                            corner_plots=False, latent_plots=False,
                            **({'interweave': True, 'warmup': 1500, 'draws': 5000} | setup),
                            seed=seed, predictive_draws=n_paths)

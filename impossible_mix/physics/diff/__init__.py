"""DDSP-style differentiable port of the physics engine.

This subpackage mirrors a subset of `impossible_mix.physics` (modal and
drip events for now) using **pure PyTorch tensors**. All parameters are
differentiable: given an audio target, one can recover the physical
parameters (droplet radius, viscosity, modal frequencies, t60s, ...) by
gradient descent on a spectral loss.

The numpy engine in `impossible_mix.physics.*` is unchanged. The two APIs
coexist; this one is opt-in and only loaded when imported.

Public API:
    synth_modal_impact_diff(params, sr, n_samples) -> Tensor
    synth_drip_event_diff(params, sr, n_samples) -> Tensor
    multi_resolution_stft_loss(pred, target) -> Tensor
    fit_drip_event(target, sr, ...) -> (recovered_params, history)
"""

from impossible_mix.physics.diff.modal import (
    ModalParamsT,
    synth_modal_impact_diff,
)
from impossible_mix.physics.diff.droplet import (
    DripParamsT,
    synth_drip_event_diff,
)
from impossible_mix.physics.diff.losses import multi_resolution_stft_loss
from impossible_mix.physics.diff.granular import (
    GranularFlowParamsT,
    synth_granular_flow_diff,
)
from impossible_mix.physics.diff.reverb import IRParamsT, synth_reverb_diff
from impossible_mix.physics.diff.friction import (
    FrictionParamsT,
    synth_scrape_diff,
)
from impossible_mix.physics.diff.rolling_droplet import (
    RollingDropletParamsT,
    synth_rolling_droplet_diff,
)
from impossible_mix.physics.diff.fitting import (
    fit_drip_event,
    fit_drip_event_multistart,
    fit_friction,
    fit_granular_flow,
    fit_modal_impact,
    fit_modal_impact_multistart,
    fit_reverb_ir,
    fit_rolling_droplet,
)

__all__ = [
    "ModalParamsT",
    "synth_modal_impact_diff",
    "DripParamsT",
    "synth_drip_event_diff",
    "multi_resolution_stft_loss",
    "fit_drip_event",
    "fit_drip_event_multistart",
    "fit_modal_impact",
    "fit_modal_impact_multistart",
    "GranularFlowParamsT",
    "synth_granular_flow_diff",
    "fit_granular_flow",
    "IRParamsT",
    "synth_reverb_diff",
    "fit_reverb_ir",
    "FrictionParamsT",
    "synth_scrape_diff",
    "fit_friction",
    "RollingDropletParamsT",
    "synth_rolling_droplet_diff",
    "fit_rolling_droplet",
]

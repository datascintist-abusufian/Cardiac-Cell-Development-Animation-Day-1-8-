"""cardiosim — in-silico hiPSC-cardiomyocyte culture model and analysis pipeline.

Modules
    model     latent maturation/damage states, cell agents, survival, ground-truth phenotypes
    ep        coupled FitzHugh–Nagumo monolayer, calcium handling and active tension
    render    synthetic fluorescence images and bright-field beating videos
    analysis  image morphometrics, sarcomere spectra, optical-flow motion and calcium-trace metrics
    stats     longitudinal statistics, change-points, GP forecasting, survival, sensitivity analysis
    io        reading uploaded videos and image stacks
"""
__version__ = "3.0.0"

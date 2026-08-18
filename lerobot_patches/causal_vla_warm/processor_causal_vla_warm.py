"""CausalVLA-Warm uses the same processor pipeline as CausalVLA-v2."""

from ..causal_vla.processor_causal_vla import make_causal_vla_pre_post_processors


def make_causal_vla_warm_pre_post_processors(config, dataset_stats=None):
    return make_causal_vla_pre_post_processors(config, dataset_stats)

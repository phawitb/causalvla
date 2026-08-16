"""COVER-Safe uses the COVER-Base processor pipeline."""

from ..cover_base.processor_cover_base import make_cover_base_pre_post_processors


def make_cover_safe_pre_post_processors(config, dataset_stats=None):
    return make_cover_base_pre_post_processors(config, dataset_stats)

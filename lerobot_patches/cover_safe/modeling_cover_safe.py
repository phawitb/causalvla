"""COVER-Safe policy specialization."""

from ..cover_base.modeling_cover_base import CoverBasePolicy
from .configuration_cover_safe import CoverSafeConfig


class CoverSafePolicy(CoverBasePolicy):
    config_class = CoverSafeConfig
    name = "cover_safe"

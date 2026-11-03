"""Label Studio REST client: lives in ``galecore.labelstudio`` (shared with the API's
active-learning sampler, ADR-020); re-exported here for the data-engine CLI."""

from galecore.labelstudio import LabelStudioClient

__all__ = ["LabelStudioClient"]

"""Compatibility shim for the vendored AP-BWE model."""

from GPT_SoVITS.vendor.ap_bwe.models.model import APNet_BWE_Model, ConvNeXtBlock, get_padding

__all__ = ["APNet_BWE_Model", "ConvNeXtBlock", "get_padding"]

"""Stubs so the vendored GFPGAN / CodeFormer graphs do not need pip ``basicsr``."""
from __future__ import annotations

import logging
from typing import Any, Iterable, List, Union

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


class _ArchRegistry:
    """No-op stand-in for ``basicsr.utils.registry.ARCH_REGISTRY``."""

    def register(self, cls: Any = None, *_args: Any, **_kwargs: Any):
        def _decorate(target: Any) -> Any:
            return target

        if cls is not None and callable(cls) and not _kwargs:
            return cls
        return _decorate


ARCH_REGISTRY = _ArchRegistry()


def get_root_logger(name: str = "afrbench.restore") -> logging.Logger:
    return logging.getLogger(name)


@torch.no_grad()
def default_init_weights(
    module_list: Union[nn.Module, Iterable[nn.Module]],
    scale: float = 1,
    bias_fill: float = 0,
    **kwargs: Any,
) -> None:
    """Kaiming init used by the clean StyleGAN2 graph (copied from basicsr)."""
    modules: List[nn.Module]
    if isinstance(module_list, nn.Module):
        modules = [module_list]
    else:
        modules = list(module_list)
    for module in modules:
        for child in module.modules():
            if isinstance(child, nn.Conv2d):
                nn.init.kaiming_normal_(child.weight, **kwargs)
                child.weight.data *= scale
                if child.bias is not None:
                    child.bias.data.fill_(bias_fill)
            elif isinstance(child, nn.Linear):
                nn.init.kaiming_normal_(child.weight, **kwargs)
                child.weight.data *= scale
                if child.bias is not None:
                    child.bias.data.fill_(bias_fill)
            elif isinstance(child, _BatchNorm):
                nn.init.constant_(child.weight, 1)
                if child.bias is not None:
                    child.bias.data.fill_(bias_fill)

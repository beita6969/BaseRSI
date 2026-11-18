import torch.nn.functional as F

from basersi import config as C

PATCH_KERNEL = "model.visual.patch_embed.proj.weight"


def box_resample(weight, new_patch):
    old_patch = weight.shape[-1]
    if old_patch % new_patch:
        raise C.ConfigError(f"box resampling needs a patch size dividing {old_patch}, got {new_patch}")
    factor = old_patch // new_patch
    return F.avg_pool3d(weight.float(), (1, factor, factor)) * factor * factor


def flat_patch_response_change(old, new):
    return float((new.sum((3, 4)) - old.float().sum((3, 4))).abs().max())


def resample_patch_kernel(weight, new_patch):
    new = box_resample(weight, new_patch)
    return new.to(weight.dtype), flat_patch_response_change(weight, new)

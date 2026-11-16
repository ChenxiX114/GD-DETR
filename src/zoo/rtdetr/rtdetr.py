"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'zoom', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None,
                 zoom=None, eval_inject=False):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        # SPAZ opt-in: zoom=None (default) -> native behavior unchanged.
        self.zoom = zoom
        # SPAZ: also run the zoom injection at eval (variant B). Off by default
        # so train-only injection (variant A) stays configurable for ablation.
        self.eval_inject = eval_inject
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        images = x.clone()  # keep raw images for expert priors
        x = self.backbone(x)
        # encoder targets: full targets at train; at eval only the tooth mask
        # (for the eval-available anatomical prior) so the prior doesn't collapse.
        if self.training:
            enc_targets = targets
        elif targets is not None and any(
                isinstance(t, dict) and 'tooth_mask' in t for t in targets):
            enc_targets = [{k: t[k] for k in ('tooth_mask',)} for t in targets]
        else:
            enc_targets = None
        try:
            x = self.encoder(x, images=images, targets=enc_targets)
        except TypeError:
            x = self.encoder(x)

        # SPAZ: inject high-res tooth detail into P2 (opt-in). Eval-time
        # injection is gated by eval_inject (variant B); variant A keeps the
        # injection training-only.
        if (self.zoom is not None and targets is not None
                and (self.training or self.eval_inject)
                and isinstance(x, list) and 'orig_image' in targets[0]):
            origs = [t.get('orig_image').to(x[0].device)
                     for t in targets if 'orig_image' in t]
            boxes = [t.get('tooth_boxes') for t in targets
                     if 'tooth_boxes' in t]
            if len(origs) == len(boxes) == len(targets):
                x[0] = self.zoom(origs, boxes, x[0])

        # Inject MDK dictionary into decoder for query-guided selection
        if hasattr(self.decoder, 'set_dictionary') and hasattr(self.encoder, 'mdk'):
            self.decoder.set_dictionary(self.encoder.mdk.dictionary)

        x = self.decoder(x, targets if self.training else None)

        # Pass through DeFE info to criterion (backward compatible)
        if isinstance(x, dict):
            if hasattr(self.encoder, '_defe_info'):
                x['defe'] = self.encoder._defe_info
            if hasattr(self.encoder, '_mdk_contrast_loss'):
                x['mdk_contrast_loss'] = self.encoder._mdk_contrast_loss

            # DSQH: supervise the query-selection head with the DeFE density map
            if (self.training and getattr(self.decoder, 'dsqh_weight', 0.0) > 0
                    and hasattr(self.encoder, '_defe_info')
                    and isinstance(self.encoder._defe_info, dict)
                    and self.encoder._defe_info.get('density') is not None):
                x['dsqh_loss'] = self.decoder.compute_dsqh_loss(
                    self.encoder._defe_info['density'])

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 

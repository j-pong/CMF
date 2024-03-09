"""
This file is based on the code from: https://github.com/mariodoebler/test-time-adaptation
"""

import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from copy import deepcopy
from methods.base import TTAMethod
from models.model import ResNetDomainNet126
from augmentations.transforms_cotta import get_tta_transforms

from methods.tent import softmax_entropy

def to_float(t):
    return t.float() if torch.is_floating_point(t) else t

@torch.no_grad()
def kernel(
    model, 
    src_model, 
    bias=0.99, 
    normalization_constant=1e-4
):
    energy_buffer =  []
    for param, src_param in zip(model.parameters(), src_model.parameters()):
        energy = F.cosine_similarity(
            to_float(src_param[:].data[:].flatten()), 
            to_float(param.flatten()), 
            dim=-1)

        energy_buffer.append(energy)

    energy = torch.stack(energy_buffer, dim=0).mean()
    energy = (bias - energy) / normalization_constant
                
    return energy

@torch.no_grad()
def moments(
    model, 
    src_model, 
    alpha=0.99, 
    update_all=False, 
):
    # energy_buffer =  []
    if alpha < 1.0:
        for param, src_param in zip(model.parameters(), src_model.parameters()):
            if param.requires_grad or update_all: 
                fp32_param = to_float(alpha * param[:].data[:]) + (1 - alpha) * to_float(src_param[:].data[:])
                param.data[:] = fp32_param.half()
    return model


@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x


class CMF(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        model = model.half()
        super().__init__(cfg, model, num_classes)
        param_size_ratio = self.print_amount_trainable_params()

        self.use_weighting = cfg.CMF.USE_WEIGHTING
        self.use_prior_correction = cfg.CMF.USE_PRIOR_CORRECTION
        self.use_consistency = cfg.CMF.USE_CONSISTENCY
        self.momentum_probs = cfg.CMF.MOMENTUM_PROBS
        self.temperature = cfg.CMF.TEMPERATURE
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).cuda()
        self.tta_transform = get_tta_transforms(self.dataset_name, padding_mode="reflect", cotta_augs=False)

        # setup loss functions
        self.sce = SymmetricCrossEntropy()
        self.slr = SoftLikelihoodRatio()
        self.ent = Entropy()

        # copy and freeze the source model
        if isinstance(model, ResNetDomainNet126):
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        delattr(module, hook.name)

        self.src_model = deepcopy(self.model)
        for param in self.src_model.parameters():
            param.detach_()
            
        # CMF
        self.post_type = cfg.CMF.TYPE
        self.hidden_model = deepcopy(self.model)
        for param in self.hidden_model.parameters():
            param.detach_() 
        
        self.alpha = cfg.CMF.ALPHA
        
        self.hidden_var = 0
        self.q = cfg.CMF.Q * param_size_ratio
        
        self.gamma = cfg.CMF.GAMMA
        
        self.models = [self.src_model, self.model, self.hidden_model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()
        
    @torch.enable_grad()
    def target_generalization_loss(self, x):
        outputs = self.model(x)

        if self.use_weighting:
            with torch.no_grad():
                # calculate diversity based weight
                weights_div = 1 - F.cosine_similarity(self.class_probs_ema.unsqueeze(dim=0), outputs.softmax(1), dim=1)
                weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min())
                mask = weights_div < weights_div.mean()

                # calculate certainty based weight
                weights_cert = - self.ent(logits=outputs)
                weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min())

                # calculate the final weights
                weights = torch.exp(weights_div * weights_cert / self.temperature)
                weights[mask] = 0.

                self.class_probs_ema = update_model_probs(x_ema=self.class_probs_ema, x=outputs.softmax(1).mean(0), momentum=self.momentum_probs)

        # calculate the soft likelihood ratio loss
        loss_out = self.slr(logits=outputs)

        # weight the loss
        if self.use_weighting:
            loss_out = loss_out * weights
            loss_out = loss_out[~mask]
        loss = loss_out.sum() / self.batch_size

        # calculate the consistency loss
        if self.use_consistency:
            outputs_aug = self.model(self.tta_transform(x[~mask]))
            loss += (self.sce(x=outputs_aug, x_ema=outputs[~mask]) * weights[~mask]).sum() / self.batch_size

        if self.use_prior_correction:
            prior = outputs.softmax(1).mean(0)
            smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
            smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
            outputs = outputs * smoothed_prior
        
        return outputs, loss
    
    @torch.no_grad()
    def bayesian_filtering(self):
        # 1. predict step
        # NOTE: self.post_type==lp is the default, 
        # in which case the predict step and update step can be combined to reduce computation. 
        # For clarity, they are separated in the code.
        recovered_model = moments(
            model=self.hidden_model,
            src_model=self.src_model, 
            alpha=self.alpha,
            update_all=True
        )
        
        # 2. update step
        self.hidden_var = self.alpha ** 2 * self.hidden_var + self.q
            
        r = (1 - self.q)
        self.beta = r / (self.hidden_var + r)
        self.beta = self.beta if self.beta > 0.89 else 0.89
        self.beta = self.beta if self.beta < 0.9999 else 1.0
        
        self.hidden_var = self.beta * self.hidden_var
        self.hidden_model = moments(
            model=recovered_model, 
            src_model=self.model, 
            alpha=self.beta,
            update_all=True
        )
        
        # 3. parameter ensemble step
        self.model = moments(
            model=self.model, 
            src_model=recovered_model if self.post_type == "op" else self.hidden_model, 
            alpha=self.gamma
        )
        
        # logging
        if self.cfg.TEST.DEBUG:
            tgt_energy = kernel(
                model=self.model, 
                src_model=self.src_model, 
                bias=0, 
                normalization_constant=1.0
            )
            hidden_energy = kernel(
                model=self.hidden_model, 
                src_model=self.src_model, 
                bias=0, 
                normalization_constant=1.0
            )
            res ={
                "tgt_energy": tgt_energy,
                "hidden_energy": hidden_energy,
            }
        else: 
            res = None

        return res

    def forward_and_adapt(self, x):
        imgs_test = x[0].half()

        output, loss = self.target_generalization_loss(imgs_test)

        # update the model
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        res = self.bayesian_filtering()

        if res is not None:
            res = {
                "output": output, 
                "tgt_energy": res["tgt_energy"].item() if torch.is_tensor(res["tgt_energy"]) else res["tgt_energy"],
                "hidden_energy": res["hidden_energy"].item() if torch.is_tensor(res["hidden_energy"]) else res["hidden_energy"],
            }
        else:
            res = {
                "output": output, 
            }

        return res

    def collect_params(self):
        """Collect the affine scale + shift parameters from normalization layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model."""
        self.model.eval()
        self.model.requires_grad_(False)
        # re-enable gradient for normalization layers
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)


class SoftLikelihoodRatio(nn.Module):
    def __init__(self, clip=0.99, eps=1e-5):
        super(SoftLikelihoodRatio, self).__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits):
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return - (probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)).sum(1)


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1-self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

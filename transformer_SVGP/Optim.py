'''A wrapper class for optimizer '''
import numpy as np
import torch
import torch.optim

from transformers import AdamW, get_scheduler


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(gp, opt,likelihood, transformer):
    params = [{'params': transformer.parameters()}]
    if opt.use_gp:
        params += [{'params': gp.parameters()}, {'params': likelihood.parameters()}]

    return NoamOpt(opt.d_model, opt.n_layers, opt.n_warmup_steps,
            torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9))

def get_huggingface_opt(gp, opt, likelihood, transformer):
    params = [{'params': transformer.parameters()}]
    if opt.use_gp:
        params += [{'params': gp.parameters()}, {'params': likelihood.parameters()}]

    # optimizer = AdamW(params, lr=1e-3)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=opt.n_warmup_steps,
        num_training_steps=opt.epoch,
    )

    return optimizer, lr_scheduler

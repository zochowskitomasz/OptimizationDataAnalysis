import torch
from torch import optim
from optim.Optimizer import (
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _foreach_doc,
    _fused_doc,
    _maximize_doc,
    _params_doc,
    _to_scalar,
    _use_grad_for_differentiable,
    DeviceDict,
    Optimizer,
    ParamsT,
)

class Alg1Optim(Optimizer):
    def __init__(self,
                params, 
                lr=1e-3,
                momentum=0.9,
                clipper=1.0,):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                #

        defaults = dict(lr=lr, momentum=momentum, clipper=clipper)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            clipper = group['clipper']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                #initialization

                if len(state) == 0:
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m_t = state['m']

                # 1. Gradient clipping

                grad_norm = torch.norm(grad, p=2)
                clip_coef = clipper / (grad_norm + 1e-8)

                # min(tau, |grad|) can be realized with clamp()

                clip_coef = torch.clamp(clip_coef, max=1.0)
                g_clip = grad * clip_coef

                # 2. Momentum update
                m_t.mul_(momentum).add_(g_clip, alpha=1.0 - momentum)


                # 3. Normalized weight update
                m_norm = torch.norm(m_t, p=2)
                direction = m_t / (m_norm + 1e-8)
                p.add_(direction, alpha=-lr)
            
        return loss
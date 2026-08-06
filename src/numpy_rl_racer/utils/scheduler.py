import numpy as np


class LRScheduler:
    def __init__(self, initial_lr):
        self.lr = initial_lr

    def step(self):
        raise NotImplementedError


class ExponentialDecay(LRScheduler):
    def __init__(self, initial_lr, decay_rate):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def step(self):
        self.lr *= self.decay_rate


class StepDecay(LRScheduler):
    def __init__(self, initial_lr, drop_rate, drop_every):
        super().__init__(initial_lr)
        self.drop_rate = drop_rate
        self.drop_every = drop_every
        self._steps = 0

    def step(self):
        self._steps += 1
        if self._steps % self.drop_every == 0:
            self.lr *= self.drop_rate


class CosineAnnealingLR(LRScheduler):
    def __init__(self, initial_lr, eta_min=0.0, t_max=100000):
        super().__init__(initial_lr)
        if t_max < 1:
            raise ValueError(f"t_max must be >= 1, got {t_max}")
        self.initial_lr = initial_lr
        self.eta_min = eta_min
        self.t_max = t_max
        self._steps = 0

    def step(self):
        self._steps += 1
        capped = min(self._steps, self.t_max)
        self.lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * \
            (1.0 + np.cos(np.pi * capped / self.t_max))

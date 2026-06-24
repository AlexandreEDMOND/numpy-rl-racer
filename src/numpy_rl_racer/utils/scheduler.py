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

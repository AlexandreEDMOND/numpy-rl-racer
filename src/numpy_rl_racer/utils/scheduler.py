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


class LinearWarmup(LRScheduler):
    def __init__(self, initial_lr=0.0, final_lr=None, warmup_steps=None, post_scheduler=None):
        if final_lr is None:
            raise ValueError("final_lr must be provided")
        if warmup_steps is None:
            raise ValueError("warmup_steps must be provided")
        super().__init__(initial_lr)
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.post_scheduler = post_scheduler
        self._steps = 0

    def step(self):
        if self._steps < self.warmup_steps:
            self._steps += 1
            ratio = self._steps / self.warmup_steps
            self.lr = self.initial_lr + ratio * (self.final_lr - self.initial_lr)
        else:
            self._steps += 1
            if self.post_scheduler is not None:
                self.post_scheduler.step()
                self.lr = self.post_scheduler.lr
            else:
                self.lr = self.final_lr

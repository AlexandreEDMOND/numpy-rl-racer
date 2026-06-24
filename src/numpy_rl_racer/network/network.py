import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)


class Dense:
    def __init__(self, in_features, out_features, weight_decay=0.0):
        self.w = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self._x = None
        self.grad_w = None
        self.grad_b = None
        self.weight_decay = weight_decay

    def forward(self, x):
        self._x = x
        return x @ self.w + self.b

    def backward(self, grad_output):
        x = self._x
        if x.ndim == 1:
            self.grad_w = np.outer(x, grad_output)
            self.grad_w += self.weight_decay * self.w
            self.grad_b = grad_output.copy()
            return grad_output @ self.w.T
        self.grad_w = x.T @ grad_output
        self.grad_w += self.weight_decay * self.w
        self.grad_b = np.sum(grad_output, axis=0)
        return grad_output @ self.w.T

    def __repr__(self):
        return f"Dense({self.w.shape[0]}, {self.w.shape[1]})"


class NoisyLinear:
    def __init__(self, in_features, out_features, weight_decay=0.0, rng=None):
        self.in_features = in_features
        self.out_features = out_features
        self.rng = rng
        self.weight_decay = weight_decay

        self.w = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)

        sigma_init = 0.5 / np.sqrt(in_features)
        self.sigma_w = np.full((in_features, out_features), sigma_init)
        self.sigma_b = np.full(out_features, sigma_init)

        self.eps_w = None
        self.eps_b = None
        self._x = None
        self.grad_w = None
        self.grad_b = None
        self.grad_sigma_w = None
        self.grad_sigma_b = None

    def reset_noise(self):
        _rng = self.rng if self.rng is not None else np.random
        p, q = self.in_features, self.out_features
        eps_i = _rng.randn(p)
        eps_j = _rng.randn(q)

        def _f(x):
            return np.sign(x) * np.sqrt(np.abs(x))

        self.eps_w = np.outer(_f(eps_i), _f(eps_j))
        self.eps_b = _f(eps_j)

    def forward(self, x):
        self._x = x
        self.reset_noise()
        w_noisy = self.w + self.sigma_w * self.eps_w
        b_noisy = self.b + self.sigma_b * self.eps_b
        return x @ w_noisy + b_noisy

    def backward(self, grad_output):
        x = self._x
        w_noisy = self.w + self.sigma_w * self.eps_w

        if x.ndim == 1:
            grad_w = np.outer(x, grad_output)
            self.grad_b = grad_output.copy()
            grad_input = grad_output @ w_noisy.T
        else:
            grad_w = x.T @ grad_output
            self.grad_b = np.sum(grad_output, axis=0)
            grad_input = grad_output @ w_noisy.T

        self.grad_w = grad_w + self.weight_decay * self.w
        self.grad_sigma_w = grad_w * self.eps_w
        self.grad_sigma_b = self.grad_b * self.eps_b
        return grad_input

    def __repr__(self):
        return f"NoisyLinear({self.w.shape[0]}, {self.w.shape[1]})"


class MLP:
    def __init__(self, layer_sizes, output_activation=None):
        assert len(layer_sizes) >= 2
        self.layers = [Dense(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.output_activation = output_activation

    def forward(self, x):
        self._cached_pre_relu = []
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                self._cached_pre_relu.append(x.copy())
                x = relu(x)
        if self.output_activation == "sigmoid":
            x = 1.0 / (1.0 + np.exp(-x))
        elif self.output_activation == "tanh":
            x = np.tanh(x)
        return x

    def backward(self, grad_output):
        for i in range(len(self.layers) - 1, -1, -1):
            if i < len(self.layers) - 1:
                pre = self._cached_pre_relu[i]
                grad_output = grad_output * (pre > 0)
            grad_output = self.layers[i].backward(grad_output)
        return grad_output

    def parameters(self):
        return [(layer.w, layer.b) for layer in self.layers]

    def __repr__(self):
        return f"MLP(layers={self.layers}, output_activation={self.output_activation})"


class DuelingMLP:
    def __init__(self, state_dim, hidden_sizes, n_actions):
        layer_sizes = [state_dim] + list(hidden_sizes)
        self.shared_encoder = [Dense(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.value_layer = Dense(hidden_sizes[-1], 1)
        self.advantage_layer = Dense(hidden_sizes[-1], n_actions)
        self.layers = self.shared_encoder + [self.value_layer, self.advantage_layer]
        self._n_actions = n_actions

    def forward(self, x):
        self._cached_pre_relu = []
        for i, layer in enumerate(self.shared_encoder):
            x = layer.forward(x)
            self._cached_pre_relu.append(x.copy())
            x = relu(x)
        v = self.value_layer.forward(x)
        a = self.advantage_layer.forward(x)
        a_mean = np.mean(a, axis=1, keepdims=True)
        return v + a - a_mean

    def backward(self, grad_output):
        grad_a = grad_output - np.mean(grad_output, axis=1, keepdims=True)
        grad_v = np.sum(grad_output, axis=1, keepdims=True)
        grad_x = self.advantage_layer.backward(grad_a)
        grad_x += self.value_layer.backward(grad_v)
        for i in range(len(self.shared_encoder) - 1, -1, -1):
            pre = self._cached_pre_relu[i]
            grad_x = grad_x * (pre > 0)
            grad_x = self.shared_encoder[i].backward(grad_x)
        return grad_x

    def parameters(self):
        return [(layer.w, layer.b) for layer in self.layers]

    def __repr__(self):
        return f"DuelingMLP(encoder={self.shared_encoder}, value={self.value_layer}, advantage={self.advantage_layer})"


class SGD:
    def __init__(self, mlp, lr=1e-3, scheduler=None, momentum=0.0, max_grad_norm=None):
        self.mlp = mlp
        self.scheduler = scheduler
        self.lr = scheduler.lr if scheduler is not None else lr
        self.momentum = momentum
        self.max_grad_norm = max_grad_norm
        self._velocities = None

    def _update_sigma(self, layer, lr, scale):
        if hasattr(layer, "grad_sigma_w"):
            layer.sigma_w -= lr * scale * layer.grad_sigma_w
            layer.sigma_b -= lr * scale * layer.grad_sigma_b

    def _update_sigma_momentum(self, layer, vel, momentum, lr, scale):
        if hasattr(layer, "grad_sigma_w"):
            if "sigma_w" not in vel:
                vel["sigma_w"] = np.zeros_like(layer.sigma_w)
                vel["sigma_b"] = np.zeros_like(layer.sigma_b)
            vel["sigma_w"] = momentum * vel["sigma_w"] - lr * scale * layer.grad_sigma_w
            vel["sigma_b"] = momentum * vel["sigma_b"] - lr * scale * layer.grad_sigma_b
            layer.sigma_w += vel["sigma_w"]
            layer.sigma_b += vel["sigma_b"]

    def step(self):
        if self.max_grad_norm is not None:
            total_norm_sq = 0.0
            for layer in self.mlp.layers:
                total_norm_sq += np.sum(layer.grad_w ** 2) + np.sum(layer.grad_b ** 2)
                if hasattr(layer, "grad_sigma_w"):
                    total_norm_sq += np.sum(layer.grad_sigma_w ** 2) + np.sum(layer.grad_sigma_b ** 2)
            total_norm = np.sqrt(total_norm_sq)
            scale = self.max_grad_norm / total_norm if total_norm > self.max_grad_norm else 1.0
        else:
            scale = 1.0

        if self.momentum == 0.0:
            for layer in self.mlp.layers:
                layer.w -= self.lr * scale * layer.grad_w
                layer.b -= self.lr * scale * layer.grad_b
                self._update_sigma(layer, self.lr, scale)
        else:
            if self._velocities is None:
                self._velocities = {}
                for layer in self.mlp.layers:
                    vel = {
                        "w": np.zeros_like(layer.w),
                        "b": np.zeros_like(layer.b),
                    }
                    self._update_sigma_momentum(layer, vel, 0.0, 0.0, 0.0)
                    self._velocities[layer] = vel
            for layer in self.mlp.layers:
                vel = self._velocities[layer]
                vel["w"] = self.momentum * vel["w"] - self.lr * scale * layer.grad_w
                vel["b"] = self.momentum * vel["b"] - self.lr * scale * layer.grad_b
                layer.w += vel["w"]
                layer.b += vel["b"]
                self._update_sigma_momentum(layer, vel, self.momentum, self.lr, scale)
        if self.scheduler is not None:
            self.scheduler.step()
            self.lr = self.scheduler.lr


class Adam:
    def __init__(self, mlp, lr=1e-3, scheduler=None, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, max_grad_norm=None):
        self.mlp = mlp
        self.scheduler = scheduler
        self.lr = scheduler.lr if scheduler is not None else lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self._moments = None
        self._t = 0

    def _update_sigma_adam(self, layer, m, v, t, lr, scale):
        if hasattr(layer, "grad_sigma_w"):
            if "sigma_w" not in m:
                m["sigma_w"] = np.zeros_like(layer.sigma_w)
                m["sigma_b"] = np.zeros_like(layer.sigma_b)
                v["sigma_w"] = np.zeros_like(layer.sigma_w)
                v["sigma_b"] = np.zeros_like(layer.sigma_b)

            g_w = lr * scale * layer.grad_sigma_w
            g_b = lr * scale * layer.grad_sigma_b

            m["sigma_w"] = self.beta1 * m["sigma_w"] + (1 - self.beta1) * g_w
            m["sigma_b"] = self.beta1 * m["sigma_b"] + (1 - self.beta1) * g_b
            v["sigma_w"] = self.beta2 * v["sigma_w"] + (1 - self.beta2) * g_w ** 2
            v["sigma_b"] = self.beta2 * v["sigma_b"] + (1 - self.beta2) * g_b ** 2

            m_hat_w = m["sigma_w"] / (1 - self.beta1 ** t)
            m_hat_b = m["sigma_b"] / (1 - self.beta1 ** t)
            v_hat_w = v["sigma_w"] / (1 - self.beta2 ** t)
            v_hat_b = v["sigma_b"] / (1 - self.beta2 ** t)

            layer.sigma_w -= m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            layer.sigma_b -= m_hat_b / (np.sqrt(v_hat_b) + self.eps)

    def step(self):
        self._t += 1
        t = self._t

        if self.max_grad_norm is not None:
            total_norm_sq = 0.0
            for layer in self.mlp.layers:
                total_norm_sq += np.sum(layer.grad_w ** 2) + np.sum(layer.grad_b ** 2)
                if hasattr(layer, "grad_sigma_w"):
                    total_norm_sq += np.sum(layer.grad_sigma_w ** 2) + np.sum(layer.grad_sigma_b ** 2)
            total_norm = np.sqrt(total_norm_sq)
            scale = self.max_grad_norm / total_norm if total_norm > self.max_grad_norm else 1.0
        else:
            scale = 1.0

        if self._moments is None:
            self._moments = {}
            for layer in self.mlp.layers:
                m = {"w": np.zeros_like(layer.w), "b": np.zeros_like(layer.b)}
                v = {"w": np.zeros_like(layer.w), "b": np.zeros_like(layer.b)}
                self._update_sigma_adam(layer, m, v, 1, 0.0, 0.0)
                self._moments[layer] = (m, v)

        for layer in self.mlp.layers:
            m, v = self._moments[layer]

            gw = scale * layer.grad_w
            gb = scale * layer.grad_b

            if self.weight_decay != 0.0:
                gw = gw + self.weight_decay * layer.w
                gb = gb + self.weight_decay * layer.b

            m["w"] = self.beta1 * m["w"] + (1 - self.beta1) * gw
            m["b"] = self.beta1 * m["b"] + (1 - self.beta1) * gb
            v["w"] = self.beta2 * v["w"] + (1 - self.beta2) * gw ** 2
            v["b"] = self.beta2 * v["b"] + (1 - self.beta2) * gb ** 2

            m_hat_w = m["w"] / (1 - self.beta1 ** t)
            m_hat_b = m["b"] / (1 - self.beta1 ** t)
            v_hat_w = v["w"] / (1 - self.beta2 ** t)
            v_hat_b = v["b"] / (1 - self.beta2 ** t)

            layer.w -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)

            self._update_sigma_adam(layer, m, v, t, self.lr, scale)

        if self.scheduler is not None:
            self.scheduler.step()
            self.lr = self.scheduler.lr

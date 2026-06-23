import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)


class Dense:
    def __init__(self, in_features, out_features):
        self.w = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self._x = None
        self.grad_w = None
        self.grad_b = None

    def forward(self, x):
        self._x = x
        return x @ self.w + self.b

    def backward(self, grad_output):
        x = self._x
        if x.ndim == 1:
            self.grad_w = np.outer(x, grad_output)
            self.grad_b = grad_output.copy()
            return grad_output @ self.w.T
        self.grad_w = x.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0)
        return grad_output @ self.w.T

    def __repr__(self):
        return f"Dense({self.w.shape[0]}, {self.w.shape[1]})"


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


class SGD:
    def __init__(self, mlp, lr=1e-3):
        self.mlp = mlp
        self.lr = lr

    def step(self):
        for layer in self.mlp.layers:
            layer.w -= self.lr * layer.grad_w
            layer.b -= self.lr * layer.grad_b

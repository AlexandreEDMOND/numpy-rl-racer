import numpy as np


def relu(x):
    return np.maximum(0.0, x)


class Dense:
    def __init__(self, in_features, out_features):
        self.w = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)

    def forward(self, x):
        return x @ self.w + self.b

    def __repr__(self):
        return f"Dense({self.w.shape[0]}, {self.w.shape[1]})"


class MLP:
    def __init__(self, layer_sizes, output_activation=None):
        assert len(layer_sizes) >= 2
        self.layers = [Dense(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.output_activation = output_activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = relu(x)
        if self.output_activation == "sigmoid":
            x = 1.0 / (1.0 + np.exp(-x))
        elif self.output_activation == "tanh":
            x = np.tanh(x)
        return x

    def parameters(self):
        return [(layer.w, layer.b) for layer in self.layers]

    def __repr__(self):
        return f"MLP(layers={self.layers}, output_activation={self.output_activation})"

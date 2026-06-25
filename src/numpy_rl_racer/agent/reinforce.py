import numpy as np

from numpy_rl_racer.network import MLP, PolicyNetwork, SGD, Adam, mse_loss


ACTIONS = np.array([
    [-0.5, 1.0],
    [0.5, 1.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, -0.5],
], dtype=np.float64)

N_ACTIONS = len(ACTIONS)


def discounted_sum(rewards, gamma):
    T = len(rewards)
    returns = np.zeros(T)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


class REINFORCEAgent:
    def __init__(self, state_dim, hidden_sizes=None, lr=1e-3, gamma=0.99,
                 use_baseline=True, use_value_network=False,
                 optimizer_type="sgd", scheduler=None, momentum=0.0,
                 weight_decay=0.0, max_grad_norm=None, seed=None):
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        self.state_dim = state_dim
        self.hidden_sizes = list(hidden_sizes)
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.use_value_network = use_value_network
        self.rng = np.random.RandomState(seed) if seed is not None else None

        layer_sizes = [state_dim] + list(hidden_sizes) + [N_ACTIONS]
        self.policy_net = PolicyNetwork(layer_sizes)
        for layer in self.policy_net.layers:
            layer.weight_decay = weight_decay

        if optimizer_type == "adam":
            self.optimizer = Adam(
                self.policy_net, lr=lr, scheduler=scheduler,
                weight_decay=weight_decay, max_grad_norm=max_grad_norm,
            )
        else:
            self.optimizer = SGD(
                self.policy_net, lr=lr, scheduler=scheduler,
                momentum=momentum, max_grad_norm=max_grad_norm,
            )

        if use_value_network:
            v_layer_sizes = [state_dim] + list(hidden_sizes) + [1]
            self.value_net = MLP(v_layer_sizes)
            for layer in self.value_net.layers:
                layer.weight_decay = weight_decay
            self.value_optimizer = SGD(
                self.value_net, lr=lr, scheduler=None,
                momentum=momentum, max_grad_norm=max_grad_norm,
            )

    def act(self, state, training=True):
        state_2d = state.reshape(1, -1)
        logits = self.policy_net.forward(state_2d).flatten()
        if not training:
            return int(np.argmax(logits))
        return self.policy_net.sample_action(logits, rng=self.rng)

    def train_step(self, states, actions, rewards):
        T = len(rewards)
        returns = discounted_sum(rewards, self.gamma)

        if self.use_value_network:
            values = self.value_net.forward(states).flatten()
            advantages = returns - values
            value_grad = 2.0 * (values - returns) / T
            self.value_net.backward(value_grad.reshape(-1, 1))
            self.value_optimizer.step()
        elif self.use_baseline:
            advantages = returns - np.mean(returns)
        else:
            advantages = returns

        logits = self.policy_net.forward(states)
        probs = self.policy_net.get_probs(logits)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(T), actions] = 1.0
        grad_logits = advantages[:, np.newaxis] * (probs - one_hot) / T

        self.policy_net.backward(grad_logits)
        self.optimizer.step()

        log_probs = self.policy_net.log_prob(actions, logits)
        loss = float(np.mean(-log_probs * advantages))

        if self.use_value_network:
            v_loss = float(mse_loss(values, returns))
            return loss, v_loss
        return loss

    def save(self, path):
        params = {}
        for i, layer in enumerate(self.policy_net.layers):
            params[f"layer_{i}_w"] = layer.w
            params[f"layer_{i}_b"] = layer.b

        opt = self.optimizer
        if isinstance(opt, SGD) and opt._velocities is not None:
            for i, layer in enumerate(self.policy_net.layers):
                vel = opt._velocities.get(layer)
                if vel is not None:
                    params[f"vel_{i}_w"] = vel["w"]
                    params[f"vel_{i}_b"] = vel["b"]
        elif isinstance(opt, Adam) and opt._moments is not None:
            for i, layer in enumerate(self.policy_net.layers):
                m, v = opt._moments.get(layer, (None, None))
                if m is not None:
                    params[f"adam_m_{i}_w"] = m["w"]
                    params[f"adam_m_{i}_b"] = m["b"]
                    params[f"adam_v_{i}_w"] = v["w"]
                    params[f"adam_v_{i}_b"] = v["b"]
            params["adam_t"] = np.array(opt._t)

        if self.use_value_network:
            for i, layer in enumerate(self.value_net.layers):
                params[f"v_layer_{i}_w"] = layer.w
                params[f"v_layer_{i}_b"] = layer.b
            vopt = self.value_optimizer
            if vopt._velocities is not None:
                for i, layer in enumerate(self.value_net.layers):
                    vel = vopt._velocities.get(layer)
                    if vel is not None:
                        params[f"v_vel_{i}_w"] = vel["w"]
                        params[f"v_vel_{i}_b"] = vel["b"]

        if self.rng is not None:
            state = self.rng.get_state()
            params["rng_key"] = state[1]
            params["rng_pos"] = np.array(state[2])
            params["rng_has_gauss"] = np.array(state[3])
            params["rng_cached_gaussian"] = np.array(state[4])

        params["hidden_sizes"] = np.array(self.hidden_sizes)
        params["state_dim"] = np.array(self.state_dim)
        params["n_actions"] = np.array(N_ACTIONS)
        np.savez(path, **params)

    def load(self, path):
        data = np.load(path, allow_pickle=False)

        saved_hidden = list(data["hidden_sizes"])
        saved_state_dim = int(data["state_dim"])
        if saved_hidden != self.hidden_sizes:
            raise ValueError(
                f"Saved model hidden_sizes={saved_hidden} do not match "
                f"agent hidden_sizes={self.hidden_sizes}"
            )
        if saved_state_dim != self.state_dim:
            raise ValueError(
                f"Saved model state_dim={saved_state_dim} does not match "
                f"agent state_dim={self.state_dim}"
            )

        for i, layer in enumerate(self.policy_net.layers):
            key_w = f"layer_{i}_w"
            key_b = f"layer_{i}_b"
            if key_w in data and key_b in data:
                layer.w[:] = data[key_w]
                layer.b[:] = data[key_b]

        opt = self.optimizer
        if isinstance(opt, SGD) and "vel_0_w" in data:
            if opt._velocities is None:
                opt._velocities = {}
            for i, layer in enumerate(self.policy_net.layers):
                kw, kb = f"vel_{i}_w", f"vel_{i}_b"
                if kw in data and kb in data:
                    opt._velocities[layer] = {
                        "w": data[kw].copy(),
                        "b": data[kb].copy(),
                    }
        elif isinstance(opt, Adam) and "adam_m_0_w" in data:
            if opt._moments is None:
                opt._moments = {}
            for i, layer in enumerate(self.policy_net.layers):
                kw_m, kb_m = f"adam_m_{i}_w", f"adam_m_{i}_b"
                kw_v, kb_v = f"adam_v_{i}_w", f"adam_v_{i}_b"
                if kw_m in data and kb_m in data:
                    m = {"w": data[kw_m].copy(), "b": data[kb_m].copy()}
                    v = {"w": data[kw_v].copy(), "b": data[kb_v].copy()}
                    opt._moments[layer] = (m, v)
            if "adam_t" in data:
                opt._t = int(data["adam_t"])

        if self.use_value_network and "v_layer_0_w" in data:
            for i, layer in enumerate(self.value_net.layers):
                key_w = f"v_layer_{i}_w"
                key_b = f"v_layer_{i}_b"
                if key_w in data and key_b in data:
                    layer.w[:] = data[key_w]
                    layer.b[:] = data[key_b]
            if "v_vel_0_w" in data:
                vopt = self.value_optimizer
                if vopt._velocities is None:
                    vopt._velocities = {}
                for i, layer in enumerate(self.value_net.layers):
                    kw, kb = f"v_vel_{i}_w", f"v_vel_{i}_b"
                    if kw in data and kb in data:
                        vopt._velocities[layer] = {
                            "w": data[kw].copy(),
                            "b": data[kb].copy(),
                        }

        if self.rng is not None and "rng_key" in data:
            self.rng.set_state((
                "MT19937",
                data["rng_key"],
                int(data["rng_pos"]),
                int(data["rng_has_gauss"]),
                float(data["rng_cached_gaussian"]),
            ))

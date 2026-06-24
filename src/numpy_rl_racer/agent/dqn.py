import numpy as np

from numpy_rl_racer.network import DuelingMLP, MLP, SGD


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
        self.data = [None] * capacity
        self.size = 0
        self.pos = 0

    def add(self, priority, data):
        idx = self.capacity + self.pos
        self.data[self.pos] = data
        self._update(idx, priority)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 1:
            idx //= 2
            self.tree[idx] += change

    def get(self, s):
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[1]

    def update_priority(self, idx, priority):
        self._update(idx, priority)

ACTIONS = np.array([
    [-0.5, 1.0],
    [0.5, 1.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, -0.5],
], dtype=np.float64)

N_ACTIONS = len(ACTIONS)


class ReplayBuffer:
    def __init__(self, capacity=10000, rng=None):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.rng = rng

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        _rng = self.rng if self.rng is not None else np.random
        indices = _rng.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta0=0.4, beta_anneal_steps=100000, rng=None):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta0 = beta0
        self.beta = beta0
        self.beta_anneal_steps = beta_anneal_steps
        self.max_priority = 1.0
        self._step = 0
        self.rng = rng

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        self._step += 1
        self.beta = min(1.0, self.beta0 + (1.0 - self.beta0) * self._step / self.beta_anneal_steps)

        batch = []
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size)
        total = self.tree.total()
        segment = total / batch_size

        _rng = self.rng if self.rng is not None else np.random
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = _rng.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            indices[i] = idx
            priorities[i] = priority
            batch.append(data)

        probs = priorities / total
        is_weights = (1.0 / (len(batch) * probs)) ** self.beta
        is_weights /= is_weights.max()

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        return states, actions, rewards, next_states, dones, is_weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = abs(td) + 1e-6
            self.max_priority = max(self.max_priority, priority)
            self.tree.update_priority(idx, priority)

    def __len__(self):
        return self.tree.size


class DQNAgent:
    def __init__(self, state_dim, hidden_sizes=None, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update_freq=100,
                 use_double_dqn=True, use_per=False, alpha=0.6, beta0=0.4,
                 beta_anneal_steps=100000, tau=0.0, seed=None,
                 use_dueling_dqn=False, n_step=1, scheduler=None,
                 momentum=0.0, weight_decay=0.0, max_grad_norm=None):
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        self.rng = np.random.RandomState(seed) if seed is not None else None
        if use_dueling_dqn:
            self.online_net = DuelingMLP(state_dim, hidden_sizes, N_ACTIONS)
            self.target_net = DuelingMLP(state_dim, hidden_sizes, N_ACTIONS)
        else:
            self.online_net = MLP([state_dim] + list(hidden_sizes) + [N_ACTIONS])
            self.target_net = MLP([state_dim] + list(hidden_sizes) + [N_ACTIONS])
        for layer in self.online_net.layers:
            layer.weight_decay = weight_decay
        for layer in self.target_net.layers:
            layer.weight_decay = weight_decay
        self._hard_update_target()
        self.optimizer = SGD(self.online_net, lr=lr, scheduler=scheduler, momentum=momentum, max_grad_norm=max_grad_norm)
        self.use_per = use_per
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=alpha, beta0=beta0,
                                                         beta_anneal_steps=beta_anneal_steps,
                                                         rng=self.rng)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, rng=self.rng)
        self.gamma = gamma
        self.gamma_n = gamma ** n_step
        self.n_step = n_step
        self._n_step_buffer = []
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.tau = tau
        self._step_counter = 0
        self._last_avg_q = float("nan")

    def save(self, path):
        params = {}
        for i, layer in enumerate(self.online_net.layers):
            params[f"layer_{i}_w"] = layer.w
            params[f"layer_{i}_b"] = layer.b
        np.savez(path, **params)

    def load(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.online_net.layers):
            key_w = f"layer_{i}_w"
            key_b = f"layer_{i}_b"
            if key_w in data and key_b in data:
                layer.w[:] = data[key_w]
                layer.b[:] = data[key_b]
        self._hard_update_target()

    def _hard_update_target(self):
        for src, dst in zip(self.online_net.layers, self.target_net.layers):
            dst.w[:] = src.w
            dst.b[:] = src.b

    def _soft_update_target(self, tau):
        for src, dst in zip(self.online_net.layers, self.target_net.layers):
            dst.w[:] = tau * src.w + (1.0 - tau) * dst.w
            dst.b[:] = tau * src.b + (1.0 - tau) * dst.b

    def act(self, state, training=True):
        _rng = self.rng if self.rng is not None else np.random
        if training and _rng.random() < self.epsilon:
            return _rng.randint(N_ACTIONS)
        q_values = self.online_net.forward(state.reshape(1, -1)).flatten()
        return int(np.argmax(q_values))

    def train_step(self, state, action, reward, next_state, done):
        self._n_step_buffer.append((state, action, reward, next_state, done))

        if len(self._n_step_buffer) < self.n_step and not done:
            return 0.0

        n_state, n_action = self._n_step_buffer[0][0], self._n_step_buffer[0][1]
        G = 0.0
        n_next_state = next_state
        n_done = False
        for i in range(min(self.n_step, len(self._n_step_buffer))):
            _, _, r, ns, d = self._n_step_buffer[i]
            G += (self.gamma ** i) * r
            n_next_state = ns
            n_done = d
            if d:
                break

        self.replay_buffer.push(n_state, n_action, G, n_next_state, n_done)
        self._n_step_buffer.pop(0)

        if done:
            self._n_step_buffer.clear()

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        if self.use_per:
            states, actions, rewards, next_states, dones, is_weights, indices = \
                self.replay_buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        next_q_target = self.target_net.forward(next_states)
        if self.use_double_dqn:
            next_q_online = self.online_net.forward(next_states)
            best_actions = np.argmax(next_q_online, axis=1)
            max_next_q = next_q_target[np.arange(self.batch_size), best_actions]
        else:
            max_next_q = np.max(next_q_target, axis=1)
        target_q = rewards + self.gamma_n * max_next_q * (1.0 - dones)

        current_q = self.online_net.forward(states)
        self._last_avg_q = float(np.mean(np.max(current_q, axis=1)))
        q_sa = current_q[np.arange(self.batch_size), actions]

        if self.use_per:
            loss = np.mean(is_weights * (target_q - q_sa) ** 2)
            grad_q = np.zeros_like(current_q)
            grad_q[np.arange(self.batch_size), actions] = \
                2.0 * is_weights * (q_sa - target_q) / self.batch_size
        else:
            loss = np.mean((target_q - q_sa) ** 2)
            grad_q = np.zeros_like(current_q)
            grad_q[np.arange(self.batch_size), actions] = \
                2.0 * (q_sa - target_q) / self.batch_size

        self.online_net.backward(grad_q)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._step_counter += 1
        if self.tau > 0.0:
            self._soft_update_target(self.tau)
        elif self._step_counter % self.target_update_freq == 0:
            self._hard_update_target()

        if self.use_per:
            td_errors = np.abs(target_q - q_sa)
            self.replay_buffer.update_priorities(indices, td_errors)

        return float(loss)

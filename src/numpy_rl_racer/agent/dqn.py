import numpy as np

from numpy_rl_racer.network import MLP, SGD

ACTIONS = np.array([
    [-0.5, 1.0],
    [0.5, 1.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, -0.5],
], dtype=np.float64)

N_ACTIONS = len(ACTIONS)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, hidden_sizes=None, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update_freq=100):
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        self.online_net = MLP([state_dim] + list(hidden_sizes) + [N_ACTIONS])
        self.target_net = MLP([state_dim] + list(hidden_sizes) + [N_ACTIONS])
        self._hard_update_target()
        self.optimizer = SGD(self.online_net, lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._step_counter = 0

    def _hard_update_target(self):
        for src, dst in zip(self.online_net.layers, self.target_net.layers):
            dst.w[:] = src.w
            dst.b[:] = src.b

    def act(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        q_values = self.online_net.forward(state.reshape(1, -1)).flatten()
        return int(np.argmax(q_values))

    def train_step(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        next_q = self.target_net.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        target_q = rewards + self.gamma * max_next_q * (1.0 - dones)

        current_q = self.online_net.forward(states)
        q_sa = current_q[np.arange(self.batch_size), actions]

        loss = np.mean((target_q - q_sa) ** 2)

        grad_q = np.zeros_like(current_q)
        grad_q[np.arange(self.batch_size), actions] = 2.0 * (q_sa - target_q) / self.batch_size

        self.online_net.backward(grad_q)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._step_counter += 1
        if self._step_counter % self.target_update_freq == 0:
            self._hard_update_target()

        return float(loss)

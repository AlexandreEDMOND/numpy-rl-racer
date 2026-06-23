import numpy as np

from numpy_rl_racer.agent.dqn import DQNAgent, ReplayBuffer, N_ACTIONS
from numpy_rl_racer.network import Dense, MLP, SGD


def test_replay_buffer_push_and_len():
    buf = ReplayBuffer(capacity=5)
    assert len(buf) == 0
    for i in range(5):
        buf.push(np.array([float(i)]), i, float(i), np.array([float(i + 1)]), False)
    assert len(buf) == 5


def test_replay_buffer_overflow():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(np.array([float(i)]), i, float(i), np.array([float(i + 1)]), False)
    assert len(buf) == 3


def test_replay_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=10)
    for i in range(10):
        buf.push(np.array([float(i)]), i, float(i), np.array([float(i + 1)]), False)
    states, actions, rewards, next_states, dones = buf.sample(4)
    assert states.shape == (4, 1)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 1)
    assert dones.shape == (4,)


def test_dqn_act_greedy():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3)
    agent.epsilon = 0.0
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action = agent.act(state, training=False)
    assert 0 <= action < N_ACTIONS


def test_dqn_act_exploration():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3)
    agent.epsilon = 1.0
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    actions = set()
    for _ in range(200):
        actions.add(agent.act(state, training=True))
    assert actions == set(range(N_ACTIONS))


def test_dqn_training_step_runs():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, batch_size=4)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        action = agent.act(state, training=True)
        next_state = state + np.random.randn(6) * 0.01
        reward = 0.1
        done = False
        agent.train_step(state, action, reward, next_state, done)


def test_dqn_training_loss_decreases():
    np.random.seed(2)
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-2, batch_size=16)
    agent.epsilon = 0.5
    losses = []
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(150):
        action = agent.act(state, training=True)
        next_state = state + np.random.randn(6) * 0.01
        reward = 0.1
        done = False
        loss = agent.train_step(state, action, reward, next_state, done)
        if loss > 0:
            losses.append(loss)
    if len(losses) >= 40:
        assert np.mean(losses[-20:]) < np.mean(losses[:20])


def test_dense_backward_shape():
    layer = Dense(4, 8)
    x = np.random.randn(3, 4)
    layer.forward(x)
    grad = np.random.randn(3, 8)
    grad_in = layer.backward(grad)
    assert grad_in.shape == (3, 4)
    assert layer.grad_w.shape == (4, 8)
    assert layer.grad_b.shape == (8,)


def test_mlp_backward_shape():
    mlp = MLP([4, 8, 2])
    x = np.random.randn(3, 4)
    mlp.forward(x)
    grad = np.random.randn(3, 2)
    grad_in = mlp.backward(grad)
    assert grad_in.shape == (3, 4)
    for layer in mlp.layers:
        assert layer.grad_w is not None
        assert layer.grad_b is not None


def test_sgd_step_changes_weights():
    mlp = MLP([4, 8, 2])
    opt = SGD(mlp, lr=0.01)
    x = np.random.randn(2, 4)
    mlp.forward(x)
    grad = np.random.randn(2, 2)
    mlp.backward(grad)
    old_w = mlp.layers[0].w.copy()
    opt.step()
    assert not np.allclose(mlp.layers[0].w, old_w)


def test_dqn_save_and_load(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3)
    for layer in agent.online_net.layers:
        layer.w[:] = 1.0
        layer.b[:] = 2.0

    path = str(tmp_path / "test_model.npz")
    agent.save(path)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3)
    agent2.load(path)

    for l1, l2 in zip(agent.online_net.layers, agent2.online_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)

    for l1, l2 in zip(agent.online_net.layers, agent2.target_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)


def test_hard_update_target():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3)
    for src, dst in zip(agent.online_net.layers, agent.target_net.layers):
        dst.w[:] = 0.0
        dst.b[:] = 0.0
    agent._hard_update_target()
    for src, dst in zip(agent.online_net.layers, agent.target_net.layers):
        np.testing.assert_array_equal(src.w, dst.w)
        np.testing.assert_array_equal(src.b, dst.b)


def test_double_dqn_target_computation():
    batch_size = 4
    agent = DQNAgent(state_dim=1, hidden_sizes=[1], lr=0.0,
                     batch_size=batch_size, buffer_size=batch_size)

    for layer in agent.online_net.layers + agent.target_net.layers:
        layer.w[:] = 0.0
        layer.b[:] = 0.0

    agent.online_net.layers[-1].b[:] = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    agent.target_net.layers[-1].b[:] = np.array([0.9, 0.0, 0.0, 0.0, 0.5])

    for _ in range(batch_size - 1):
        agent.replay_buffer.push(np.array([0.0]), 0, 0.0, np.array([0.0]), False)

    loss = agent.train_step(np.array([0.0]), 0, 0.0, np.array([0.0]), False)

    gamma = agent.gamma
    online_out = agent.online_net.forward(np.array([[0.0]]))[0]
    best_a = int(np.argmax(online_out))
    target_out = agent.target_net.forward(np.array([[0.0]]))[0]
    expected_target = 0.0 + gamma * target_out[best_a]
    q_sa = online_out[0]
    expected_loss = np.mean((expected_target - q_sa) ** 2)

    assert np.isclose(loss, expected_loss, rtol=1e-6), (
        f"Loss {loss} does not match Double DQN target {expected_loss}"
    )


def test_standard_dqn_target_computation_when_disabled():
    batch_size = 4
    agent = DQNAgent(
        state_dim=1,
        hidden_sizes=[1],
        lr=0.0,
        batch_size=batch_size,
        buffer_size=batch_size,
        use_double_dqn=False,
    )

    for layer in agent.online_net.layers + agent.target_net.layers:
        layer.w[:] = 0.0
        layer.b[:] = 0.0

    agent.online_net.layers[-1].b[:] = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    agent.target_net.layers[-1].b[:] = np.array([0.9, 0.0, 0.0, 0.0, 0.5])

    for _ in range(batch_size - 1):
        agent.replay_buffer.push(np.array([0.0]), 0, 0.0, np.array([0.0]), False)

    loss = agent.train_step(np.array([0.0]), 0, 0.0, np.array([0.0]), False)

    expected_target = agent.gamma * 0.9
    q_sa = 0.0
    expected_loss = np.mean((expected_target - q_sa) ** 2)

    assert np.isclose(loss, expected_loss, rtol=1e-6), (
        f"Loss {loss} does not match standard DQN target {expected_loss}"
    )

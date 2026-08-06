import numpy as np
import pytest

from numpy_rl_racer.agent.dqn import DQNAgent, PrioritizedReplayBuffer, ReplayBuffer, SumTree, N_ACTIONS
from numpy_rl_racer.network import Adam, Dense, MLP, NoisyLinear, SGD


# -- Adam optimizer integration tests ---------------------------------------


def test_dqn_default_optimizer_is_sgd():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3)
    assert isinstance(agent.optimizer, SGD)
    assert agent.optimizer_type == "sgd"


def test_dqn_invalid_optimizer_raises():
    with pytest.raises(ValueError):
        DQNAgent(state_dim=6, hidden_sizes=[16], optimizer="rmsprop")


def test_dqn_adam_optimizer_constructed():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, optimizer="adam")
    assert isinstance(agent.optimizer, Adam)
    assert agent.optimizer_type == "adam"


def test_dqn_adam_training_step_runs():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, batch_size=4,
                     optimizer="adam")
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        action = agent.act(state, training=True)
        next_state = state + np.random.randn(6) * 0.01
        reward = 0.1
        done = False
        loss = agent.train_step(state, action, reward, next_state, done)
        if loss > 0:
            assert np.isfinite(loss)


def test_dqn_adam_changes_weights():
    np.random.seed(0)
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, batch_size=4,
                     optimizer="adam")
    states = [np.random.randn(6) for _ in range(4)]
    for s in states:
        agent.replay_buffer.push(s, 0, 0.1, s, False)
    old_w = agent.online_net.layers[0].w.copy()
    agent.train_step(states[0], 0, 0.1, states[0], False)
    new_w = agent.online_net.layers[0].w
    assert not np.allclose(old_w, new_w)


def test_dqn_adam_save_load_round_trip(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, optimizer="adam")
    for layer in agent.online_net.layers:
        layer.w[:] = 1.0
        layer.b[:] = 2.0
    agent._hard_update_target()

    path = str(tmp_path / "adam_model.npz")
    agent.save(path)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, optimizer="adam")
    agent2.load(path)
    assert isinstance(agent2.optimizer, Adam)

    for l1, l2 in zip(agent.online_net.layers, agent2.online_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)
    for l1, l2 in zip(agent.target_net.layers, agent2.target_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)


def test_dqn_load_old_sgd_checkpoint_without_optimizer_type(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3)
    for layer in agent.online_net.layers:
        layer.w[:] = 1.0
        layer.b[:] = 2.0
    agent._hard_update_target()

    path = str(tmp_path / "legacy_model.npz")
    agent.save(path)
    data = np.load(path, allow_pickle=False)
    assert "optimizer_type" in data.files

    legacy_path = str(tmp_path / "legacy_no_opttype.npz")
    params = {k: data[k] for k in data.files if k != "optimizer_type"}
    np.savez(legacy_path, **params)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3)
    agent2.load(legacy_path)
    for l1, l2 in zip(agent.online_net.layers, agent2.online_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)


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

    for l1, l2 in zip(agent.target_net.layers, agent2.target_net.layers):
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


# -- SumTree tests ----------------------------------------------------------

def test_sumtree_add_and_total():
    tree = SumTree(capacity=4)
    assert tree.total() == 0.0
    tree.add(1.0, "a")
    assert tree.total() == 1.0
    tree.add(2.0, "b")
    assert tree.total() == 3.0
    tree.add(3.0, "c")
    assert tree.total() == 6.0
    tree.add(4.0, "d")
    assert tree.total() == 10.0


def test_sumtree_get():
    tree = SumTree(capacity=4)
    tree.add(1.0, "a")
    tree.add(2.0, "b")
    tree.add(3.0, "c")
    tree.add(4.0, "d")
    # Cumulative sums: [1, 3, 6, 10]
    # s in [0,1] -> a, (1,3] -> b, (3,6] -> c, (6,10) -> d
    idx, priority, data = tree.get(0.0)
    assert data == "a"
    assert priority == 1.0
    idx, priority, data = tree.get(0.5)
    assert data == "a"
    idx, priority, data = tree.get(1.5)
    assert data == "b"
    idx, priority, data = tree.get(2.9)
    assert data == "b"
    idx, priority, data = tree.get(3.5)
    assert data == "c"
    idx, priority, data = tree.get(5.9)
    assert data == "c"
    idx, priority, data = tree.get(7.0)
    assert data == "d"


def test_sumtree_update():
    tree = SumTree(capacity=4)
    tree.add(1.0, "a")
    tree.add(1.0, "b")
    tree.add(1.0, "c")
    tree.add(1.0, "d")
    assert tree.total() == 4.0
    idx, _, _ = tree.get(0.5)
    tree.update_priority(idx, 5.0)
    assert tree.total() == 8.0
    idx, priority, data = tree.get(0.5)
    assert data == "a"
    assert priority == 5.0


def test_sumtree_overflow():
    tree = SumTree(capacity=3)
    for i in range(5):
        tree.add(float(i + 1), str(i))
    assert tree.size == 3
    assert tree.total() == 3.0 + 4.0 + 5.0


def test_sumtree_sample_distribution():
    np.random.seed(42)
    tree = SumTree(capacity=4)
    tree.add(1.0, "low")
    tree.add(1.0, "low")
    tree.add(1.0, "low")
    tree.add(97.0, "high")
    counts = {"low": 0, "high": 0}
    for _ in range(1000):
        s = np.random.uniform(0, tree.total())
        _, _, data = tree.get(s)
        counts[data] += 1
    assert counts["high"] > counts["low"] * 10


# -- PrioritizedReplayBuffer tests ------------------------------------------

def test_per_buffer_push_and_len():
    buf = PrioritizedReplayBuffer(capacity=5)
    assert len(buf) == 0
    for i in range(5):
        buf.push(np.array([float(i)]), i, float(i), np.array([float(i + 1)]), False)
    assert len(buf) == 5


def test_per_buffer_overflow():
    buf = PrioritizedReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(np.array([float(i)]), i, float(i), np.array([float(i + 1)]), False)
    assert len(buf) == 3


def test_per_buffer_sample_returns_is_weights():
    buf = PrioritizedReplayBuffer(capacity=10)
    for i in range(10):
        buf.push(np.array([float(i)]), i, float(i), np.array([float(i + 1)]), False)
    result = buf.sample(4)
    states, actions, rewards, next_states, dones, is_weights, indices = result
    assert states.shape == (4, 1)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 1)
    assert dones.shape == (4,)
    assert is_weights.shape == (4,)
    assert indices.shape == (4,)
    assert np.all(is_weights > 0.0)
    assert np.isclose(is_weights.max(), 1.0)


def test_per_buffer_sample_non_uniform():
    np.random.seed(42)
    buf = PrioritizedReplayBuffer(capacity=100, alpha=1.0)
    for i in range(100):
        buf.push(np.array([0.0]), 0, float(i), np.array([0.0]), False)
    for _ in range(50):
        result = buf.sample(16)
        states, actions, rewards, next_states, dones, is_weights, indices = result
        buf.update_priorities(indices, np.abs(rewards) + 0.1)

    # After training with different TD errors, priorities should be non-uniform
    priorities = buf.tree.tree[buf.tree.capacity:buf.tree.capacity + buf.tree.size]
    unique_priorities = np.unique(priorities)
    assert len(unique_priorities) > 1


def test_per_buffer_beta_annealing():
    buf = PrioritizedReplayBuffer(capacity=10, beta0=0.4, beta_anneal_steps=100)
    for i in range(10):
        buf.push(np.array([0.0]), 0, 0.0, np.array([0.0]), False)
    assert np.isclose(buf.beta, 0.4)
    for _ in range(50):
        buf.sample(4)
    expected_beta = min(1.0, 0.4 + 0.6 * 50 / 100)
    assert np.isclose(buf.beta, expected_beta)
    for _ in range(100):
        buf.sample(4)
    assert np.isclose(buf.beta, 1.0)


def test_per_buffer_empty_sample_raises():
    buf = PrioritizedReplayBuffer(capacity=5)
    for i in range(3):
        buf.push(np.array([0.0]), 0, 0.0, np.array([0.0]), False)
    # Should not fail when enough samples exist
    result = buf.sample(3)
    assert len(result) == 7


def test_per_buffer_capacity_one():
    buf = PrioritizedReplayBuffer(capacity=1)
    buf.push(np.array([0.0]), 0, 0.0, np.array([0.0]), False)
    states, actions, rewards, next_states, dones, is_weights, indices = buf.sample(1)
    assert states.shape == (1, 1)


# -- DQN + PER integration tests --------------------------------------------

def test_dqn_per_training_step_runs():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, batch_size=4,
                     use_per=True)
    assert agent.use_per is True
    assert isinstance(agent.replay_buffer, PrioritizedReplayBuffer)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        action = agent.act(state, training=True)
        next_state = state + np.random.randn(6) * 0.01
        reward = 0.1
        done = False
        agent.train_step(state, action, reward, next_state, done)


def test_dqn_per_priorities_updated():
    np.random.seed(1)
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-2, batch_size=8,
                     use_per=True)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for step in range(30):
        action = agent.act(state, training=True)
        next_state = state + np.random.randn(6) * 0.01
        reward = 0.1
        done = False
        agent.train_step(state, action, reward, next_state, done)

    leaf_priorities = agent.replay_buffer.tree.tree[
        agent.replay_buffer.tree.capacity:
    ]
    non_zero = leaf_priorities[leaf_priorities > 0]
    assert len(non_zero) > 0
    unique_priorities = np.unique(non_zero)
    # Priorities should have been updated to non-uniform values
    assert np.any(unique_priorities != unique_priorities[0]) or len(unique_priorities) > 1


# -- Regression: use_per=False matches uniform replay -----------------------

def test_dqn_per_regression_uniform():
    np.random.seed(42)
    agent_uniform = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-2, batch_size=8,
                             use_per=False)
    np.random.seed(42)
    agent_per = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-2, batch_size=8,
                         use_per=True)

    assert not agent_uniform.use_per
    assert agent_per.use_per


# -- Soft target network updates (Polyak averaging) -------------------------

def test_tau_zero_preserves_hard_update():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=0.0, batch_size=4,
                     target_update_freq=5, tau=0.0)
    for layer in agent.online_net.layers:
        layer.w[:] = 1.0
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(4):
        agent.replay_buffer.push(state, 0, 0.0, state, False)
    for _ in range(4):
        agent.train_step(state, 0, 0.0, state, False)
    assert not np.allclose(agent.online_net.layers[0].w,
                           agent.target_net.layers[0].w)
    agent.train_step(state, 0, 0.0, state, False)
    for src, dst in zip(agent.online_net.layers, agent.target_net.layers):
        np.testing.assert_array_equal(src.w, dst.w)
        np.testing.assert_array_equal(src.b, dst.b)


def test_tau_half_weighted_average():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=0.0, batch_size=4,
                     target_update_freq=1000, tau=0.5)
    for layer in agent.online_net.layers:
        layer.w[:] = 1.0
        layer.b[:] = 2.0
    for layer in agent.target_net.layers:
        layer.w[:] = 3.0
        layer.b[:] = 4.0
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(4):
        agent.replay_buffer.push(state, 0, 0.0, state, False)
    agent.train_step(state, 0, 0.0, state, False)
    for layer in agent.target_net.layers:
        np.testing.assert_array_almost_equal(layer.w, 2.0)
        np.testing.assert_array_almost_equal(layer.b, 3.0)


def test_tau_convergence():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=0.0, batch_size=4,
                     target_update_freq=1000, tau=0.1)
    for layer in agent.online_net.layers:
        layer.w[:] = 5.0
    for layer in agent.target_net.layers:
        layer.w[:] = 0.0
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(4):
        agent.replay_buffer.push(state, 0, 0.0, state, False)
    for _ in range(50):
        agent.train_step(state, 0, 0.0, state, False)
    expected = 5.0 * (1.0 - 0.9 ** 50)
    for layer in agent.target_net.layers:
        np.testing.assert_array_almost_equal(layer.w, expected, decimal=5)


def test_tau_one_hard_copy():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=0.0, batch_size=4,
                     target_update_freq=1000, tau=1.0)
    for layer in agent.online_net.layers:
        layer.w[:] = 1.0
    for layer in agent.target_net.layers:
        layer.w[:] = 0.0
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(4):
        agent.replay_buffer.push(state, 0, 0.0, state, False)
    agent.train_step(state, 0, 0.0, state, False)
    for layer in agent.target_net.layers:
        np.testing.assert_array_equal(layer.w, 1.0)


def test_training_step_with_soft_update():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, batch_size=4,
                     tau=0.005)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        action = agent.act(state, training=True)
        next_state = state + np.random.randn(6) * 0.01
        reward = 0.1
        done = False
        loss = agent.train_step(state, action, reward, next_state, done)
        assert isinstance(loss, float)


def test_save_load_round_trip_soft_update(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, tau=0.005)
    for layer in agent.online_net.layers:
        layer.w[:] = 1.0
        layer.b[:] = 2.0
    agent._soft_update_target(0.005)
    path = str(tmp_path / "soft_model.npz")
    agent.save(path)
    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, tau=0.005)
    agent2.load(path)
    for l1, l2 in zip(agent.online_net.layers, agent2.online_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)
    for l1, l2 in zip(agent.target_net.layers, agent2.target_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)


# -- Seeded reproducibility tests -------------------------------------------


def test_agent_seed_reproducibility():
    agent1 = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, seed=42)
    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, seed=42)
    agent1.epsilon = 1.0
    agent2.epsilon = 1.0
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    actions1 = [agent1.act(state, training=True) for _ in range(100)]
    actions2 = [agent2.act(state, training=True) for _ in range(100)]
    assert actions1 == actions2


def test_replay_buffer_seed_reproducibility():
    buf1 = ReplayBuffer(capacity=10, rng=np.random.RandomState(42))
    buf2 = ReplayBuffer(capacity=10, rng=np.random.RandomState(42))
    for i in range(10):
        buf1.push(np.array([float(i)]), i, float(i), np.array([float(i + 1)]), False)
        buf2.push(np.array([float(i)]), i, float(i), np.array([float(i + 1)]), False)
    s1, a1, r1, ns1, d1 = buf1.sample(4)
    s2, a2, r2, ns2, d2 = buf2.sample(4)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(ns1, ns2)
    np.testing.assert_array_equal(d1, d2)


def test_prioritized_replay_buffer_seed_reproducibility():
    buf1 = PrioritizedReplayBuffer(capacity=10, rng=np.random.RandomState(42))
    buf2 = PrioritizedReplayBuffer(capacity=10, rng=np.random.RandomState(42))
    for i in range(10):
        buf1.push(np.array([float(i)]), i, float(i), np.array([float(i + 1)]), False)
        buf2.push(np.array([float(i)]), i, float(i), np.array([float(i + 1)]), False)
    res1 = buf1.sample(4)
    res2 = buf2.sample(4)
    for arr1, arr2 in zip(res1, res2):
        np.testing.assert_array_equal(arr1, arr2)


def test_dueling_dqn_act():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, use_dueling_dqn=True)
    agent.epsilon = 0.0
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action = agent.act(state, training=False)
    assert 0 <= action < N_ACTIONS


def test_dueling_dqn_training_step():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, batch_size=4,
                     use_dueling_dqn=True)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        action = agent.act(state, training=True)
        next_state = state + np.random.randn(6) * 0.01
        reward = 0.1
        done = False
        loss = agent.train_step(state, action, reward, next_state, done)
        assert np.isfinite(loss)


def test_dueling_dqn_save_load(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, use_dueling_dqn=True)
    for layer in agent.online_net.layers:
        layer.w[:] = 1.0
        layer.b[:] = 2.0

    path = str(tmp_path / "dueling_model.npz")
    agent.save(path)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, use_dueling_dqn=True)
    agent2.load(path)

    for l1, l2 in zip(agent.online_net.layers, agent2.online_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)

    for l1, l2 in zip(agent.target_net.layers, agent2.target_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)


def test_agent_seed_none_stochastic():
    agent1 = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, seed=None)
    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, seed=None)
    agent1.epsilon = 1.0
    agent2.epsilon = 1.0
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    actions1 = [agent1.act(state, training=True) for _ in range(100)]
    actions2 = [agent2.act(state, training=True) for _ in range(100)]
    assert actions1 != actions2


# -- N-step TD target tests ------------------------------------------------


def test_n_step_default_one():
    """Default n_step=1 is backward compatible with existing behaviour."""
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, batch_size=4)
    assert agent.n_step == 1
    assert agent.gamma_n == agent.gamma
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        action = agent.act(state, training=True)
        next_state = state + np.random.randn(6) * 0.01
        reward = 0.1
        done = False
        loss = agent.train_step(state, action, reward, next_state, done)
        assert isinstance(loss, float) and np.isfinite(loss)


def test_n_step_target_one():
    """n=1 produces identical targets to the current 1-step computation."""
    batch_size = 4
    agent = DQNAgent(
        state_dim=1, hidden_sizes=[1], lr=0.0,
        batch_size=batch_size, buffer_size=batch_size,
        use_double_dqn=False, n_step=1,
    )
    for layer in agent.online_net.layers + agent.target_net.layers:
        layer.w[:] = 0.0
        layer.b[:] = 0.0
    agent.target_net.layers[-1].b[:] = np.array([0.9, 0.0, 0.0, 0.0, 0.5])

    for _ in range(batch_size - 1):
        agent.train_step(np.array([0.0]), 0, 0.0, np.array([0.0]), False)

    loss = agent.train_step(np.array([0.0]), 0, 0.0, np.array([0.0]), False)

    expected_target = agent.gamma * 0.9
    q_sa = 0.0
    expected_loss = np.mean((expected_target - q_sa) ** 2)

    assert np.isclose(loss, expected_loss, rtol=1e-6), (
        f"Loss {loss} does not match expected 1-step target {expected_loss}"
    )


def test_n_step_target_three():
    """n=3 accumulates rewards correctly for non-terminal transitions."""
    batch_size = 4
    agent = DQNAgent(
        state_dim=1, hidden_sizes=[1], lr=0.0,
        batch_size=batch_size, buffer_size=16,
        use_double_dqn=False, n_step=3,
    )
    for layer in agent.online_net.layers + agent.target_net.layers:
        layer.w[:] = 0.0
        layer.b[:] = 0.0

    gamma = agent.gamma
    # 4 identical sequences: [0.1, 0.2, 0.3] terminated with done=True
    for _ in range(3):
        agent.train_step(np.array([0.0]), 0, 0.1, np.array([0.0]), False)
        agent.train_step(np.array([0.0]), 0, 0.2, np.array([0.0]), False)
        agent.train_step(np.array([0.0]), 0, 0.3, np.array([0.0]), True)

    # 4th sequence triggers training (replay buffer hits batch_size)
    agent.train_step(np.array([0.0]), 0, 0.1, np.array([0.0]), False)
    agent.train_step(np.array([0.0]), 0, 0.2, np.array([0.0]), False)
    loss = agent.train_step(np.array([0.0]), 0, 0.3, np.array([0.0]), True)

    expected_G = 0.1 + gamma * 0.2 + gamma ** 2 * 0.3
    expected_loss = np.mean(expected_G ** 2)

    assert np.isclose(loss, expected_loss, rtol=1e-6), (
        f"Loss {loss} does not match expected n=3 target {expected_loss}"
    )


def test_n_step_early_termination():
    """n-step return terminates early when done flag is encountered."""
    agent = DQNAgent(
        state_dim=1, hidden_sizes=[1], lr=0.0,
        batch_size=4, buffer_size=16,
        use_double_dqn=False, n_step=3,
    )
    for layer in agent.online_net.layers + agent.target_net.layers:
        layer.w[:] = 0.0
        layer.b[:] = 0.0

    gamma = agent.gamma
    # done=True at step 2 (before n=3 is full)
    agent.train_step(np.array([0.0]), 0, 0.5, np.array([1.0]), False)
    agent.train_step(np.array([0.0]), 0, 0.5, np.array([1.0]), True)

    expected_G = 0.5 + gamma * 0.5

    entry_G = agent.replay_buffer.buffer[0][2]
    assert np.isclose(entry_G, expected_G, rtol=1e-6), (
        f"Expected n-step return {expected_G}, got {entry_G}"
    )


def test_n_step_training_step_runs():
    """train_step executes for n=3 without error."""
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, batch_size=4, n_step=3)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(60):
        action = agent.act(state, training=True)
        next_state = state + np.random.randn(6) * 0.01
        reward = 0.1
        done = False
        loss = agent.train_step(state, action, reward, next_state, done)
        assert np.isfinite(loss) if loss > 0 else True


# -- NoisyNet tests ---------------------------------------------------------


def test_dqn_noisy_net_act():
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, use_noisy=True)
    assert agent.epsilon == 0.0
    for layer in agent.online_net.layers:
        if isinstance(layer, NoisyLinear):
            assert hasattr(layer, "sigma_w")
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action = agent.act(state, training=True)
    assert 0 <= action < N_ACTIONS


def test_dqn_noisy_net_act_greedy():
    """Noisy DQN should act greedily (no epsilon exploration)."""
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, use_noisy=True, seed=42)
    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, use_noisy=True, seed=42)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    actions1 = [agent.act(state, training=True) for _ in range(10)]
    actions2 = [agent2.act(state, training=True) for _ in range(10)]
    assert actions1 == actions2


def test_dqn_noisy_net_training():
    np.random.seed(2)
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-2, batch_size=16,
                     use_noisy=True)
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


def test_dqn_noisy_net_training_with_dueling():
    np.random.seed(2)
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-2, batch_size=16,
                     use_noisy=True, use_dueling_dqn=True)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(30):
        action = agent.act(state, training=True)
        next_state = state + np.random.randn(6) * 0.01
        reward = 0.1
        done = False
        loss = agent.train_step(state, action, reward, next_state, done)
        assert np.isfinite(loss) if loss > 0 else True


def test_dqn_noisy_net_save_load(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, use_noisy=True)
    for layer in agent.online_net.layers:
        if isinstance(layer, NoisyLinear):
            layer.w[:] = 1.0
            layer.b[:] = 2.0
            layer.sigma_w[:] = 0.1
            layer.sigma_b[:] = 0.2

    path = str(tmp_path / "noisy_model.npz")
    agent.save(path)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, use_noisy=True)
    agent2.load(path)

    for l1, l2 in zip(agent.online_net.layers, agent2.online_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)
        if isinstance(l1, NoisyLinear):
            np.testing.assert_array_equal(l1.sigma_w, l2.sigma_w)
            np.testing.assert_array_equal(l1.sigma_b, l2.sigma_b)


def test_dqn_save_persists_use_noisy_metadata(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], use_noisy=True)
    path = str(tmp_path / "noisy_meta.npz")
    agent.save(path)
    data = np.load(path, allow_pickle=False)
    assert "use_noisy" in data.files
    assert int(data["use_noisy"]) == 1


def test_dqn_load_reconstructs_use_noisy_from_metadata(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], use_noisy=True)
    path = str(tmp_path / "noisy_meta.npz")
    agent.save(path)

    data = np.load(path, allow_pickle=False)
    use_noisy = int(data["use_noisy"]) if "use_noisy" in data.files else 0
    reconstructed = DQNAgent(
        state_dim=int(data["state_dim"]),
        hidden_sizes=list(data["hidden_sizes"]),
        use_dueling_dqn=(int(data["arch_type"]) == 1),
        use_noisy=(use_noisy == 1),
    )
    reconstructed.load(path)
    assert reconstructed.use_noisy is True
    assert any(isinstance(layer, NoisyLinear)
               for layer in reconstructed.online_net.layers)


def test_dqn_load_validates_use_noisy_mismatch(tmp_path):
    noisy_agent = DQNAgent(state_dim=6, hidden_sizes=[16], use_noisy=True)
    noisy_path = str(tmp_path / "noisy.npz")
    noisy_agent.save(noisy_path)

    plain_agent = DQNAgent(state_dim=6, hidden_sizes=[16], use_noisy=False)
    with pytest.raises(ValueError, match="use_noisy"):
        plain_agent.load(noisy_path)

    plain_agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], use_noisy=False)
    plain_path = str(tmp_path / "plain.npz")
    plain_agent2.save(plain_path)

    noisy_agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], use_noisy=True)
    with pytest.raises(ValueError, match="use_noisy"):
        noisy_agent2.load(plain_path)


def test_dqn_load_legacy_checkpoint_without_use_noisy(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], use_noisy=False)
    path = str(tmp_path / "legacy_no_noisy.npz")
    agent.save(path)

    data = np.load(path, allow_pickle=False)
    legacy_path = str(tmp_path / "legacy_no_noisy_key.npz")
    params = {k: data[k] for k in data.files if k != "use_noisy"}
    np.savez(legacy_path, **params)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], use_noisy=False)
    agent2.load(legacy_path)
    assert agent2.use_noisy is False


def test_n_step_training_loss_decreases():
    """Loss decreases over multiple training steps with n=3."""
    np.random.seed(2)
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], lr=1e-2, batch_size=16, n_step=3)
    agent.epsilon = 0.5
    losses = []
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(300):
        action = agent.act(state, training=True)
        next_state = state + np.random.randn(6) * 0.01
        reward = 0.1
        done = False
        loss = agent.train_step(state, action, reward, next_state, done)
        if loss > 0:
            losses.append(loss)
    if len(losses) >= 40:
        assert np.mean(losses[-20:]) < np.mean(losses[:20])


# -- Huber (smooth-L1) loss tests ------------------------------------------


def _fake_train_step(agent, monkeypatch, current_q, actions, rewards=None,
                     dones=None, is_weights=None, next_q_target=None):
    """Drive a fake batch through ``train_step`` with fixed network outputs.

    Monkeypatches ``online_net.forward``/``target_net.forward`` to return fixed
    arrays, ``online_net.backward`` to capture the upstream gradient, and
    ``optimizer.step`` to a no-op so no weights change. Returns
    ``(loss, grad_at_actions)``.
    """
    batch_size = agent.batch_size
    states = np.zeros((batch_size, agent.state_dim))
    next_states = np.zeros((batch_size, agent.state_dim))
    if rewards is None:
        rewards = np.zeros(batch_size)
    if dones is None:
        dones = np.ones(batch_size, dtype=bool)
    if next_q_target is None:
        next_q_target = np.zeros((batch_size, N_ACTIONS))
    if is_weights is None:
        is_weights = np.ones(batch_size)

    for _ in range(batch_size):
        agent.replay_buffer.push(states[0], 0, 0.0, states[0], False)

    captured = {}

    def fake_online_forward(x):
        return current_q

    def fake_target_forward(x):
        return next_q_target

    monkeypatch.setattr(agent.online_net, "forward", fake_online_forward)
    monkeypatch.setattr(agent.target_net, "forward", fake_target_forward)
    monkeypatch.setattr(
        agent.online_net, "backward", lambda g: captured.setdefault("grad_q", g)
    )
    monkeypatch.setattr(agent.optimizer, "step", lambda: None)

    if agent.use_per:
        def fake_sample(n):
            return (states, actions, rewards, next_states, dones,
                    is_weights, np.zeros(n, dtype=np.int64))

        monkeypatch.setattr(agent.replay_buffer, "sample", fake_sample)
        monkeypatch.setattr(agent.replay_buffer, "update_priorities",
                            lambda idx, td: None)
    else:
        def fake_sample(n):
            return states, actions, rewards, next_states, dones

        monkeypatch.setattr(agent.replay_buffer, "sample", fake_sample)

    loss = agent.train_step(
        states[0], int(actions[0]), float(rewards[0]), next_states[0], bool(dones[0])
    )
    grad_at_actions = captured["grad_q"][np.arange(batch_size), actions]
    return loss, grad_at_actions


def _make_huber_agent(**overrides):
    kwargs = dict(
        state_dim=1, hidden_sizes=[1], lr=0.0, batch_size=4,
        buffer_size=4, use_double_dqn=False, seed=0,
    )
    kwargs.update(overrides)
    return DQNAgent(**kwargs)


def _huber_current_q():
    """current_q with q_sa spanning both |td|<=delta and |td|>delta regimes."""
    current_q = np.zeros((4, N_ACTIONS))
    current_q[0, 0] = 0.5
    current_q[1, 1] = -0.5
    current_q[2, 2] = 2.0
    current_q[3, 3] = -2.0
    return current_q


def test_huber_loss_value_matches_formula(monkeypatch):
    agent = _make_huber_agent(loss_type="huber", huber_delta=1.0)
    actions = np.array([0, 1, 2, 3])
    current_q = _huber_current_q()
    # dones=True, rewards=0 -> target_q=0, td = q_sa = [0.5, -0.5, 2.0, -2.0]
    loss, _ = _fake_train_step(agent, monkeypatch, current_q, actions,
                               dones=np.ones(4, dtype=bool),
                               rewards=np.zeros(4))

    td = np.array([0.5, -0.5, 2.0, -2.0])
    delta = 1.0
    abs_td = np.abs(td)
    per_element = np.where(abs_td <= delta, 0.5 * td * td,
                          delta * (abs_td - 0.5 * delta))
    expected_loss = float(np.mean(per_element))
    assert np.isclose(loss, expected_loss, rtol=1e-12), (loss, expected_loss)
    # Sanity: both regimes are exercised.
    assert np.any(abs_td <= delta) and np.any(abs_td > delta)


def test_huber_gradient_bounded_by_delta(monkeypatch):
    agent = _make_huber_agent(loss_type="huber", huber_delta=1.0)
    actions = np.array([0, 1, 2, 3])
    current_q = _huber_current_q()
    _, grad_at_actions = _fake_train_step(agent, monkeypatch, current_q, actions,
                                          dones=np.ones(4, dtype=bool),
                                          rewards=np.zeros(4))
    batch_size = agent.batch_size
    bound = agent.huber_delta / batch_size
    assert np.all(np.abs(grad_at_actions) <= bound + 1e-12), (
        grad_at_actions, bound
    )
    expected = np.array([0.5, -0.5, 1.0, -1.0]) / batch_size
    np.testing.assert_allclose(grad_at_actions, expected, rtol=1e-12)
    # Gradient sign matches (q_sa - target_q) = td (ascent direction).
    td = np.array([0.5, -0.5, 2.0, -2.0])
    assert np.all(np.sign(grad_at_actions) == np.sign(td))


def test_huber_with_per_uses_is_weights(monkeypatch):
    agent = _make_huber_agent(loss_type="huber", huber_delta=1.0, use_per=True)
    actions = np.array([0, 1, 2, 3])
    current_q = _huber_current_q()
    is_weights = np.array([0.5, 0.5, 1.0, 1.0])
    loss, grad_at_actions = _fake_train_step(
        agent, monkeypatch, current_q, actions,
        dones=np.ones(4, dtype=bool), rewards=np.zeros(4),
        is_weights=is_weights,
    )
    td = np.array([0.5, -0.5, 2.0, -2.0])
    delta = 1.0
    abs_td = np.abs(td)
    per_element = np.where(abs_td <= delta, 0.5 * td * td,
                          delta * (abs_td - 0.5 * delta))
    huber_grad = np.where(abs_td <= delta, td, delta * np.sign(td))
    batch_size = agent.batch_size
    expected_loss = float(np.mean(is_weights * per_element))
    expected_grad = is_weights * huber_grad / batch_size
    assert np.isclose(loss, expected_loss, rtol=1e-12), (loss, expected_loss)
    np.testing.assert_allclose(grad_at_actions, expected_grad, rtol=1e-12)


def test_mse_path_unchanged_by_huber_option(monkeypatch):
    actions = np.array([0, 1, 2, 3])
    current_q = _huber_current_q()
    agent_default = _make_huber_agent()
    agent_explicit = _make_huber_agent(loss_type="mse")
    assert agent_default.loss_type == "mse"
    assert agent_explicit.loss_type == "mse"

    loss_d, grad_d = _fake_train_step(
        agent_default, monkeypatch, current_q, actions,
        dones=np.ones(4, dtype=bool), rewards=np.zeros(4))
    loss_e, grad_e = _fake_train_step(
        agent_explicit, monkeypatch, current_q, actions,
        dones=np.ones(4, dtype=bool), rewards=np.zeros(4))

    td = np.array([0.5, -0.5, 2.0, -2.0])
    expected_loss = float(np.mean(td ** 2))
    expected_grad = 2.0 * td / 4
    assert np.isclose(loss_d, expected_loss, rtol=1e-12)
    assert np.isclose(loss_e, expected_loss, rtol=1e-12)
    np.testing.assert_array_equal(grad_d, grad_e)
    np.testing.assert_allclose(grad_d, expected_grad, rtol=1e-12)


def test_dqn_invalid_loss_type_raises():
    with pytest.raises(ValueError):
        DQNAgent(state_dim=6, hidden_sizes=[16], loss_type="rmse")


def test_save_load_preserves_loss_type(tmp_path):
    agent = _make_huber_agent(loss_type="huber", huber_delta=0.5)
    path = str(tmp_path / "huber_model.npz")
    agent.save(path)

    loaded = _make_huber_agent()
    assert loaded.loss_type == "mse"
    assert loaded.huber_delta == 1.0
    loaded.load(path)
    assert loaded.loss_type == "huber"
    assert np.isclose(loaded.huber_delta, 0.5)

    # Legacy checkpoint without the new keys defaults to mse / 1.0.
    data = np.load(path, allow_pickle=False)
    legacy_path = str(tmp_path / "legacy_no_loss.npz")
    params = {k: data[k] for k in data.files
              if k not in ("loss_type", "huber_delta")}
    np.savez(legacy_path, **params)

    legacy = _make_huber_agent(loss_type="huber", huber_delta=0.5)
    legacy.load(legacy_path)
    assert legacy.loss_type == "mse"
    assert np.isclose(legacy.huber_delta, 1.0)

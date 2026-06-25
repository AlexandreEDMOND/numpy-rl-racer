import numpy as np
import pytest

from numpy_rl_racer.agent.reinforce import REINFORCEAgent, N_ACTIONS, discounted_sum
from numpy_rl_racer.network import PolicyNetwork, SGD, Adam, MLP
from numpy_rl_racer.utils.scheduler import ExponentialDecay


# -- PolicyNetwork tests -----------------------------------------------------

def test_policy_network_forward_shape():
    net = PolicyNetwork([4, 8, N_ACTIONS])
    x = np.random.randn(3, 4)
    out = net.forward(x)
    assert out.shape == (3, N_ACTIONS)


def test_policy_network_probs_sum_to_one():
    net = PolicyNetwork([4, 8, N_ACTIONS])
    x = np.random.randn(5, 4)
    logits = net.forward(x)
    probs = net.get_probs(logits)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-10)


def test_policy_network_probs_positive():
    net = PolicyNetwork([2, 4, N_ACTIONS])
    x = np.array([[100.0, -100.0], [-100.0, 100.0]])
    logits = net.forward(x)
    probs = net.get_probs(logits)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_policy_network_log_prob_valid():
    net = PolicyNetwork([4, 8, N_ACTIONS])
    x = np.random.randn(3, 4)
    logits = net.forward(x)
    actions = np.array([0, 2, 4])
    lp = net.log_prob(actions, logits)
    assert lp.shape == (3,)
    assert np.all(lp <= 0.0)
    assert np.all(np.isfinite(lp))


def test_policy_network_log_prob_correct():
    net = PolicyNetwork([2, 4, 3])
    logits = np.array([[2.0, 1.0, 0.1]])
    probs = net.get_probs(logits)
    expected_lp = np.log(probs[0, 0])
    lp = net.log_prob(np.array([0]), logits)
    assert np.isclose(lp[0], expected_lp)


def test_policy_network_layers_property():
    net = PolicyNetwork([4, 8, N_ACTIONS])
    assert len(net.layers) == 2
    assert net.layers[0].w.shape == (4, 8)
    assert net.layers[1].w.shape == (8, N_ACTIONS)


def test_policy_network_backward():
    net = PolicyNetwork([4, 8, N_ACTIONS])
    x = np.random.randn(3, 4)
    net.forward(x)
    grad = np.random.randn(3, N_ACTIONS)
    net.backward(grad)
    for layer in net.layers:
        assert layer.grad_w is not None
        assert layer.grad_b is not None


# -- discounted_sum tests ----------------------------------------------------

def test_discounted_sum_gamma_zero():
    rewards = np.array([1.0, 2.0, 3.0])
    result = discounted_sum(rewards, 0.0)
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(result, expected)


def test_discounted_sum_gamma_one():
    rewards = np.array([1.0, 2.0, 3.0])
    result = discounted_sum(rewards, 1.0)
    expected = np.array([6.0, 5.0, 3.0])
    np.testing.assert_allclose(result, expected)


def test_discounted_sum_gamma_point_five():
    rewards = np.array([1.0, 2.0, 3.0])
    result = discounted_sum(rewards, 0.5)
    G0 = 1.0 + 0.5 * 2.0 + 0.25 * 3.0
    G1 = 2.0 + 0.5 * 3.0
    G2 = 3.0
    np.testing.assert_allclose(result, np.array([G0, G1, G2]))


def test_discounted_sum_single_step():
    rewards = np.array([5.0])
    result = discounted_sum(rewards, 0.99)
    assert np.isclose(result[0], 5.0)


def test_discounted_sum_long_episode():
    T = 50
    rewards = np.ones(T)
    gamma = 0.9
    result = discounted_sum(rewards, gamma)
    expected = np.array([sum(gamma ** (k - t) for k in range(t, T)) for t in range(T)])
    np.testing.assert_allclose(result, expected, rtol=1e-10)


# -- REINFORCEAgent construction tests ---------------------------------------

def test_agent_construct_default():
    agent = REINFORCEAgent(state_dim=6)
    assert agent.state_dim == 6
    assert agent.hidden_sizes == [64, 64]
    assert agent.gamma == 0.99
    assert agent.use_baseline is True
    assert agent.use_value_network is False
    assert len(agent.policy_net.layers) == 3
    assert isinstance(agent.optimizer, SGD)


def test_agent_construct_custom_hidden():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[32, 32])
    assert agent.hidden_sizes == [32, 32]
    assert agent.policy_net.layers[0].w.shape == (6, 32)


def test_agent_construct_with_adam():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], optimizer_type="adam")
    assert isinstance(agent.optimizer, Adam)


def test_agent_construct_with_value_network():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], use_value_network=True)
    assert agent.use_value_network
    assert hasattr(agent, 'value_net')
    assert isinstance(agent.value_net, MLP)
    assert agent.value_net.layers[-1].w.shape == (16, 1)


def test_agent_construct_with_momentum():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], momentum=0.9)
    assert agent.optimizer.momentum == 0.9


def test_agent_construct_with_weight_decay():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], weight_decay=0.01)
    for layer in agent.policy_net.layers:
        assert layer.weight_decay == 0.01


# -- REINFORCEAgent action tests ---------------------------------------------

def test_agent_act_eval_greedy():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], seed=42)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action = agent.act(state, training=False)
    assert 0 <= action < N_ACTIONS


def test_agent_act_training_samples_all_actions():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], seed=42)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    actions = set()
    for _ in range(500):
        actions.add(agent.act(state, training=True))
    assert actions == set(range(N_ACTIONS))


def test_agent_act_seed_reproducibility():
    agent1 = REINFORCEAgent(state_dim=6, hidden_sizes=[16], seed=42)
    agent2 = REINFORCEAgent(state_dim=6, hidden_sizes=[16], seed=42)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    actions1 = [agent1.act(state, training=True) for _ in range(100)]
    actions2 = [agent2.act(state, training=True) for _ in range(100)]
    assert actions1 == actions2


# -- REINFORCEAgent training tests -------------------------------------------

def test_agent_train_step_runs():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], lr=1e-3)
    T = 20
    states = np.random.randn(T, 6)
    actions = np.random.randint(0, N_ACTIONS, size=T)
    rewards = np.random.randn(T) * 0.1
    loss = agent.train_step(states, actions, rewards)
    assert np.isfinite(loss)


def test_agent_training_loss_finite():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], lr=1e-2)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        states_list = [state + np.random.randn(6) * 0.01 for _ in range(10)]
        actions = np.random.randint(0, N_ACTIONS, size=10)
        rewards = np.full(10, 1.0)
        loss = agent.train_step(np.array(states_list), actions, rewards)
        assert np.isfinite(loss)


def test_agent_training_with_sgd_optimizer():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], lr=1e-2, optimizer_type="sgd")
    T = 20
    states = np.random.randn(T, 6)
    actions = np.random.randint(0, N_ACTIONS, size=T)
    rewards = np.random.randn(T) * 0.1
    loss = agent.train_step(states, actions, rewards)
    assert np.isfinite(loss)


def test_agent_training_with_adam_optimizer():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], lr=1e-2, optimizer_type="adam")
    T = 20
    states = np.random.randn(T, 6)
    actions = np.random.randint(0, N_ACTIONS, size=T)
    rewards = np.random.randn(T) * 0.1
    loss = agent.train_step(states, actions, rewards)
    assert np.isfinite(loss)


def test_agent_training_no_baseline():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, use_baseline=False)
    T = 10
    states = np.random.randn(T, 6)
    actions = np.random.randint(0, N_ACTIONS, size=T)
    rewards = np.random.randn(T) * 0.1
    loss = agent.train_step(states, actions, rewards)
    assert np.isfinite(loss)


def test_agent_training_with_value_network():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, use_value_network=True)
    T = 10
    states = np.random.randn(T, 6)
    actions = np.random.randint(0, N_ACTIONS, size=T)
    rewards = np.random.randn(T) * 0.1
    result = agent.train_step(states, actions, rewards)
    assert len(result) == 2
    loss, v_loss = result
    assert np.isfinite(loss)
    assert np.isfinite(v_loss)


# -- Gradient clipping tests -------------------------------------------------

def test_agent_gradient_clipping_sgd():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], lr=0.01, max_grad_norm=1.0)
    T = 10
    states = np.random.randn(T, 6)
    actions = np.random.randint(0, N_ACTIONS, size=T)
    rewards = np.random.randn(T) * 100.0
    loss = agent.train_step(states, actions, rewards)
    assert np.isfinite(loss)


def test_agent_gradient_clipping_adam():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], lr=0.01,
                           optimizer_type="adam", max_grad_norm=1.0)
    T = 10
    states = np.random.randn(T, 6)
    actions = np.random.randint(0, N_ACTIONS, size=T)
    rewards = np.random.randn(T) * 100.0
    loss = agent.train_step(states, actions, rewards)
    assert np.isfinite(loss)


# -- LR scheduler tests ------------------------------------------------------

def test_agent_lr_scheduler_exponential():
    sched = ExponentialDecay(0.1, 0.5)
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], scheduler=sched)
    lr_before = agent.optimizer.lr
    T = 5
    states = np.random.randn(T, 6)
    actions = np.random.randint(0, N_ACTIONS, size=T)
    rewards = np.random.randn(T) * 0.1
    agent.train_step(states, actions, rewards)
    assert np.isclose(agent.optimizer.lr, lr_before * 0.5)


def test_agent_lr_scheduler_decreases_over_time():
    sched = ExponentialDecay(0.1, 0.9)
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], scheduler=sched)
    lrs = []
    for _ in range(5):
        T = 5
        states = np.random.randn(T, 6)
        actions = np.random.randint(0, N_ACTIONS, size=T)
        rewards = np.random.randn(T) * 0.1
        agent.train_step(states, actions, rewards)
        lrs.append(agent.optimizer.lr)
    for i in range(1, len(lrs)):
        assert lrs[i] < lrs[i - 1]


def test_agent_without_scheduler_leaves_lr_unchanged():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], lr=0.01)
    lr_before = agent.optimizer.lr
    T = 5
    states = np.random.randn(T, 6)
    actions = np.random.randint(0, N_ACTIONS, size=T)
    rewards = np.random.randn(T) * 0.1
    agent.train_step(states, actions, rewards)
    assert agent.optimizer.lr == lr_before


# -- Checkpoint save/load tests ----------------------------------------------

def test_save_load_weights(tmp_path):
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], seed=42)
    for layer in agent.policy_net.layers:
        layer.w[:] = 1.0
        layer.b[:] = 2.0
    path = str(tmp_path / "test_model.npz")
    agent.save(path)
    agent2 = REINFORCEAgent(state_dim=6, hidden_sizes=[16], seed=99)
    agent2.load(path)
    for l1, l2 in zip(agent.policy_net.layers, agent2.policy_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)


def test_save_load_sgd_optimizer_state(tmp_path):
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], momentum=0.9, seed=42)
    state = np.random.randn(6)
    for _ in range(3):
        agent.act(state, training=True)
        T = 5
        states = np.array([state + np.random.randn(6) * 0.01 for _ in range(T)])
        actions = np.random.randint(0, N_ACTIONS, size=T)
        rewards = np.random.randn(T) * 0.1
        agent.train_step(states, actions, rewards)
    path = str(tmp_path / "sgd_state.npz")
    agent.save(path)
    agent2 = REINFORCEAgent(state_dim=6, hidden_sizes=[16], momentum=0.9, seed=99)
    agent2.load(path)
    for l1, l2 in zip(agent.policy_net.layers, agent2.policy_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)
    for i, layer in enumerate(agent.policy_net.layers):
        v1 = agent.optimizer._velocities[layer]
        v2 = agent2.optimizer._velocities[agent2.policy_net.layers[i]]
        np.testing.assert_array_equal(v1["w"], v2["w"])
        np.testing.assert_array_equal(v1["b"], v2["b"])


def test_save_load_adam_optimizer_state(tmp_path):
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], optimizer_type="adam", seed=42)
    T = 5
    for _ in range(3):
        states = np.random.randn(T, 6)
        actions = np.random.randint(0, N_ACTIONS, size=T)
        rewards = np.random.randn(T) * 0.1
        agent.train_step(states, actions, rewards)
    path = str(tmp_path / "adam_state.npz")
    agent.save(path)
    agent2 = REINFORCEAgent(state_dim=6, hidden_sizes=[16], optimizer_type="adam", seed=99)
    agent2.load(path)
    for l1, l2 in zip(agent.policy_net.layers, agent2.policy_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)
    for i, layer in enumerate(agent.policy_net.layers):
        m1, v1 = agent.optimizer._moments[layer]
        m2, v2 = agent2.optimizer._moments[agent2.policy_net.layers[i]]
        np.testing.assert_array_equal(m1["w"], m2["w"])
        np.testing.assert_array_equal(m1["b"], m2["b"])
        np.testing.assert_array_equal(v1["w"], v2["w"])
        np.testing.assert_array_equal(v1["b"], v2["b"])
    assert agent.optimizer._t == agent2.optimizer._t


def test_save_load_value_network(tmp_path):
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], use_value_network=True, seed=42)
    T = 5
    for _ in range(2):
        states = np.random.randn(T, 6)
        actions = np.random.randint(0, N_ACTIONS, size=T)
        rewards = np.random.randn(T) * 0.1
        agent.train_step(states, actions, rewards)
    path = str(tmp_path / "vn_state.npz")
    agent.save(path)
    agent2 = REINFORCEAgent(state_dim=6, hidden_sizes=[16], use_value_network=True, seed=99)
    agent2.load(path)
    for l1, l2 in zip(agent.policy_net.layers, agent2.policy_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)
    for l1, l2 in zip(agent.value_net.layers, agent2.value_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)


def test_save_load_rng_state(tmp_path):
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], seed=42)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        agent.act(state, training=True)
    path = str(tmp_path / "rng_state.npz")
    agent.save(path)
    agent2 = REINFORCEAgent(state_dim=6, hidden_sizes=[16], seed=42)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        agent2.act(state, training=True)
    ref_actions = [agent2.act(state, training=True) for _ in range(10)]
    agent2.load(path)
    loaded_actions = [agent2.act(state, training=True) for _ in range(10)]
    assert loaded_actions == ref_actions
    assert len(loaded_actions) == 10
    for a in loaded_actions:
        assert 0 <= a < N_ACTIONS


def test_save_metadata(tmp_path):
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[32, 32], seed=42)
    path = str(tmp_path / "meta.npz")
    agent.save(path)
    data = np.load(path)
    assert "hidden_sizes" in data
    assert "state_dim" in data
    assert "n_actions" in data
    assert list(data["hidden_sizes"]) == [32, 32]
    assert int(data["state_dim"]) == 6
    assert int(data["n_actions"]) == N_ACTIONS
    data.close()


def test_load_metadata_hidden_size_mismatch(tmp_path):
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[32, 32], seed=42)
    path = str(tmp_path / "mismatch.npz")
    agent.save(path)
    agent2 = REINFORCEAgent(state_dim=6, hidden_sizes=[64, 64], seed=99)
    with pytest.raises(ValueError, match="hidden_sizes"):
        agent2.load(path)


def test_load_metadata_state_dim_mismatch(tmp_path):
    agent = REINFORCEAgent(state_dim=8, hidden_sizes=[16], seed=42)
    path = str(tmp_path / "mismatch_dim.npz")
    agent.save(path)
    agent2 = REINFORCEAgent(state_dim=6, hidden_sizes=[16], seed=99)
    with pytest.raises(ValueError, match="state_dim"):
        agent2.load(path)


# -- Integration tests -------------------------------------------------------

def test_training_loop_short_episodes():
    np.random.seed(42)
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], lr=1e-3, seed=42)
    for ep in range(5):
        T = 10
        states = np.random.randn(T, 6)
        actions = np.random.randint(0, N_ACTIONS, size=T)
        rewards = np.random.randn(T) * 0.1
        loss = agent.train_step(states, actions, rewards)
        assert np.isfinite(loss)





# -- Error handling tests ----------------------------------------------------

def test_agent_negative_gamma():
    agent = REINFORCEAgent(state_dim=6, gamma=-0.5)
    T = 5
    states = np.random.randn(T, 6)
    actions = np.zeros(T, dtype=np.int64)
    rewards = np.ones(T)
    loss = agent.train_step(states, actions, rewards)
    assert np.isfinite(loss)


def test_agent_no_baseline_no_value_net():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[16], use_baseline=False, use_value_network=False)
    T = 5
    states = np.random.randn(T, 6)
    actions = np.zeros(T, dtype=np.int64)
    rewards = np.ones(T)
    loss = agent.train_step(states, actions, rewards)
    assert np.isfinite(loss)


def test_agent_hidden_sizes_preserved():
    agent = REINFORCEAgent(state_dim=6, hidden_sizes=[32, 64, 128])
    assert agent.hidden_sizes == [32, 64, 128]
    expected_shapes = [(6, 32), (32, 64), (64, 128), (128, N_ACTIONS)]
    for layer, (in_feat, out_feat) in zip(agent.policy_net.layers, expected_shapes):
        assert layer.w.shape == (in_feat, out_feat)

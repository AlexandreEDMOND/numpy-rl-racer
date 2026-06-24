import numpy as np

from numpy_rl_racer.agent.dqn import DQNAgent


def _make_agent(**kwargs):
    defaults = dict(state_dim=6, hidden_sizes=[16], lr=1e-3, batch_size=8, seed=42)
    defaults.update(kwargs)
    return DQNAgent(**defaults)


def test_save_load_full_state(tmp_path):
    agent = _make_agent(momentum=0.9)
    agent.epsilon = 0.5
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rng = np.random.RandomState(0)
    for _ in range(3):
        action = agent.act(state, training=True)
        next_state = state + rng.randn(6) * 0.01
        agent.train_step(state, action, 0.1, next_state, False)

    path = str(tmp_path / "full_state.npz")
    agent.save(path)

    agent2 = _make_agent(momentum=0.9)
    agent2.load(path)

    assert agent2.epsilon == agent.epsilon
    assert agent2._step_counter == agent._step_counter
    for l1, l2 in zip(agent.online_net.layers, agent2.online_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)
    for l1, l2 in zip(agent.target_net.layers, agent2.target_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)
    if agent.optimizer._velocities is not None:
        for i, layer in enumerate(agent.online_net.layers):
            v1 = agent.optimizer._velocities[layer]
            v2 = agent2.optimizer._velocities[layer]
            np.testing.assert_array_equal(v1["w"], v2["w"])
            np.testing.assert_array_equal(v1["b"], v2["b"])


def test_checkpoint_reproducibility(tmp_path):
    rng = np.random.RandomState(0)
    agent = _make_agent()
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        action = agent.act(state, training=True)
        next_state = state + rng.randn(6) * 0.01
        agent.train_step(state, action, 0.1, next_state, False)

    path = str(tmp_path / "ckpt.npz")
    agent.save(path)

    ref_actions, ref_losses = [], []
    for _ in range(5):
        action = agent.act(state, training=True)
        next_state = state + rng.randn(6) * 0.01
        loss = agent.train_step(state, action, 0.1, next_state, False)
        ref_actions.append(action)
        ref_losses.append(loss)

    rng_loaded = np.random.RandomState(0)
    for _ in range(20):
        rng_loaded.randn(6)

    agent2 = _make_agent()
    agent2.load(path)

    loaded_actions, loaded_losses = [], []
    for _ in range(5):
        action = agent2.act(state, training=True)
        next_state = state + rng_loaded.randn(6) * 0.01
        loss = agent2.train_step(state, action, 0.1, next_state, False)
        loaded_actions.append(action)
        loaded_losses.append(loss)

    assert ref_actions == loaded_actions
    for l1, l2 in zip(ref_losses, loaded_losses):
        assert np.isclose(l1, l2), f"Loss mismatch: {l1} vs {l2}"


def test_legacy_weights_compatibility(tmp_path):
    agent = _make_agent()
    params = {}
    for i, layer in enumerate(agent.online_net.layers):
        params[f"layer_{i}_w"] = layer.w
        params[f"layer_{i}_b"] = layer.b
    path = str(tmp_path / "legacy.npz")
    np.savez(path, **params)

    agent2 = _make_agent()
    agent2.load(path)

    assert agent2.epsilon == 1.0
    assert agent2._step_counter == 0


def test_save_load_replay_buffer(tmp_path):
    agent = _make_agent()
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rng = np.random.RandomState(0)
    for _ in range(10):
        action = agent.act(state, training=True)
        next_state = state + rng.randn(6) * 0.01
        agent.train_step(state, action, 0.1, next_state, False)

    path = str(tmp_path / "buf.npz")
    agent.save(path)

    agent2 = _make_agent()
    agent2.load(path)

    b1 = agent.replay_buffer
    b2 = agent2.replay_buffer
    assert len(b1) == len(b2)
    for i in range(len(b1)):
        for j in range(5):
            v1 = b1.buffer[i][j]
            v2 = b2.buffer[i][j]
            if isinstance(v1, np.ndarray):
                np.testing.assert_array_equal(v1, v2)
            else:
                assert v1 == v2
    assert b1.pos == b2.pos


def test_save_load_per_buffer(tmp_path):
    agent = _make_agent(use_per=True)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rng = np.random.RandomState(0)
    for _ in range(10):
        action = agent.act(state, training=True)
        next_state = state + rng.randn(6) * 0.01
        agent.train_step(state, action, 0.1, next_state, False)

    path = str(tmp_path / "per.npz")
    agent.save(path)

    agent2 = _make_agent(use_per=True)
    agent2.load(path)

    b1 = agent.replay_buffer
    b2 = agent2.replay_buffer
    assert len(b1) == len(b2)
    assert b1.beta == b2.beta
    assert b1._step == b2._step
    assert b1.max_priority == b2.max_priority
    assert b1.tree.pos == b2.tree.pos
    assert b1.tree.size == b2.tree.size
    np.testing.assert_array_equal(b1.tree.tree, b2.tree.tree)
    for i in range(b1.tree.capacity):
        d1 = b1.tree.data[i]
        d2 = b2.tree.data[i]
        if d1 is None:
            assert d2 is None
        else:
            for j in range(5):
                v1 = d1[j]
                v2 = d2[j]
                if isinstance(v1, np.ndarray):
                    np.testing.assert_array_equal(v1, v2)
                else:
                    assert v1 == v2


def test_dueling_checkpoint(tmp_path):
    agent = _make_agent(use_dueling_dqn=True)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rng = np.random.RandomState(0)
    for _ in range(20):
        action = agent.act(state, training=True)
        next_state = state + rng.randn(6) * 0.01
        agent.train_step(state, action, 0.1, next_state, False)

    path = str(tmp_path / "dueling.npz")
    agent.save(path)

    agent2 = _make_agent(use_dueling_dqn=True)
    agent2.load(path)

    for l1, l2 in zip(agent.online_net.layers, agent2.online_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)
    for l1, l2 in zip(agent.target_net.layers, agent2.target_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)

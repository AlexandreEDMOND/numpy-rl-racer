import warnings

import numpy as np
import pytest

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


def test_save_metadata(tmp_path):
    agent_mlp = DQNAgent(state_dim=6, hidden_sizes=[32, 32], seed=42)
    path_mlp = str(tmp_path / "mlp.npz")
    agent_mlp.save(path_mlp)
    data_mlp = np.load(path_mlp)
    assert "arch_type" in data_mlp
    assert "hidden_sizes" in data_mlp
    assert "use_dueling_dqn" in data_mlp
    assert "state_dim" in data_mlp
    assert "n_actions" in data_mlp
    assert int(data_mlp["arch_type"]) == 0
    assert list(data_mlp["hidden_sizes"]) == [32, 32]
    assert int(data_mlp["use_dueling_dqn"]) == 0
    assert int(data_mlp["state_dim"]) == 6
    data_mlp.close()

    agent_dueling = DQNAgent(state_dim=8, hidden_sizes=[64], use_dueling_dqn=True, seed=42)
    path_dueling = str(tmp_path / "dueling.npz")
    agent_dueling.save(path_dueling)
    data_dueling = np.load(path_dueling)
    assert int(data_dueling["arch_type"]) == 1
    assert list(data_dueling["hidden_sizes"]) == [64]
    assert int(data_dueling["use_dueling_dqn"]) == 1
    assert int(data_dueling["state_dim"]) == 8
    data_dueling.close()


def test_load_metadata_match(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], use_dueling_dqn=True, seed=42)
    path = str(tmp_path / "match.npz")
    agent.save(path)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], use_dueling_dqn=True, seed=99)
    agent2.load(path)

    for l1, l2 in zip(agent.online_net.layers, agent2.online_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)


def test_load_metadata_mismatch_dueling_into_mlp(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], use_dueling_dqn=True, seed=42)
    path = str(tmp_path / "dueling.npz")
    agent.save(path)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], use_dueling_dqn=False, seed=99)
    with pytest.raises(ValueError, match="dueling architecture"):
        agent2.load(path)


def test_load_metadata_mismatch_mlp_into_dueling(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], use_dueling_dqn=False, seed=42)
    path = str(tmp_path / "mlp.npz")
    agent.save(path)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], use_dueling_dqn=True, seed=99)
    with pytest.raises(ValueError, match="MLP architecture"):
        agent2.load(path)


def test_load_metadata_hidden_size_mismatch(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[32, 32], use_dueling_dqn=False, seed=42)
    path = str(tmp_path / "mlp.npz")
    agent.save(path)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[64, 64], use_dueling_dqn=False, seed=99)
    with pytest.raises(ValueError, match="hidden_sizes"):
        agent2.load(path)


def test_load_metadata_state_dim_mismatch(tmp_path):
    agent = DQNAgent(state_dim=8, hidden_sizes=[16], use_dueling_dqn=False, seed=42)
    path = str(tmp_path / "mlp_8d.npz")
    agent.save(path)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], use_dueling_dqn=False, seed=99)
    with pytest.raises(ValueError, match="state_dim"):
        agent2.load(path)


def test_legacy_npz_no_metadata(tmp_path):
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], seed=42)
    params = {}
    for i, layer in enumerate(agent.online_net.layers):
        params[f"layer_{i}_w"] = layer.w
        params[f"layer_{i}_b"] = layer.b
    for i, layer in enumerate(agent.target_net.layers):
        params[f"tlayer_{i}_w"] = layer.w
        params[f"tlayer_{i}_b"] = layer.b
    path = str(tmp_path / "legacy_no_meta.npz")
    np.savez(path, **params)

    agent2 = DQNAgent(state_dim=6, hidden_sizes=[16], seed=99)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agent2.load(path)
        assert len(w) == 1
        assert "does not contain architecture metadata" in str(w[0].message).lower()

    for l1, l2 in zip(agent.online_net.layers, agent2.online_net.layers):
        np.testing.assert_array_equal(l1.w, l2.w)
        np.testing.assert_array_equal(l1.b, l2.b)

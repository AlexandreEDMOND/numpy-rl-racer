import numpy as np

from numpy_rl_racer.network import Dense, DuelingMLP, MLP, SGD, relu
from numpy_rl_racer.utils.scheduler import ExponentialDecay, LRScheduler, StepDecay


def test_relu_positive():
    x = np.array([0.0, 1.0, 5.0])
    np.testing.assert_array_equal(relu(x), x)


def test_relu_negative():
    x = np.array([-3.0, -0.5, -1.0])
    np.testing.assert_array_equal(relu(x), np.zeros_like(x))


def test_relu_mixed():
    x = np.array([-1.0, 0.0, 2.0])
    np.testing.assert_array_equal(relu(x), np.array([0.0, 0.0, 2.0]))


def test_dense_forward_shape():
    layer = Dense(4, 8)
    x = np.random.randn(3, 4)
    out = layer.forward(x)
    assert out.shape == (3, 8)
    assert out.dtype == np.float64


def test_dense_output_value():
    layer = Dense(2, 2)
    layer.w = np.array([[1.0, 0.0], [0.0, 1.0]])
    layer.b = np.array([0.0, 0.0])
    x = np.array([[2.0, 3.0]])
    out = layer.forward(x)
    np.testing.assert_array_equal(out, np.array([[2.0, 3.0]]))


def test_dense_bias():
    layer = Dense(2, 2)
    layer.w = np.zeros((2, 2))
    layer.b = np.array([5.0, -3.0])
    x = np.random.randn(4, 2)
    out = layer.forward(x)
    np.testing.assert_array_equal(out, np.full((4, 2), [5.0, -3.0]))


def test_mlp_forward_shape():
    mlp = MLP([4, 8, 2])
    x = np.random.randn(5, 4)
    out = mlp.forward(x)
    assert out.shape == (5, 2)


def test_mlp_without_hidden():
    mlp = MLP([4, 2])
    x = np.random.randn(3, 4)
    out = mlp.forward(x)
    assert out.shape == (3, 2)


def test_mlp_output_activation_sigmoid():
    mlp = MLP([2, 4, 1], output_activation="sigmoid")
    for layer in mlp.layers:
        layer.w[:] = 0.0
        layer.b[:] = 0.0
    x = np.array([[100.0, -100.0]])
    out = mlp.forward(x)
    assert np.all(out > 0.0) and np.all(out < 1.0)


def test_mlp_output_activation_tanh():
    mlp = MLP([2, 4, 2], output_activation="tanh")
    x = np.random.randn(100, 2)
    out = mlp.forward(x)
    assert np.all(np.abs(out) <= 1.0 + 1e-10)


def test_mlp_output_activation_none():
    mlp = MLP([2, 4, 2])
    x = np.random.randn(3, 2)
    out = mlp.forward(x)
    assert out.shape == (3, 2)


def test_mlp_parameters():
    mlp = MLP([4, 8, 3])
    params = mlp.parameters()
    assert len(params) == 2
    w1, b1 = params[0]
    assert w1.shape == (4, 8)
    assert b1.shape == (8,)
    w2, b2 = params[1]
    assert w2.shape == (8, 3)
    assert b2.shape == (3,)


def test_mlp_parameters_deep():
    mlp = MLP([2, 4, 6, 3])
    params = mlp.parameters()
    assert len(params) == 3
    shapes = [(2, 4), (4, 6), (6, 3)]
    for (w, b), (wi, wo) in zip(params, shapes):
        assert w.shape == (wi, wo)
        assert b.shape == (wo,)


def test_mlp_batch():
    mlp = MLP([4, 16, 2])
    x = np.random.randn(32, 4)
    out = mlp.forward(x)
    assert out.shape == (32, 2)


def test_mlp_single_input():
    mlp = MLP([4, 8, 2])
    x = np.random.randn(4)
    out = mlp.forward(x)
    assert out.shape == (2,)


def test_mlp_deep_network():
    mlp = MLP([3, 16, 16, 16, 2])
    x = np.random.randn(10, 3)
    out = mlp.forward(x)
    assert out.shape == (10, 2)


def test_mlp_parameters_mutable():
    mlp = MLP([2, 4, 1])
    params = mlp.parameters()
    old_w = params[0][0].copy()
    params[0][0][:] = 0.0
    np.testing.assert_array_equal(mlp.layers[0].w, 0.0)
    mlp.layers[0].w[:] = old_w


def test_dueling_mlp_forward_output():
    n_actions = 5
    net = DuelingMLP(state_dim=4, hidden_sizes=[8], n_actions=n_actions)
    x = np.random.randn(3, 4)
    out = net.forward(x)
    encoded = relu(net._cached_pre_relu[-1])
    v = net.value_layer.forward(encoded)
    a = net.advantage_layer.forward(encoded)
    expected = v + a - np.mean(a, axis=1, keepdims=True)
    np.testing.assert_allclose(out, expected)


def test_dueling_mlp_backward():
    n_actions = 5
    net = DuelingMLP(state_dim=4, hidden_sizes=[8, 16], n_actions=n_actions)
    x = np.random.randn(3, 4)
    net.forward(x)
    grad = np.random.randn(3, n_actions)
    net.backward(grad)
    for layer in net.layers:
        assert layer.grad_w is not None, f"grad_w is None for {layer}"
        assert layer.grad_b is not None, f"grad_b is None for {layer}"
        assert not np.allclose(layer.grad_w, 0), f"grad_w is all zeros for {layer}"
        assert not np.allclose(layer.grad_b, 0), f"grad_b is all zeros for {layer}"


class TestExponentialDecay:
    def test_step_multiplies_lr_by_decay_rate(self):
        sched = ExponentialDecay(1.0, 0.5)
        sched.step()
        assert sched.lr == 0.5
        sched.step()
        assert sched.lr == 0.25

    def test_multiple_steps(self):
        sched = ExponentialDecay(1.0, 0.9)
        for _ in range(10):
            sched.step()
        np.testing.assert_almost_equal(sched.lr, 0.9 ** 10)


class TestStepDecay:
    def test_drops_lr_at_correct_intervals(self):
        sched = StepDecay(1.0, 0.5, 3)
        for _ in range(2):
            sched.step()
        assert sched.lr == 1.0
        sched.step()
        assert sched.lr == 0.5
        for _ in range(2):
            sched.step()
        assert sched.lr == 0.5
        sched.step()
        assert sched.lr == 0.25

    def test_no_drop_before_first_interval(self):
        sched = StepDecay(1.0, 0.1, 5)
        for _ in range(4):
            sched.step()
        assert sched.lr == 1.0


class TestSGDWithScheduler:
    def test_sgd_with_scheduler_produces_decreasing_lr(self):
        mlp = MLP([2, 4, 1])
        x = np.random.randn(2, 2)
        mlp.forward(x)
        mlp.backward(np.random.randn(2, 1))
        sched = ExponentialDecay(0.1, 0.5)
        opt = SGD(mlp, scheduler=sched)
        opt.step()
        first_lr = opt.lr
        opt.step()
        second_lr = opt.lr
        assert second_lr < first_lr

    def test_sgd_without_scheduler_leaves_lr_unchanged(self):
        mlp = MLP([2, 4, 1])
        x = np.random.randn(2, 2)
        mlp.forward(x)
        mlp.backward(np.random.randn(2, 1))
        opt = SGD(mlp, lr=0.01)
        lr_before = opt.lr
        opt.step()
        assert opt.lr == lr_before

    def test_sgd_scheduler_lr_takes_precedence(self):
        sched = ExponentialDecay(0.5, 0.9)
        mlp = MLP([2, 4, 1])
        opt = SGD(mlp, lr=0.01, scheduler=sched)
        assert opt.lr == 0.5


class TestSGDWithMomentum:
    def test_sgd_momentum_zero_equals_plain(self):
        mlp = MLP([2, 4, 1])
        rng = np.random.RandomState(42)
        opt = SGD(mlp, lr=0.01, momentum=0.0)
        for _ in range(5):
            x = rng.randn(3, 2)
            mlp.forward(x)
            mlp.backward(rng.randn(3, 1))
            saved = [(layer.w.copy(), layer.b.copy(), layer.grad_w.copy(), layer.grad_b.copy())
                     for layer in mlp.layers]
            opt.step()
            for i, (w0, b0, gw, gb) in enumerate(saved):
                np.testing.assert_array_equal(mlp.layers[i].w, w0 - 0.01 * gw)
                np.testing.assert_array_equal(mlp.layers[i].b, b0 - 0.01 * gb)

    def test_sgd_momentum_velocity_accumulates(self):
        mlp = MLP([2, 4, 1])
        rng = np.random.RandomState(42)
        x = rng.randn(3, 2)
        lr = 0.1
        momentum = 0.9
        mlp.forward(x)
        mlp.backward(rng.randn(3, 1))
        grad_w = [layer.grad_w.copy() for layer in mlp.layers]
        opt = SGD(mlp, lr=lr, momentum=momentum)
        opt.step()
        for layer, gw in zip(mlp.layers, grad_w):
            np.testing.assert_array_equal(opt._velocities[layer]["w"], -lr * gw)
        v1_w = [opt._velocities[layer]["w"].copy() for layer in mlp.layers]
        opt.step()
        for i, layer in enumerate(mlp.layers):
            expected = momentum * v1_w[i] - lr * grad_w[i]
            np.testing.assert_array_equal(opt._velocities[layer]["w"], expected)

    def test_sgd_momentum_separate_per_param(self):
        mlp = MLP([2, 4, 3])
        rng = np.random.RandomState(42)
        x = rng.randn(3, 2)
        mlp.forward(x)
        mlp.backward(rng.randn(3, 3))
        opt = SGD(mlp, lr=0.01, momentum=0.9)
        opt.step()
        assert len(opt._velocities) == 2
        v0_w = opt._velocities[mlp.layers[0]]["w"]
        v1_w = opt._velocities[mlp.layers[1]]["w"]
        assert v0_w.shape == (2, 4)
        assert v1_w.shape == (4, 3)
        v0_w[:] = 999.0
        assert not np.any(v1_w == 999.0)

    def test_dqn_agent_momentum_passthrough(self):
        from numpy_rl_racer.agent import DQNAgent
        agent = DQNAgent(state_dim=4, hidden_sizes=[8], lr=0.01, momentum=0.9)
        assert agent.optimizer.momentum == 0.9


class TestLRSchedulerBase:
    def test_base_class_raises_not_implemented(self):
        sched = LRScheduler(1.0)
        try:
            sched.step()
            assert False, "Expected NotImplementedError"
        except NotImplementedError:
            pass


class TestDenseWeightDecay:
    def test_dense_weight_decay_zero_equals_none(self):
        rng = np.random.RandomState(42)
        x = rng.randn(3, 4)
        grad_output = rng.randn(3, 2)

        layer_no_wd = Dense(4, 2)
        layer_no_wd.w = rng.randn(4, 2)
        layer_no_wd.b = np.zeros(2)
        layer_no_wd.forward(x)
        layer_no_wd.backward(grad_output)

        layer_wd0 = Dense(4, 2, weight_decay=0.0)
        layer_wd0.w = layer_no_wd.w.copy()
        layer_wd0.b = layer_no_wd.b.copy()
        layer_wd0.forward(x)
        layer_wd0.backward(grad_output)

        np.testing.assert_array_equal(layer_wd0.grad_w, layer_no_wd.grad_w)
        np.testing.assert_array_equal(layer_wd0.grad_b, layer_no_wd.grad_b)

    def test_dense_weight_decay_adds_penalty(self):
        rng = np.random.RandomState(42)
        x = rng.randn(3, 4)
        grad_output = rng.randn(3, 2)
        wd = 1e-4

        layer = Dense(4, 2, weight_decay=wd)
        layer.w = rng.randn(4, 2)
        layer.b = np.zeros(2)
        layer.forward(x)
        layer.backward(grad_output)

        grad_w_expected = x.T @ grad_output + wd * layer.w
        np.testing.assert_allclose(layer.grad_w, grad_w_expected)

    def test_dense_weight_decay_bias_not_affected(self):
        rng = np.random.RandomState(42)
        x = rng.randn(3, 4)
        grad_output = rng.randn(3, 2)

        layer_no_wd = Dense(4, 2)
        layer_no_wd.w = rng.randn(4, 2)
        layer_no_wd.b = np.zeros(2)
        layer_no_wd.forward(x)
        layer_no_wd.backward(grad_output)

        layer_wd = Dense(4, 2, weight_decay=0.1)
        layer_wd.w = layer_no_wd.w.copy()
        layer_wd.b = layer_no_wd.b.copy()
        layer_wd.forward(x)
        layer_wd.backward(grad_output)

        np.testing.assert_array_equal(layer_wd.grad_b, layer_no_wd.grad_b)

    def test_dqn_weight_decay_passthrough(self):
        from numpy_rl_racer.agent import DQNAgent
        agent = DQNAgent(state_dim=4, hidden_sizes=[8], lr=0.01, weight_decay=0.01)
        for layer in agent.online_net.layers:
            assert layer.weight_decay == 0.01
        for layer in agent.target_net.layers:
            assert layer.weight_decay == 0.01

    def test_dqn_weight_decay_default_zero(self):
        from numpy_rl_racer.agent import DQNAgent
        agent = DQNAgent(state_dim=4, hidden_sizes=[8], lr=0.01)
        for layer in agent.online_net.layers:
            assert layer.weight_decay == 0.0
        for layer in agent.target_net.layers:
            assert layer.weight_decay == 0.0

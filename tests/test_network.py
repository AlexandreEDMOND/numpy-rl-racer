import numpy as np

from numpy_rl_racer.network import Dense, DuelingMLP, MLP, NoisyLinear, SGD, Adam, mse_loss, relu
from numpy_rl_racer.utils.scheduler import ExponentialDecay, LinearWarmup, LRScheduler, StepDecay


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


def test_noisy_linear_forward():
    layer = NoisyLinear(4, 8)
    x = np.random.randn(3, 4)
    out = layer.forward(x)
    assert out.shape == (3, 8)
    assert out.dtype == np.float64


def test_noisy_linear_stochastic():
    layer = NoisyLinear(4, 8)
    x = np.random.randn(3, 4)
    out1 = layer.forward(x)
    out2 = layer.forward(x)
    assert not np.allclose(out1, out2)


def test_noisy_linear_reset():
    layer = NoisyLinear(4, 8)
    x = np.random.randn(3, 4)
    out1 = layer.forward(x)
    layer.reset_noise()
    out2 = layer.forward(x)
    assert not np.allclose(out1, out2)


def test_noisy_linear_seed():
    rng1 = np.random.RandomState(42)
    layer1 = NoisyLinear(4, 8, rng=rng1)
    rng2 = np.random.RandomState(42)
    layer2 = NoisyLinear(4, 8, rng=rng2)
    layer1.w[:] = 1.0
    layer1.b[:] = 0.0
    layer2.w[:] = 1.0
    layer2.b[:] = 0.0
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    out1 = layer1.forward(x)
    out2 = layer2.forward(x)
    np.testing.assert_array_equal(out1, out2)


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


class TestLinearWarmup:
    def test_lr_at_step_zero_equals_initial_lr(self):
        sched = LinearWarmup(initial_lr=0.0, final_lr=0.1, warmup_steps=10)
        assert sched.lr == 0.0

    def test_lr_at_midpoint(self):
        sched = LinearWarmup(initial_lr=0.0, final_lr=1.0, warmup_steps=10)
        for _ in range(5):
            sched.step()
        assert abs(sched.lr - 0.5) < 1e-10

    def test_lr_at_warmup_steps_equals_final_lr(self):
        sched = LinearWarmup(initial_lr=0.0, final_lr=1.0, warmup_steps=10)
        for _ in range(10):
            sched.step()
        assert sched.lr == 1.0

    def test_lr_plateaus_after_warmup(self):
        sched = LinearWarmup(initial_lr=0.0, final_lr=1.0, warmup_steps=10)
        for _ in range(11):
            sched.step()
        assert sched.lr == 1.0

    def test_post_scheduler_decays_after_warmup(self):
        post = ExponentialDecay(1.0, 0.5)
        sched = LinearWarmup(initial_lr=0.0, final_lr=1.0, warmup_steps=5, post_scheduler=post)
        for _ in range(5):
            sched.step()
        assert sched.lr == 1.0
        sched.step()
        assert sched.lr == 0.5
        sched.step()
        assert sched.lr == 0.25

    def test_works_with_sgd_optimizer(self):
        mlp = MLP([2, 4, 1])
        x = np.random.randn(2, 2)
        mlp.forward(x)
        mlp.backward(np.random.randn(2, 1))
        sched = LinearWarmup(initial_lr=0.0, final_lr=0.1, warmup_steps=5)
        opt = SGD(mlp, scheduler=sched)
        assert opt.lr == 0.0
        opt.step()
        assert abs(opt.lr - 0.02) < 1e-10

    def test_warmup_steps_zero(self):
        sched = LinearWarmup(initial_lr=0.0, final_lr=0.1, warmup_steps=0)
        assert sched.lr == 0.0
        sched.step()
        assert sched.lr == 0.1

    def test_initial_lr_equals_final_lr(self):
        sched = LinearWarmup(initial_lr=0.5, final_lr=0.5, warmup_steps=10)
        for _ in range(5):
            sched.step()
        assert sched.lr == 0.5

    def test_post_scheduler_none_plateaus(self):
        sched = LinearWarmup(initial_lr=0.0, final_lr=0.1, warmup_steps=5)
        for _ in range(10):
            sched.step()
        assert sched.lr == 0.1

    def test_nonzero_initial_lr_midpoint(self):
        sched = LinearWarmup(initial_lr=0.2, final_lr=1.0, warmup_steps=10)
        for _ in range(5):
            sched.step()
        assert abs(sched.lr - 0.6) < 1e-10

    def test_sgd_loss_decreases_on_toy_task(self):
        mlp = MLP([1, 4, 1])
        rng = np.random.RandomState(42)
        X = rng.randn(16, 1)
        y = 2.0 * X + 0.5
        sched = LinearWarmup(initial_lr=0.0, final_lr=0.001, warmup_steps=10)
        opt = SGD(mlp, scheduler=sched)
        losses = []
        for _ in range(100):
            pred = mlp.forward(X)
            loss = mse_loss(pred, y)
            losses.append(loss)
            mlp.backward(pred - y)
            opt.step()
        assert losses[-1] < losses[0]


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


class TestSGDGradientClipping:
    def test_clipping_reduces_large_gradients(self):
        mlp = MLP([2, 4, 1])
        x = np.random.randn(3, 2)
        mlp.forward(x)
        mlp.backward(np.random.randn(3, 1))
        for layer in mlp.layers:
            layer.grad_w[:] = 100.0
            layer.grad_b[:] = 100.0

        opt = SGD(mlp, lr=0.01, max_grad_norm=1.0)
        old_w = [layer.w.copy() for layer in mlp.layers]
        old_b = [layer.b.copy() for layer in mlp.layers]
        opt.step()

        effective_norm_sq = 0.0
        for i, layer in enumerate(mlp.layers):
            delta_w = (old_w[i] - layer.w) / 0.01
            delta_b = (old_b[i] - layer.b) / 0.01
            effective_norm_sq += np.sum(delta_w ** 2) + np.sum(delta_b ** 2)
        effective_norm = np.sqrt(effective_norm_sq)

        expected_norm = 1.0
        assert np.isclose(effective_norm, expected_norm, rtol=1e-6), (
            f"Effective gradient norm {effective_norm} != {expected_norm}"
        )

    def test_no_clipping_when_norm_below_threshold(self):
        mlp = MLP([2, 4, 1])
        x = np.random.randn(3, 2)
        mlp.forward(x)
        mlp.backward(np.random.randn(3, 1))
        for layer in mlp.layers:
            layer.grad_w[:] = 0.5
            layer.grad_b[:] = 0.5

        actual_norm = np.sqrt(sum(np.sum(layer.grad_w ** 2) + np.sum(layer.grad_b ** 2)
                                  for layer in mlp.layers))

        opt = SGD(mlp, lr=0.01, max_grad_norm=1000.0)
        old_w = [layer.w.copy() for layer in mlp.layers]
        old_b = [layer.b.copy() for layer in mlp.layers]
        opt.step()

        effective_norm_sq = 0.0
        for i, layer in enumerate(mlp.layers):
            delta_w = (old_w[i] - layer.w) / 0.01
            delta_b = (old_b[i] - layer.b) / 0.01
            effective_norm_sq += np.sum(delta_w ** 2) + np.sum(delta_b ** 2)
        effective_norm = np.sqrt(effective_norm_sq)

        assert np.isclose(effective_norm, actual_norm, rtol=1e-6), (
            f"Effective gradient norm {effective_norm} changed when it should not"
        )

    def test_max_grad_none_preserves_gradients(self):
        rng = np.random.RandomState(42)
        mlp1 = MLP([2, 4, 1])
        mlp2 = MLP([2, 4, 1])
        mlp2.layers[0].w = mlp1.layers[0].w.copy()
        mlp2.layers[0].b = mlp1.layers[0].b.copy()
        mlp2.layers[1].w = mlp1.layers[1].w.copy()
        mlp2.layers[1].b = mlp1.layers[1].b.copy()

        x = rng.randn(3, 2)
        grad = rng.randn(3, 1)
        mlp1.forward(x)
        mlp1.backward(grad)
        mlp2.forward(x)
        mlp2.backward(grad)

        opt_none = SGD(mlp1, lr=0.01, max_grad_norm=None)
        opt_default = SGD(mlp2, lr=0.01)

        opt_none.step()
        opt_default.step()

        for l1, l2 in zip(mlp1.layers, mlp2.layers):
            np.testing.assert_array_equal(l1.w, l2.w)
            np.testing.assert_array_equal(l1.b, l2.b)

    def test_clipping_with_momentum(self):
        mlp = MLP([2, 4, 1])
        x = np.random.randn(3, 2)
        mlp.forward(x)
        mlp.backward(np.random.randn(3, 1))
        for layer in mlp.layers:
            layer.grad_w[:] = 100.0
            layer.grad_b[:] = 100.0

        actual_norm = np.sqrt(sum(np.sum(layer.grad_w ** 2) + np.sum(layer.grad_b ** 2)
                                  for layer in mlp.layers))
        lr = 0.01
        momentum = 0.9

        opt = SGD(mlp, lr=lr, momentum=momentum, max_grad_norm=1.0)
        opt.step()

        assert opt._velocities is not None
        scale = 1.0 / actual_norm
        for layer in mlp.layers:
            vel = opt._velocities[layer]
            expected_vw = -lr * scale * np.full_like(layer.w, 100.0)
            expected_vb = -lr * scale * np.full_like(layer.b, 100.0)
            np.testing.assert_array_almost_equal(vel["w"], expected_vw)
            np.testing.assert_array_almost_equal(vel["b"], expected_vb)

    def test_dqn_agent_passes_max_grad_norm(self):
        from numpy_rl_racer.agent import DQNAgent
        agent = DQNAgent(state_dim=4, hidden_sizes=[8], lr=0.01, max_grad_norm=5.0)
        assert agent.optimizer.max_grad_norm == 5.0

    def test_dqn_agent_default_max_grad_norm(self):
        from numpy_rl_racer.agent import DQNAgent
        agent = DQNAgent(state_dim=4, hidden_sizes=[8], lr=0.01)
        assert agent.optimizer.max_grad_norm is None


class TestAdamLinearRegression:
    def test_adam_reduces_loss_on_linear_regression(self):
        rng = np.random.RandomState(42)
        mlp = MLP([4, 2])
        true_w = rng.randn(4, 2)
        true_b = rng.randn(2)
        x = rng.randn(100, 4)
        target = x @ true_w + true_b
        opt = Adam(mlp, lr=0.1, betas=(0.9, 0.999))
        loss_before = mse_loss(mlp.forward(x), target)
        for _ in range(200):
            pred = mlp.forward(x)
            grad = 2.0 * (pred - target) / x.shape[0]
            mlp.backward(grad)
            opt.step()
        loss_after = mse_loss(mlp.forward(x), target)
        assert loss_after < loss_before * 0.5, f"Loss did not decrease: {loss_before:.6f} -> {loss_after:.6f}"

    def test_adam_converges_to_correct_weights(self):
        rng = np.random.RandomState(42)
        mlp = MLP([2, 1])
        layer = mlp.layers[0]
        layer.w[:] = 0.0
        layer.b[:] = 0.0
        true_w = np.array([[2.0], [-1.0]])
        true_b = np.array([0.5])
        x = rng.randn(200, 2)
        target = x @ true_w + true_b
        opt = Adam(mlp, lr=0.1, betas=(0.9, 0.999))
        for _ in range(500):
            pred = mlp.forward(x)
            grad = 2.0 * (pred - target) / x.shape[0]
            mlp.backward(grad)
            opt.step()
        np.testing.assert_allclose(layer.w, true_w, atol=0.5)
        np.testing.assert_allclose(layer.b, true_b, atol=0.5)


class TestAdamWithMLP:
    def test_adam_mlp_forward_backward_update(self):
        rng = np.random.RandomState(42)
        mlp = MLP([4, 8, 2])
        x = rng.randn(16, 4)
        target = rng.randn(16, 2)
        opt = Adam(mlp, lr=0.01, betas=(0.9, 0.999))
        for _ in range(50):
            pred = mlp.forward(x)
            loss_grad = 2.0 * (pred - target) / x.shape[0]
            mlp.backward(loss_grad)
            opt.step()
        loss = mse_loss(mlp.forward(x), target)
        assert loss < 10.0


class TestAdamToyQuadratic:
    def test_adam_solves_toy_quadratic(self):
        rng = np.random.RandomState(42)
        mlp = MLP([3, 1])
        layer = mlp.layers[0]
        layer.w[:] = rng.randn(3, 1) * 5.0
        layer.b[:] = np.array([5.0])
        A = rng.randn(3, 3)
        A = A.T @ A
        b_vec = rng.randn(3)
        c = 1.0

        def f_and_grad(wb):
            w = wb[:3].reshape(3, 1)
            b_val = wb[3:]
            val = (w.T @ A @ w + b_vec @ w + c + 0.5 * b_val ** 2).item()
            gw = 2.0 * A @ w + b_vec.reshape(3, 1)
            gb = b_val
            return val, gw, gb

        w_init = layer.w.copy().ravel()
        b_init = np.array([layer.b.item()])
        loss_before, _, _ = f_and_grad(np.concatenate([w_init, b_init]))

        opt = Adam(mlp, lr=0.1, betas=(0.9, 0.999))
        for _ in range(1000):
            val, gw, gb = f_and_grad(np.concatenate([layer.w.ravel(), [layer.b.item()]]))
            layer.grad_w = gw.reshape(3, 1)
            layer.grad_b = gb.ravel()
            opt.step()

        w_final = layer.w.copy().ravel()
        b_final = np.array([layer.b.item()])
        loss_after, _, _ = f_and_grad(np.concatenate([w_final, b_final]))
        assert loss_after < loss_before * 0.1


class TestAdamWeightDecay:
    def test_weight_decay_penalizes_large_weights(self):
        rng = np.random.RandomState(42)
        x = rng.randn(50, 4)
        target = rng.randn(50, 2)

        mlp_wd = MLP([4, 2])
        mlp_no = MLP([4, 2])
        mlp_no.layers[0].w = mlp_wd.layers[0].w.copy()
        mlp_no.layers[0].b = mlp_wd.layers[0].b.copy()

        opt_wd = Adam(mlp_wd, lr=0.01, weight_decay=0.1, betas=(0.9, 0.999))
        opt_no = Adam(mlp_no, lr=0.01, weight_decay=0.0, betas=(0.9, 0.999))
        for _ in range(100):
            for mlp in (mlp_wd, mlp_no):
                pred = mlp.forward(x)
                grad = 2.0 * (pred - target) / x.shape[0]
                mlp.backward(grad)
            opt_wd.step()
            opt_no.step()

        norm_wd = np.sum(mlp_wd.layers[0].w ** 2)
        norm_no = np.sum(mlp_no.layers[0].w ** 2)
        assert norm_wd < norm_no


class TestAdamBiasCorrection:
    def test_bias_correction_first_step(self):
        rng = np.random.RandomState(42)
        mlp = MLP([2, 1])
        layer = mlp.layers[0]
        layer.w[:] = 0.0
        layer.b[:] = 0.0
        x = rng.randn(10, 2)
        target = x @ np.array([[1.0], [2.0]]) + 0.5
        beta1, beta2 = 0.9, 0.999
        opt = Adam(mlp, lr=0.1, betas=(beta1, beta2))
        pred = mlp.forward(x)
        grad = 2.0 * (pred - target) / x.shape[0]
        mlp.backward(grad)
        gw = layer.grad_w.copy()
        gb = layer.grad_b.copy()
        opt.step()
        m, v = opt._moments[layer]
        t = opt._t
        assert t == 1
        m_hat_w = m["w"] / (1 - beta1 ** t)
        m_hat_b = m["b"] / (1 - beta1 ** t)
        v_hat_w = v["w"] / (1 - beta2 ** t)
        v_hat_b = v["b"] / (1 - beta2 ** t)
        np.testing.assert_allclose(m_hat_w, gw, rtol=1e-6)
        np.testing.assert_allclose(m_hat_b, gb, rtol=1e-6)
        np.testing.assert_allclose(v_hat_w, gw ** 2, rtol=1e-6)
        np.testing.assert_allclose(v_hat_b, gb ** 2, rtol=1e-6)

    def test_bias_correction_second_step(self):
        rng = np.random.RandomState(42)
        mlp = MLP([2, 1])
        layer = mlp.layers[0]
        x = rng.randn(10, 2)
        target = x @ np.array([[1.0], [2.0]]) + 0.5
        beta1, beta2 = 0.9, 0.999
        opt = Adam(mlp, lr=0.1, betas=(beta1, beta2))
        pred = mlp.forward(x)
        grad = 2.0 * (pred - target) / x.shape[0]
        mlp.backward(grad)
        gw1 = layer.grad_w.copy()
        gb1 = layer.grad_b.copy()
        opt.step()
        pred = mlp.forward(x)
        mlp.backward(2.0 * (pred - target) / x.shape[0])
        gw2 = layer.grad_w.copy()
        gb2 = layer.grad_b.copy()
        opt.step()
        m, v = opt._moments[layer]
        t = opt._t
        assert t == 2
        m_hat_w = m["w"] / (1 - beta1 ** t)
        m_hat_b = m["b"] / (1 - beta1 ** t)
        v_hat_w = v["w"] / (1 - beta2 ** t)
        v_hat_b = v["b"] / (1 - beta2 ** t)
        expected_mw_hat = (beta1 * gw1 + gw2) / (1 + beta1)
        expected_mb_hat = (beta1 * gb1 + gb2) / (1 + beta1)
        expected_vw_hat = (beta2 * gw1 ** 2 + gw2 ** 2) / (1 + beta2)
        expected_vb_hat = (beta2 * gb1 ** 2 + gb2 ** 2) / (1 + beta2)
        np.testing.assert_allclose(m_hat_w, expected_mw_hat, rtol=1e-6)
        np.testing.assert_allclose(m_hat_b, expected_mb_hat, rtol=1e-6)
        np.testing.assert_allclose(v_hat_w, expected_vw_hat, rtol=1e-6)
        np.testing.assert_allclose(v_hat_b, expected_vb_hat, rtol=1e-6)


class TestAdamLRScheduler:
    def test_adam_with_scheduler_produces_decreasing_lr(self):
        mlp = MLP([2, 4, 1])
        x = np.random.randn(2, 2)
        mlp.forward(x)
        mlp.backward(np.random.randn(2, 1))
        sched = ExponentialDecay(0.1, 0.5)
        opt = Adam(mlp, scheduler=sched)
        opt.step()
        first_lr = opt.lr
        opt.step()
        second_lr = opt.lr
        assert second_lr < first_lr

    def test_adam_without_scheduler_leaves_lr_unchanged(self):
        mlp = MLP([2, 4, 1])
        x = np.random.randn(2, 2)
        mlp.forward(x)
        mlp.backward(np.random.randn(2, 1))
        opt = Adam(mlp, lr=0.01)
        lr_before = opt.lr
        opt.step()
        assert opt.lr == lr_before

    def test_adam_scheduler_lr_takes_precedence(self):
        sched = ExponentialDecay(0.5, 0.9)
        mlp = MLP([2, 4, 1])
        opt = Adam(mlp, lr=0.01, scheduler=sched)
        assert opt.lr == 0.5


class _SingleLayerNet:
    def __init__(self, layer):
        self.layers = [layer]


class TestAdamWithNoisyLinear:
    def test_adam_noisy_linear_forward_backward_update(self):
        layer = NoisyLinear(4, 3, rng=np.random.RandomState(42))
        net = _SingleLayerNet(layer)
        x = np.random.randn(8, 4)
        target = np.random.randn(8, 3)
        opt = Adam(net, lr=0.01)
        for _ in range(20):
            pred = layer.forward(x)
            grad = 2.0 * (pred - target) / x.shape[0]
            layer.backward(grad)
            opt.step()
        assert layer.sigma_w is not None

    def test_adam_noisy_linear_reset_sigma_still_functions(self):
        layer = NoisyLinear(4, 3, rng=np.random.RandomState(42))
        net = _SingleLayerNet(layer)
        x = np.random.randn(5, 4)
        opt = Adam(net, lr=0.01)
        layer.forward(x)
        layer.backward(np.random.randn(5, 3))
        opt.step()
        old_sigma_w = layer.sigma_w.copy()
        old_sigma_b = layer.sigma_b.copy()
        layer.reset_noise()
        assert np.allclose(layer.sigma_w, old_sigma_w)
        assert np.allclose(layer.sigma_b, old_sigma_b)


class TestAdamWithDuelingMLP:
    def test_adam_dueling_mlp_forward_backward_update(self):
        net = DuelingMLP(state_dim=4, hidden_sizes=[8], n_actions=3)
        x = np.random.randn(6, 4)
        target = np.random.randn(6, 3)
        opt = Adam(net, lr=0.01)
        for _ in range(20):
            pred = net.forward(x)
            grad = 2.0 * (pred - target) / x.shape[0]
            net.backward(grad)
            opt.step()
        for layer in net.layers:
            assert layer.grad_w is not None


class TestAdamGradientClipping:
    def test_clipping_reduces_large_gradients(self):
        mlp = MLP([2, 4, 1])
        x = np.random.randn(3, 2)
        mlp.forward(x)
        mlp.backward(np.random.randn(3, 1))
        for layer in mlp.layers:
            layer.grad_w[:] = 100.0
            layer.grad_b[:] = 100.0

        original_lr = 0.01
        opt = Adam(mlp, lr=original_lr, max_grad_norm=1.0)
        old_w = [layer.w.copy() for layer in mlp.layers]
        old_b = [layer.b.copy() for layer in mlp.layers]
        opt.step()

        effective_norm_sq = 0.0
        for i, layer in enumerate(mlp.layers):
            effective_norm_sq += np.sum((old_w[i] - layer.w) ** 2) + np.sum((old_b[i] - layer.b) ** 2)
        assert np.sqrt(effective_norm_sq) > 0, "Update should not be zero"
        for layer in mlp.layers:
            layer.grad_w[:] = 100.0
            layer.grad_b[:] = 100.0

        opt2 = Adam(mlp, lr=original_lr, max_grad_norm=0.1)
        old_w2 = [layer.w.copy() for layer in mlp.layers]
        opt2.step()
        change_norm_sq = 0.0
        for i, layer in enumerate(mlp.layers):
            change_norm_sq += np.sum((old_w2[i] - layer.w) ** 2)
        change_norm = np.sqrt(change_norm_sq)

        for layer in mlp.layers:
            layer.grad_w[:] = 100.0
            layer.grad_b[:] = 100.0
        opt3 = Adam(mlp, lr=original_lr, max_grad_norm=0.01)
        old_w3 = [layer.w.copy() for layer in mlp.layers]
        opt3.step()
        change_norm2_sq = 0.0
        for i, layer in enumerate(mlp.layers):
            change_norm2_sq += np.sum((old_w3[i] - layer.w) ** 2)
        change_norm2 = np.sqrt(change_norm2_sq)
        assert change_norm2 < change_norm, "Tighter clipping should reduce update magnitude"

    def test_no_clipping_when_norm_below_threshold(self):
        mlp = MLP([2, 4, 1])
        x = np.random.randn(3, 2)
        mlp.forward(x)
        mlp.backward(np.random.randn(3, 1))
        for layer in mlp.layers:
            layer.grad_w[:] = 0.5
            layer.grad_b[:] = 0.5

        opt = Adam(mlp, lr=0.01, max_grad_norm=1000.0)
        opt.step()

        gw = mlp.layers[0].grad_w.copy()
        assert np.sqrt(np.sum(gw ** 2)) < 1000.0


class TestAdamIndependentState:
    def test_two_independent_optimizers(self):
        rng = np.random.RandomState(42)
        mlp1 = MLP([2, 1])
        mlp2 = MLP([2, 1])
        mlp2.layers[0].w = mlp1.layers[0].w.copy()
        mlp2.layers[0].b = mlp1.layers[0].b.copy()
        x = rng.randn(10, 2)
        target = x @ np.array([[1.0], [2.0]]) + 0.5
        opt1 = Adam(mlp1, lr=0.1, betas=(0.9, 0.999))
        opt2 = Adam(mlp2, lr=0.1, betas=(0.9, 0.999))
        pred1 = mlp1.forward(x)
        grad1 = 2.0 * (pred1 - target) / x.shape[0]
        mlp1.backward(grad1)
        pred2 = mlp2.forward(x)
        grad2 = 2.0 * (pred2 - target) / x.shape[0]
        mlp2.backward(grad2)
        opt1.step()
        opt2.step()
        assert opt1._t == 1
        assert opt2._t == 1
        m1, v1 = opt1._moments[mlp1.layers[0]]
        m2, v2 = opt2._moments[mlp2.layers[0]]
        np.testing.assert_array_equal(m1["w"], m2["w"])
        np.testing.assert_array_equal(v1["w"], v2["w"])

        m1["w"][:] = 999.0
        assert not np.any(m2["w"] == 999.0), "Modifying one optimizer's state should not affect the other"

import gymnasium as gym
import numpy as np
import pytest

import aal_sim  # noqa: F401  (registers the env id)
from aal_sim import MachineTendingEnv, ScriptedOperator, Step
from aal_sim.hitl import OPERATOR, ROBOT, EpisodeLog, Gate, run_episode, sirius_weights, summarize
from aal_sim.policy import EnsemblePolicy


def test_gym_api():
    env = gym.make("AskAssistLearn/MachineTending-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert np.all(np.isfinite(obs))
    assert info["step"] == Step.PICK


def test_determinism():
    a = MachineTendingEnv(faults="hard").reset(seed=7)[0]
    b = MachineTendingEnv(faults="hard").reset(seed=7)[0]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("faults", ["none", "hard"])
def test_operator_completes_sop(faults):
    env = MachineTendingEnv(faults=faults)
    for seed in range(5):
        log = run_episode(env, "teleop", operator=ScriptedOperator(), seed=seed)
        assert log.success, (faults, seed, log.faults)
        assert log.operator_time > 0


def test_slip_drops_part_and_operator_recovers():
    env = MachineTendingEnv()
    log = run_episode(env, "teleop", seed=1, options={"force_faults": ["slip"]})
    assert "slip" in log.faults
    assert log.regressions >= 1
    assert log.success


def test_hitl_asks_and_logs_operator_samples():
    env = MachineTendingEnv()
    policy = EnsemblePolicy(env.obs_dim, 5, k=3, hidden=32)   # untrained: should get stuck and ask
    log = run_episode(env, "hitl", policy=policy, gate=Gate(stall_s=3.0), seed=0)
    assert log.asks
    assert (np.asarray(log.who) == OPERATOR).any()
    s = summarize([log])
    assert s["operator_min_per_robot_hour"] > 0


def test_sirius_weights_zero_pre_intervention():
    log = EpisodeLog(obs=[np.zeros(3)] * 60, act=[np.zeros(5)] * 60, who=[ROBOT] * 40 + [OPERATOR] * 20)
    _, _, w = sirius_weights([log], operator_w=2.0, robot_w=1.0, pre_window_s=0.5)
    assert np.all(w[:30] == 1.0) and np.all(w[30:40] == 0.0) and np.all(w[40:] == 2.0)


def test_policy_fit_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    O = rng.normal(size=(256, 4)).astype(np.float32)
    A = np.tanh(O[:, :2] @ np.ones((2, 5))).astype(np.float32) * 0.5
    pol = EnsemblePolicy(4, 5, k=2, hidden=32)
    loss = pol.fit(O, A, np.ones(len(O)), steps=300, batch=64)
    assert loss < 0.05
    pol.save(tmp_path / "p.pt")
    a1, _ = pol.act(O[0])
    a2, _ = EnsemblePolicy.load(tmp_path / "p.pt").act(O[0])
    np.testing.assert_allclose(a1, a2, atol=1e-6)

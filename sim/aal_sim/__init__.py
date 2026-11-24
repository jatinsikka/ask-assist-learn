"""Ask, Assist, Learn: MuJoCo machine-tending cell with human-in-the-loop learning."""

from gymnasium.envs.registration import register

from .env import FaultConfig, MachineTendingEnv, Step
from .expert import ScriptedOperator

register(id="AskAssistLearn/MachineTending-v0", entry_point="aal_sim.env:MachineTendingEnv")

__all__ = ["FaultConfig", "MachineTendingEnv", "ScriptedOperator", "Step"]

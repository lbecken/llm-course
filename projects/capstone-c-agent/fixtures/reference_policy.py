"""Reference policy for Capstone C: plan by search where the environment can be searched,
a fixed rule otherwise. It uses no model. It is the upper baseline your agent is compared
with, and an example of the required interface."""

from mlle.agents.envs import NumberGame
from mlle.agents.policies import SimulatedReActLLM, plan_then_execute


def make_policy(env):
    if isinstance(env, NumberGame):
        return plan_then_execute(env)  # breadth-first search over the three operations

    def explore(goal, history, actions):  # key-and-door world: the rule of the simulated stand-in, without mistakes
        state = next(s.observation for s in reversed(history) if s.status != "invalid")
        return SimulatedReActLLM._heuristic(goal, state, actions)

    return explore

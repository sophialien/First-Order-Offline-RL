import gym
import chex
import jax
import numpy as np
import jax.numpy as jnp

from copy import deepcopy
from tqdm.auto import trange
from typing import Sequence, Dict, Callable, Tuple, Union


@chex.dataclass(frozen=True)
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array([0.0]), jnp.array([0.0])) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        new_accumulators = deepcopy(self.accumulators)
        for key, value in updates.items():
            acc, steps = new_accumulators[key]
            new_accumulators[key] = (acc + value, steps + 1)

        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        # cumulative_value / total_steps
        return {k: np.array(v[0] / v[1]) for k, v in self.accumulators.items()}


def normalize(arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8) -> jax.Array:
    return (arr - mean) / (std + eps)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def evaluate(env: gym.Env, params, action_fn: Callable, num_episodes: int, seed: int) -> np.ndarray:
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    returns = []
    for _ in trange(num_episodes, desc="Eval", leave=False):
    # for _ in range(num_episodes):
        obs, done = env.reset(), False
        total_reward = 0.0
        while not done:
            action = np.asarray(jax.device_get(action_fn(params, obs)))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)

def apply_advQ_attack(last_states, actions, delta_state, critic, critic_params, robust_eps, rappo_on_state):
    # Random start.
    attack_adv_eps = robust_eps * jnp.abs(delta_state)
    clamp_min = last_states - attack_adv_eps
    clamp_max = last_states + attack_adv_eps

    key = jax.random.PRNGKey(758493)
    noise = (jax.random.uniform(key, shape=last_states.shape)*2-1)*attack_adv_eps*0.1
    states = last_states + noise

    def value_fn(states):
        return jnp.mean(critic.apply_fn(critic_params, states, actions).min(0))

    for _ in range(10):
        value, grad_states = jax.value_and_grad(value_fn)(states)
        update = jnp.sign(grad_states) * robust_eps / 10
        states = jnp.minimum(jnp.maximum(states - update, clamp_min), clamp_max)

    return jax.lax.stop_gradient(states)

def reset_state(env, adv_state):
    names = {'Hopper': 1 , 'Humanoid':2 , 'Walker2d': 1, 'Ant':2, 'HalfCheetah':1, 'Swimmer':2}
    #start = names[env.unwrapped.spec.id]
    start = 1

    current_state = env.sim.get_state()
    #print(adv_state.shape)
    current_state.qpos[start:] = adv_state[0:len(current_state.qpos)-start]
    current_state.qvel[:] = adv_state[len(current_state.qpos)-start:len(current_state.qpos)-start+len(current_state.qvel)]

    env.sim.set_state(current_state)
    env.sim.forward()


def adv_evaluate(env: gym.Env, params, action_fn: Callable, num_episodes: int, seed: int, critic, critic_params, adv_eps) -> np.ndarray:
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    returns = []
    for _ in trange(num_episodes, desc="Eval", leave=False):
    # for _ in range(num_episodes):
        obs, done = env.reset(), False
        obs_old=obs
        total_reward = 0.0
        steps = 1
        while not done:
            action = np.asarray(jax.device_get(action_fn(params, obs)))
            obs, reward, done, _ = env.step(action)
            if adv_eps != 0:
                if steps % 10 == 0:
                    adv_obs = apply_advQ_attack(obs, action, obs_old-obs, critic, critic_params, robust_eps=adv_eps, rappo_on_state=False)
                    reset_state(env, adv_obs)
            steps += 1
            total_reward += reward
            obs_old=obs
        returns.append(total_reward)
    return np.array(returns)

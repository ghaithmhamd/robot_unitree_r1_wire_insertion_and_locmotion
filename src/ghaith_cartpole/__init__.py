from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl.runner import MjlabOnPolicyRunner
from ghaith_cartpole.cartpole_env_cfg import (
    cartpole_balance_env_cfg,
    cartpole_swingup_env_cfg,
    cartpole_ppo_runner_cfg,
)

register_mjlab_task(
    task_id="Ghaith-Cartpole-Balance",
    env_cfg=cartpole_balance_env_cfg(),
    play_env_cfg=cartpole_balance_env_cfg(play=True),
    rl_cfg=cartpole_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
    task_id="Ghaith-Cartpole-Swingup",
    env_cfg=cartpole_swingup_env_cfg(),
    play_env_cfg=cartpole_swingup_env_cfg(play=True),
    rl_cfg=cartpole_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)

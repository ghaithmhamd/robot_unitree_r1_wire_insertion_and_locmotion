from pathlib import Path
SRC_PATH = Path(__file__).parent

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from .env_cfg import r1_flat_env_cfg
from .rl_cfg import r1_ppo_runner_cfg

register_mjlab_task(
    task_id="Ghaith-Velocity-Flat-Unitree-R1-Wire",
    env_cfg=r1_flat_env_cfg(),
    play_env_cfg=r1_flat_env_cfg(play=True),
    rl_cfg=r1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
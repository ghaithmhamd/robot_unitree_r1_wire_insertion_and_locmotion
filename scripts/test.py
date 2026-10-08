from ghaith_r1_wire.env_cfg import r1_flat_env_cfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

cfg = r1_flat_env_cfg(play=True)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

robot_asset = env.scene["robot"]
print("joint_names:", robot_asset.joint_names)
print("len:", len(robot_asset.joint_names))
# in a local mdp_events.py or wherever your custom mdp funcs live
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity


# def randomize_joint_qpos_by_pattern(env, env_ids, asset_cfg, pos_ranges):
#     import re
#     asset: Entity = env.scene[asset_cfg.name]
#     joint_ids = asset_cfg.joint_ids
#     joint_names = [asset.joint_names[i] for i in joint_ids]

#     low = torch.empty(len(joint_ids))
#     high = torch.empty(len(joint_ids))
#     for i, name in enumerate(joint_names):
#         lo, hi = (-0.1, 0.1)
#         for pattern, rng in pos_ranges.items():
#             if re.fullmatch(pattern, name):
#                 lo, hi = rng
#                 break
#         low[i], high[i] = lo, hi
#     low, high = low.to(env.device), high.to(env.device)

#     n = len(env_ids)
#     new_qpos = low + torch.rand(n, len(joint_ids), device=env.device) * (high - low)

#     asset.write_joint_state_to_sim(
#         position=new_qpos, velocity=torch.zeros_like(new_qpos),
#         joint_ids=joint_ids, env_ids=env_ids,
#     )

#     # Reshape env_ids to (N, 1) so it broadcasts against joint_ids (K,) -> (N, K)
#     env_ids_col = env_ids.reshape(-1, 1) if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=env.device).reshape(-1, 1)

#     asset.set_joint_position_target(
#         position=new_qpos, joint_ids=joint_ids, env_ids=env_ids_col,
#     )

def randomize_joint_qpos_by_pattern(env, env_ids, asset_cfg, pos_ranges):
    import re
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    joint_names = [asset.joint_names[i] for i in joint_ids]

    low = torch.empty(len(joint_ids))
    high = torch.empty(len(joint_ids))
    for i, name in enumerate(joint_names):
        lo, hi = (-0.1, 0.1)
        for pattern, rng in pos_ranges.items():
            if re.fullmatch(pattern, name):
                lo, hi = rng
                break
        low[i], high[i] = lo, hi
    low, high = low.to(env.device), high.to(env.device)

    n = len(env_ids)
    new_qpos = low + torch.rand(n, len(joint_ids), device=env.device) * (high - low)

    # <-- the write_joint_state_to_sim call that used to be here is DELETED.
    # Now the joint only gets a new TARGET, and has to physically move
    # there through the PD controller — real motion, real acceleration.

    env_ids_col = env_ids.reshape(-1, 1) if isinstance(env_ids, torch.Tensor) \
        else torch.as_tensor(env_ids, device=env.device).reshape(-1, 1)

    asset.set_joint_position_target(
        position=new_qpos, joint_ids=joint_ids, env_ids=env_ids_col,
    )
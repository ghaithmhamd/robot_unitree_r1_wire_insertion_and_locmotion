"""Unitree R1 velocity environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from .assets.unitree_r1.r1_constants import get_r1_robot_cfg, R1_WALKING_ACTION_SCALE
from .mdp_events import randomize_joint_qpos_by_pattern  # your module
from mjlab.envs.mdp import observations as obs_fns

# Foot sites and geoms from unitree_rl_mjlab R1 MJCF
SITE_NAMES = ("left_foot", "right_foot")
GEOM_NAMES = tuple(
    f"{side}_foot{i}_collision"
    for side in ("left", "right")
    for i in range(1, 8)
)

# Actuated joints only (no head #no arms)
ACTUATED_JOINTS = (
    # Lower Body
    ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
    # Waist
    "waist_roll_joint", "waist_yaw_joint",
    # # Upper Body 
    # ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    # # Arms
    # ".*_elbow_joint", ".*_wrist_roll_joint",
    # # Head
    # "head_pitch_joint", "head_yaw_joint",
    # # Fingers
    # "left_joint1_1", "right_joint1_1",

)
UPPER_BODY_JOINTS = (
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_joint", ".*_wrist_roll_joint",
    "head_pitch_joint", "head_yaw_joint",
    "left_joint1_1", "right_joint1_1",
)

def r1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree R1 flat terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": get_r1_robot_cfg()}

    # Set raycast sensor frame to R1 pelvis
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "pelvis"

    # Wire foot height scan to foot sites
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in SITE_NAMES
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

    # Foot contact sensors
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    # Action scale
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = R1_WALKING_ACTION_SCALE
    joint_pos_action.actuator_names = ACTUATED_JOINTS

    # Viewer
    cfg.viewer.body_name = "torso_link"
    cfg.sim.nconmax = 128

    # Command viz height
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.05

    # Events
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = GEOM_NAMES
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    UPPER_BODY_POS_RANGES = {
        # Shoulders
        r".*_shoulder_pitch_joint": (-3.14, 2.09),   # full range: -3.1416 .. 2.0944 (same both sides)
        r"left_shoulder_roll_joint": (-0.20, 2.45),   # limit: -0.22689 .. 2.4784
        r"right_shoulder_roll_joint": (-2.45, 0.20),  # limit: -2.47849 .. 0.2268 (mirrored!)
        r".*_shoulder_yaw_joint": (-1.90, 1.90),      # limit: -1.9199 .. 1.9199
        # Elbow / wrist
        r".*_elbow_joint": (-0.95, 2.15),             # limit: -0.97564 .. 2.1852
        r".*_wrist_roll_joint": (-1.90, 1.90),        # limit: -1.9199 .. 1.9199
        # Head
        r"head_pitch_joint": (-1.0, 1.0),             # limit: -0.62832 .. 0.62832
        r"head_yaw_joint": (-2.0, 2.0),               # limit: -2.0071 .. 2.0071
        # Fingers (using the tighter of left/right limits per joint)
        r"left_joint1_1": (-0.02, 0.0245),    
        r"right_joint1_1": (-0.02, 0.0245),     
    }
    # cfg.events["randomize_upper_body"] = EventTermCfg(
    #     func=randomize_joint_qpos_by_pattern,
    #     mode="interval",
    #     interval_range_s=(0.5, 2.0),  # continuously re-pose during the episode
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINTS),
    #         "pos_ranges": UPPER_BODY_POS_RANGES,
    #     },
    # )
    # cfg.events["randomize_upper_body"] = EventTermCfg(
    #     func=randomize_joint_qpos_by_pattern,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINTS),
    #         "pos_ranges": UPPER_BODY_POS_RANGES,
    #     },
    # )
    cfg.events["randomize_upper_body"] = EventTermCfg(
        func=randomize_joint_qpos_by_pattern,
        mode="interval",
        interval_range_s=(1.0, 1.0),   # every 1 second
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINTS),
            "pos_ranges": UPPER_BODY_POS_RANGES,
        },
    )

    upper_body_cfg = SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINTS)

    upper_body_term = ObservationTermCfg(
        func=obs_fns.joint_pos_rel,
        params={"asset_cfg": upper_body_cfg},
    )

    cfg.observations["actor"].terms["upper_body_joint_pos"] = upper_body_term
    cfg.observations["critic"].terms["upper_body_joint_pos"] = upper_body_term
    # Rewards - pose stds (actuated joints only)
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=ACTUATED_JOINTS
    )
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.5,
        r".*ankle_pitch.*": 0.15,
        r".*ankle_roll.*": 0.1,
        r".*waist_yaw.*": 0.15,
        r".*waist_roll.*": 0.1,
        # r".*shoulder_pitch.*": 0.15,
        # r".*shoulder_roll.*": 0.1,
        # r".*shoulder_yaw.*": 0.1,
        # r".*elbow.*": 0.1,
        # r".*wrist.*": 0.1,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.25,
        r".*hip_yaw.*": 0.25,
        r".*knee.*": 0.5,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        r".*waist_yaw.*": 0.25,
        r".*waist_roll.*": 0.1,
        # r".*shoulder_pitch.*": 0.25,
        # r".*shoulder_roll.*": 0.1,
        # r".*shoulder_yaw.*": 0.1,
        # r".*elbow.*": 0.1,
        # r".*wrist.*": 0.1,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = SITE_NAMES

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["air_time"].weight = 0.0

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )

    # Switch to flat terrain
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove terrain raycast sensor
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)

    # Play mode
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    return cfg

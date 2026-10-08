"""Unitree R1 constants."""

from pathlib import Path

import mujoco

#from src import SRC_PATH
from ghaith_r1_wire import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
#from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

R1_XML: Path = (
  SRC_PATH / "assets" / "unitree_r1" / "xmls" / "r1.xml"
)
assert R1_XML.exists()

#def get_assets(meshdir: str) -> dict[str, bytes]:
#  assets: dict[str, bytes] = {}
#  update_assets(assets, R1_XML.parent / "assets", meshdir)
#  return assets
def get_assets(meshdir: str) -> dict[str, bytes]:
  raise NotImplementedError("update_assets was removed in mjlab 1.4.0 — not needed for viewing.")


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(R1_XML))
  # spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

R1_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch.*",
    ".*_hip_roll.*",
    ".*_hip_yaw.*",
    ".*_knee.*",
  ),
  stiffness=100.0,
  damping=2.0,
  effort_limit=60.0,
  armature=0.01,
)
R1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch.*",
    ".*_ankle_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=50.0,
  armature=0.01,
)
R1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_.*",
  ),
  stiffness=100.0,
  damping=2.0,
  effort_limit=60.0,
  armature=0.01,
)
R1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch.*",
    ".*_shoulder_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=60.0,
  armature=0.01,
)
R1_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_yaw.*",
    ".*_elbow.*",
    ".*_wrist_roll.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=33.0,
  armature=0.01,
)
R1_ACTUATOR_HAND = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "left_joint1_1",
    "right_joint1_1",
  ),
  stiffness=800.0,
  damping=20.0,
  effort_limit=5000.0,
  armature=1,
)
R1_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "head_pitch_joint",
    "head_yaw_joint",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=33.0,
  armature=0.01,
)

##
# Keyframe config.
##

R1_STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    #".*_elbow_joint": 0.87,
    ".*_elbow_joint": 0.0,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,

    "left_joint1_1": 0.0,
    "right_joint1_1": 0.0,

    "head_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)
"""
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={
        r"^(left|right)_foot[1-7]_collision$": 3,
        r"^(left|right)_(thumb|index|middle|ring|pinky).*_collision$": 6,
        ".*_collision": 1,
    },
    priority={
        r"^(left|right)_foot[1-7]_collision$": 1,
        r"^(left|right)_(thumb|index|middle|ring|pinky).*_collision$": 1,
    },
    friction={
        r"^(left|right)_foot[1-7]_collision$": (0.6,),
        r"^(left|right)_(thumb|index|middle|ring|pinky).*_collision$": (1.0,),
    },
)
"""
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)
# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

R1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    R1_ACTUATOR_LEG,
    R1_ACTUATOR_ANKLE,
    R1_ACTUATOR_WAIST,
    R1_ACTUATOR_ARM,
    R1_ACTUATOR_WRIST,
    R1_ACTUATOR_HAND,
    R1_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=1.0,
)
def get_r1_robot_cfg() -> EntityCfg:
  """Get a fresh R1 robot configuration instance.
  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=R1_STANDING_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=R1_ARTICULATION,
  )
R1_ACTION_SCALE: dict[str, float] = {}
for a in R1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    R1_ACTION_SCALE[n] = 0.25 * e / s


############### for_the_walking_env_cfg ###############
R1_WALKING_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    R1_ACTUATOR_LEG,
    R1_ACTUATOR_ANKLE,
    R1_ACTUATOR_WAIST,
    # R1_ACTUATOR_ARM,
    # R1_ACTUATOR_WRIST,
    # R1_ACTUATOR_HAND,
    # R1_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)

R1_WALKING_ACTION_SCALE: dict[str, float] = {}
for a in R1_WALKING_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    R1_WALKING_ACTION_SCALE[n] = 0.25 * e / s
######################################################

if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity
  robot = Entity(get_r1_robot_cfg())
  viewer.launch(robot.spec.compile())

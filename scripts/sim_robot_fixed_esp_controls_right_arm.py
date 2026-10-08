import os
import sys

# Locate all conda nvidia libraries for PyCUDA and ONNX Runtime
site_pkgs = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
nvidia_dir = os.path.join(site_pkgs, "nvidia")

nvidia_libs = []
if os.path.exists(nvidia_dir):
    for pkg in os.listdir(nvidia_dir):
        lib_path = os.path.join(nvidia_dir, pkg, "lib")
        if os.path.isdir(lib_path):
            nvidia_libs.append(lib_path)

os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_libs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["BNB_CUDA_VERSION"] = "121"

LOCAL_MODEL_CACHE = "/home/ghaith-mhamdi/ghaith_mjlab_project/src/ghaith_r1_wire/model/roboflow_cache"
os.environ["MODEL_CACHE_DIR"] = LOCAL_MODEL_CACHE

# ==============================================================
# 0. Set Local Cache Directory & Offline Environment FIRST
# ==============================================================

import re
import time
import threading
import numpy as np
import cv2
import mujoco
import mujoco.viewer

from mjlab.entity.entity import Entity
from ghaith_r1_wire.assets.unitree_r1.r1_constants import get_r1_robot_cfg
from inference import get_model
from ghaith_r1_wire.read_joints_esp32_wifi import JointsReaderWiFi

ASSETS = "/home/ghaith-mhamdi/ghaith_mjlab_project/src/ghaith_r1_wire/assets"

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = "mhamdis-workspace/connector-hole-keypoints-4-rfdetr-keypoint-preview-t1"

if ROBOFLOW_API_KEY is None:
    raise RuntimeError("Set ROBOFLOW_API_KEY in environment before running")

print(f"Loading local Roboflow model from: {LOCAL_MODEL_CACHE}...")
local_model = get_model(model_id=ROBOFLOW_MODEL_ID, api_key=ROBOFLOW_API_KEY)
CONFIDENCE_THRESHOLD = 0.5

# ==============================================================
# Fixed poses — legs/waist/head/left-arm held here, right arm teleoperated
# ==============================================================
TRAINING_JOINT_DEFAULTS = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.0,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
    "left_joint1_1": 0.0,
    "right_joint1_1": 0.0,
    "head_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
}
SPAWN_POSE = {**TRAINING_JOINT_DEFAULTS}

# SIM_DT = 0.005
SIM_DT = 0.0005
DECIMATION = 4
POLICY_DT = SIM_DT * DECIMATION  # kept only for the pacing loop below

# ==============================================================
# Build Robot & Scene
# ==============================================================
robot = Entity(get_r1_robot_cfg())
spec = robot.spec

table = mujoco.MjSpec.from_file(f"{ASSETS}/table/table.xml")
connector = mujoco.MjSpec.from_file(f"{ASSETS}/connector/connector.xml")
cable = mujoco.MjSpec.from_file(f"{ASSETS}/cable/cable.xml")

table_frame = spec.worldbody.add_frame(pos=[0.8, 0, -0.1])
spec.attach(table, frame=table_frame, prefix="table_")

connector_frame = spec.worldbody.add_frame(pos=[0.45, 0.0, 0.675], euler=[-1.57, 0, 0])
spec.attach(connector, frame=connector_frame, prefix="connector_")

# cable_frame = spec.worldbody.add_frame(pos=[0.6, -0.1, 0.78], euler=[0, 0, 3.14])
cable_frame = spec.worldbody.add_frame(pos=[0.45, 0.0, 0.9], euler=[0, 0, 3.14])
spec.attach(cable, frame=cable_frame, prefix="cable_")

grid_tex = spec.add_texture(
    name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=[0.1, 0.2, 0.3], rgb2=[0.2, 0.3, 0.4], width=300, height=300,
)
grid_mat = spec.add_material(name="grid", reflectance=0.2)
grid_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
grid_mat.texrepeat = [8, 8]

spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
    pos=[0, 0, -0.05], size=[5, 5, 0.1], material="grid",
)
spec.worldbody.add_light(name="light", pos=[0, 0, 6])

# 1. Compile spec to model
model = spec.compile()

model.opt.gravity[:] = [0, 0, -9.81]
model.opt.timestep = SIM_DT
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON

# 2. CREATE MJDATA ONCE
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# 3. Set pelvis floating base pose — also stored here as the FIXED pose
# we re-pin every step (see the sim loop) so the robot can't fall/drift.
pelvis_jid = model.joint("floating_base_joint").id
qadr = model.jnt_qposadr[pelvis_jid]
PELVIS_QPOS_FIXED = np.array([0.1, 0.0, 0.675, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
data.qpos[qadr:qadr+7] = PELVIS_QPOS_FIXED

# 4. Set initial posture for ALL joints in data.qpos
for jid in range(model.njnt):
    jname = model.joint(jid).name
    for pattern, val in SPAWN_POSE.items():
        if re.fullmatch(pattern, jname):
            data.qpos[model.jnt_qposadr[jid]] = val
            break

mujoco.mj_forward(model, data)

# 5. Initialize target posture in data.ctrl for ALL actuators — this is
# now the ONLY place leg/waist/head/left-arm targets get set; nothing
# updates them afterward, so PD just holds this pose indefinitely.
for act_id in range(model.nu):
    joint_id = model.actuator(act_id).trnid[0]
    qpos_adr = model.jnt_qposadr[joint_id]
    data.ctrl[act_id] = data.qpos[qpos_adr]

# ==============================================================
# Right-arm teleop setup
# ==============================================================
SHOULDER_PITCH_ACT_ID = model.actuator("right_shoulder_pitch_joint").id
SHOULDER_ROLL_ACT_ID  = model.actuator("right_shoulder_roll_joint").id
SHOULDER_YAW_ACT_ID   = model.actuator("right_shoulder_yaw_joint").id
ELBOW_ACT_ID          = model.actuator("right_elbow_joint").id
WRIST_ACT_ID          = model.actuator("right_wrist_roll_joint").id
GRIPPER_ACT_ID = model.actuator("right_joint1_1").id

joints_reader = JointsReaderWiFi(listen_port=4210)
joints_reader.start()

pelvis_id = model.body("pelvis").id

# ==============================================================
# Vision (camera + Roboflow detection)
# ==============================================================
CAM_NAME = "robot_eye"

def draw_predictions(img_bgr, predictions):
    out = img_bgr.copy()
    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x0, y0 = int(x - w / 2), int(y - h / 2)
        x1, y1 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{pred['class']} {pred['confidence']:.2f}"
        cv2.putText(out, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for kp in pred.get("keypoints", []):
            if kp["confidence"] < CONFIDENCE_THRESHOLD: continue
            kx, ky = int(kp["x"]), int(kp["y"])
            cv2.circle(out, (kx, ky), 4, (0, 0, 255), -1)
            cv2.putText(out, kp["class"], (kx + 5, ky - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return out

latest_frame = None
latest_preds = []
frame_lock = threading.Lock()
stop_vision_thread = False

def local_vision_worker():
    global latest_preds
    while not stop_vision_thread:
        img_to_process = None
        with frame_lock:
            if latest_frame is not None:
                img_to_process = latest_frame.copy()

        if img_to_process is not None:
            try:
                results = local_model.infer(img_to_process)[0]
                preds = []
                raw_preds = getattr(results, "predictions", [])
                for pred in raw_preds:
                    p_dict = {
                        "x": pred.x, "y": pred.y,
                        "width": pred.width, "height": pred.height,
                        "class": getattr(pred, "class_name", getattr(pred, "label", "")),
                        "confidence": pred.confidence,
                        "keypoints": []
                    }
                    if hasattr(pred, "keypoints") and pred.keypoints:
                        for kp in pred.keypoints:
                            p_dict["keypoints"].append({
                                "x": kp.x, "y": kp.y,
                                "confidence": kp.confidence,
                                "class": getattr(kp, "class_name", getattr(kp, "label", ""))
                            })
                    preds.append(p_dict)
                latest_preds = preds
            except Exception as e:
                print("Local vision error:", e)

        time.sleep(0.01)

def display_worker():
    cv2.startWindowThread()
    while not stop_vision_thread:
        current_frame = None
        current_preds = []
        with frame_lock:
            if latest_frame is not None:
                current_frame = latest_frame.copy()
                current_preds = list(latest_preds)

        if current_frame is not None:
            img_bgr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            img_bgr = draw_predictions(img_bgr, current_preds)
            cv2.imshow("Robot Eye View", img_bgr)
            cv2.waitKey(1)

        time.sleep(0.016)

    cv2.destroyAllWindows()

v_thread = threading.Thread(target=local_vision_worker, daemon=True)
d_thread = threading.Thread(target=display_worker, daemon=True)
v_thread.start()
d_thread.start()

cam_renderer = mujoco.Renderer(model, height=480, width=640)

print("\n--- SIMULATION RUNNING (FIXED STANCE, RIGHT-ARM TELEOP ONLY) ---")

# ==============================================================
# Simulation Loop
# ==============================================================
step_counter = 0

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_step_time = time.perf_counter()

        while viewer.is_running():
            # Right-arm teleop from ESP32 — only thing driven every step.
            data.ctrl[SHOULDER_PITCH_ACT_ID] = joints_reader.get_shoulder_pitch_joint_angle()
            data.ctrl[SHOULDER_ROLL_ACT_ID]  = joints_reader.get_shoulder_roll_joint_angle()
            data.ctrl[SHOULDER_YAW_ACT_ID]   = joints_reader.get_shoulder_yaw_joint_angle()
            data.ctrl[ELBOW_ACT_ID]          = joints_reader.get_elbow_joint_angle()
            data.ctrl[WRIST_ACT_ID]          = joints_reader.get_wrist_roll_joint_angle()
            data.ctrl[GRIPPER_ACT_ID] = joints_reader.get_gripper_joint_pos()

            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            # Pin the pelvis to the fixed stance every step — hard guarantee
            # against any drift/fall, on top of the PD-held leg pose.
            data.qpos[qadr:qadr+7] = PELVIS_QPOS_FIXED
            data.qvel[model.jnt_dofadr[pelvis_jid]:model.jnt_dofadr[pelvis_jid]+6] = 0.0
            mujoco.mj_forward(model, data)

            step_counter += 1

            if step_counter % 3 == 0:
                cam_renderer.update_scene(data, camera=CAM_NAME)
                rgb_frame = cam_renderer.render()
                with frame_lock:
                    latest_frame = rgb_frame

            viewer.sync()

            next_step_time += POLICY_DT
            sleep_time = next_step_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_step_time = time.perf_counter()

except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
finally:
    stop_vision_thread = True
    joints_reader.stop()
    cv2.destroyAllWindows()
    print("Clean shutdown complete.")
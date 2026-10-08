import os
import mujoco
import mujoco.viewer
import cv2
import numpy as np
from mjlab.entity import Entity
from ghaith_r1_wire.assets.unitree_r1.r1_constants import get_r1_robot_cfg
import threading
import time

from inference_sdk import InferenceHTTPClient

ASSETS = "/home/ghaith-mhamdi/ghaith_mjlab_project/src/ghaith_r1_wire/assets"

# ---- Roboflow config ----
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")  # export ROBOFLOW_API_KEY=... beforehand
ROBOFLOW_MODEL_ID = "mhamdis-workspace/connector-hole-keypoints-4-rfdetr-keypoint-preview-t1"
# If this errors with "model not found", check the Deploy tab — some projects need
# a version suffix, e.g. ROBOFLOW_MODEL_ID + "/1"

if ROBOFLOW_API_KEY is None:
    raise RuntimeError(
        "Set ROBOFLOW_API_KEY in your environment before running "
        "(export ROBOFLOW_API_KEY=your_key)"
    )

rf_client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY,
)

# Run inference at most this often (hosted API round-trip is ~100-300ms).
# Since this now runs synchronously inside camera_loop, the display will
# pause for that long every time inference fires — raise this if the
# stutter is annoying.
INFERENCE_INTERVAL_S = 0.5
CONFIDENCE_THRESHOLD = 0.5  # keypoint-level; adjust to match what you used in the Roboflow UI test

robot_entity = Entity(get_r1_robot_cfg())
spec = robot_entity.spec

table = mujoco.MjSpec.from_file(f"{ASSETS}/table/table.xml")
connector = mujoco.MjSpec.from_file(f"{ASSETS}/connector/connector.xml")
#cable = mujoco.MjSpec.from_file(f"{ASSETS}/cable/cable.xml")
cable = mujoco.MjSpec.from_file(f"{ASSETS}/cable/cable_urdf.xml")

table_frame = spec.worldbody.add_frame(pos=[0.8, 0, 0])
spec.attach(table, frame=table_frame, prefix="table_")

#connector_frame = spec.worldbody.add_frame(pos=[0.6, 0.1, 0.8], euler=[-1.57, 0, 0])
connector_frame = spec.worldbody.add_frame(pos=[0.5, 0.1, 0.8], euler=[-1.57, 0, 0])
spec.attach(connector, frame=connector_frame, prefix="connector_")

#cable_frame = spec.worldbody.add_frame(pos=[0.6, -0.1, 0.78], euler=[0, 0, 3.14])
cable_frame = spec.worldbody.add_frame(pos=[1.0, -0.1, 0.78], euler=[0, 0, 3.14])
spec.attach(cable, frame=cable_frame, prefix="cable_")

# ---- Cartpole-style visuals ----

grid_tex = spec.add_texture(
    name="grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=[0.1, 0.2, 0.3],
    rgb2=[0.2, 0.3, 0.4],
    width=300,
    height=300,
)
grid_mat = spec.add_material(
    name="grid",
    reflectance=0.2,
)
grid_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
grid_mat.texrepeat = [8, 8]

deco_mat = spec.add_material(name="decoration", rgba=[0.3, 0.3, 0.3, 1])

spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    pos=[0, 0, -0.05],
    size=[5, 5, 0.1],
    material="grid",
)

spec.worldbody.add_light(name="light", pos=[0, 0, 6])

spec.worldbody.add_camera(
    name="fixed",
    pos=[0, -4, 1],
    zaxis=[0, -1, 0],
)
spec.worldbody.add_camera(
    name="lookatrobot",
    mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
    targetbody="table_table",
    pos=[0, -2, 2],
)

model = spec.compile()
model.opt.gravity[:] = [0, 0, 0]

data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

########
CAM_NAME = "robot_eye"
CAM_FPS = 20
stop_camera = False

# Only camera_loop ever touches this — no lock needed, but kept for
# clarity/safety in case you later read it from elsewhere.
_latest_predictions = []


def run_inference(img_rgb: np.ndarray):
    """Send one frame to the hosted Roboflow model and return its predictions list."""
    try:
        result = rf_client.infer(img_rgb, model_id=ROBOFLOW_MODEL_ID)
        return result.get("predictions", [])
    except Exception as e:
        print("Roboflow inference failed:", e)
        return []


def draw_predictions(img_bgr: np.ndarray, predictions: list) -> np.ndarray:
    out = img_bgr.copy()
    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x0, y0 = int(x - w / 2), int(y - h / 2)
        x1, y1 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{pred['class']} {pred['confidence']:.2f}"
        cv2.putText(out, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        for kp in pred.get("keypoints", []):
            if kp["confidence"] < CONFIDENCE_THRESHOLD:
                continue
            kx, ky = int(kp["x"]), int(kp["y"])
            cv2.circle(out, (kx, ky), 4, (0, 0, 255), -1)
            cv2.putText(out, kp["class"], (kx + 5, ky - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def camera_loop():
    """The only background thread, the only place that touches OpenGL.
    Renders + displays the camera feed, and every INFERENCE_INTERVAL_S
    calls Roboflow directly on the frame it just rendered — no second
    Renderer, no second thread involved in inference."""
    cam_renderer = mujoco.Renderer(model, height=480, width=640)
    interval = 1.0 / CAM_FPS
    last_infer_time = 0.0
    global _latest_predictions
    while not stop_camera:
        t0 = time.time()
        cam_renderer.update_scene(data, camera=CAM_NAME)
        img_rgb = cam_renderer.render()

        now = time.time()
        if now - last_infer_time >= INFERENCE_INTERVAL_S:
            last_infer_time = now
            _latest_predictions = run_inference(img_rgb)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = draw_predictions(img_bgr, _latest_predictions)

        cv2.imshow("Robot Eye View", img_bgr)
        cv2.waitKey(1)
        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))
    cv2.destroyAllWindows()


cv2.startWindowThread()
cam_thread = threading.Thread(target=camera_loop, daemon=True)
cam_thread.start()
###########

mujoco.viewer.launch(model, data)
stop_camera = True
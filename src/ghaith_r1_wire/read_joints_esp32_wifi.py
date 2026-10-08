"""
Joints reader — 5 potentiometers (on ESP32) over WiFi UDP.

Runs a background thread that listens for UDP packets from the ESP32,
each containing 5 comma-separated joint angles IN RADIANS (already
mapped to each joint's range on the ESP32 side), e.g.
"-0.50000,-1.20000,0.30000,1.00000,-0.20000", in this fixed order:
shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll.

No IMU, no orientation, no x/y/z, no IK -- each potentiometer directly
teleoperates one joint. Values are clipped to the corresponding
*_range_rad bounds as a safety net (the ESP32 sketch also clips to
these same ranges, so this is redundant unless the two get out of
sync -- keep them matching).

- get_shoulder_pitch_joint_angle() -> radians
- get_shoulder_roll_joint_angle()  -> radians
- get_shoulder_yaw_joint_angle()   -> radians
- get_elbow_joint_angle()          -> radians
- get_wrist_roll_joint_angle()     -> radians

Matches the ESP32 sketch (esp32_5pot_joints_udp.ino) that sends
`udp.print("q_shoulder_pitch,q_shoulder_roll,q_shoulder_yaw,q_elbow,q_wrist_roll")`
in radians.

ASSUMPTIONS (confirm these match your ESP32 sketch before trusting readings):
  - Packet field order is shoulder_pitch,shoulder_roll,shoulder_yaw,elbow,wrist_roll.
  - Values arrive already in radians, no unit conversion needed.
  - Default ranges below are taken from the MJCF <joint range="..."> values
    you provided -- update them here (and in the .ino) if your model differs.

File name and class name (read_joints_esp32_wifi.py / JointsReaderWiFi)
are fixed going forward -- adding a future joint means extending this
file (new <name>_range_rad param, new self._<name>_rad field, new
get_<name>_joint_angle() method, one more value in the packet/_worker
parse), not renaming anything.
"""

import socket
import threading
import time

import numpy as np


class JointsReaderWiFi:
    """Background UDP listener for 5 potentiometer-driven right-arm joint angles."""

    def __init__(
        self,
        listen_port: int = 4210,
        shoulder_pitch_range_rad: tuple[float, float] = (-3.1416, 2.0944),
        shoulder_roll_range_rad: tuple[float, float] = (-2.47849, 0.2268),
        shoulder_yaw_range_rad: tuple[float, float] = (-1.9199, 1.9199),
        elbow_range_rad: tuple[float, float] = (-0.97564, 2.1852),
        wrist_roll_range_rad: tuple[float, float] = (-1.9199, 1.9199),
        gripper_range_m: tuple[float, float] = (-0.02, 0.0251),
        socket_timeout_s: float = 0.5,
    ):
        self.listen_port = listen_port
        self.shoulder_pitch_range_rad = shoulder_pitch_range_rad
        self.shoulder_roll_range_rad = shoulder_roll_range_rad
        self.shoulder_yaw_range_rad = shoulder_yaw_range_rad
        self.elbow_range_rad = elbow_range_rad
        self.wrist_roll_range_rad = wrist_roll_range_rad
        self.gripper_range_m = gripper_range_m
        self.socket_timeout_s = socket_timeout_s

        # All joints default to 0.0 until the first packet arrives.
        self._shoulder_pitch_rad = 0.0
        self._shoulder_roll_rad = 0.0
        self._shoulder_yaw_rad = 0.0
        self._elbow_rad = 0.0
        self._wrist_roll_rad = 0.0
        self._gripper_m = 0.0
        self._lock = threading.Lock()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._last_packet_time = 0.0

    def start(self) -> None:
        """Start the background UDP-listening thread."""
        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop (call on shutdown)."""
        self._stop = True

    def get_shoulder_pitch_joint_angle(self) -> float:
        """Latest shoulder pitch angle, in radians, clipped to shoulder_pitch_range_rad."""
        with self._lock:
            return self._shoulder_pitch_rad

    def get_shoulder_roll_joint_angle(self) -> float:
        """Latest shoulder roll angle, in radians, clipped to shoulder_roll_range_rad."""
        with self._lock:
            return self._shoulder_roll_rad

    def get_shoulder_yaw_joint_angle(self) -> float:
        """Latest shoulder yaw angle, in radians, clipped to shoulder_yaw_range_rad."""
        with self._lock:
            return self._shoulder_yaw_rad

    def get_elbow_joint_angle(self) -> float:
        """Latest elbow angle, in radians, clipped to elbow_range_rad."""
        with self._lock:
            return self._elbow_rad

    def get_wrist_roll_joint_angle(self) -> float:
        """Latest wrist roll angle, in radians, clipped to wrist_roll_range_rad."""
        with self._lock:
            return self._wrist_roll_rad

    def get_gripper_joint_pos(self) -> float:
        """Latest gripper slide position, in meters, clipped to gripper_range_m."""
        with self._lock:
            return self._gripper_m

    def seconds_since_last_packet(self) -> float:
        """Useful for detecting a dead/disconnected ESP32."""
        return time.time() - self._last_packet_time

    def _worker(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.listen_port))
        sock.settimeout(self.socket_timeout_s)

        while not self._stop:
            try:
                data, _addr = sock.recvfrom(64)
                text = data.decode(errors="ignore").strip()
                if not text:
                    continue

                parts = text.split(",")
                if len(parts) != 6:
                    continue  # malformed packet, skip it

                sp_rad, sr_rad, sy_rad, el_rad, wr_rad, gr_m = (float(p) for p in parts)
                sy_rad = - sy_rad

                sp_rad = float(np.clip(sp_rad, *self.shoulder_pitch_range_rad))
                sr_rad = float(np.clip(sr_rad, *self.shoulder_roll_range_rad))
                sy_rad = float(np.clip(sy_rad, *self.shoulder_yaw_range_rad))
                el_rad = float(np.clip(el_rad, *self.elbow_range_rad))
                wr_rad = float(np.clip(wr_rad, *self.wrist_roll_range_rad))
                gr_m   = float(np.clip(gr_m, *self.gripper_range_m))

                with self._lock:
                    self._shoulder_pitch_rad = sp_rad
                    self._shoulder_roll_rad = sr_rad
                    self._shoulder_yaw_rad = sy_rad
                    self._elbow_rad = el_rad
                    self._wrist_roll_rad = wr_rad
                    self._gripper_m = gr_m
                self._last_packet_time = time.time()

            except socket.timeout:
                continue  # no packet in this window, keep last known values
            except ValueError:
                continue  # malformed packet, skip it

        sock.close()


if __name__ == "__main__":
    # Quick standalone test: run this file directly to print live readings
    # without needing the full sim. Turn each pot individually and confirm
    # the corresponding value here changes, and that it moves in the
    # direction you expect (e.g. turning the shoulder_pitch pot clockwise
    # should increase or decrease that value consistently -- verify before
    # wiring into the sim, since a flipped joint direction on a real robot
    # arm can cause unexpected/unsafe motion).
    reader = JointsReaderWiFi(listen_port=4210)
    reader.start()
    print("Listening for 5 joint angles over UDP on port 4210... Ctrl+C to stop.")
    try:
        while True:
            sp = reader.get_shoulder_pitch_joint_angle()
            sr = reader.get_shoulder_roll_joint_angle()
            sy = reader.get_shoulder_yaw_joint_angle()
            el = reader.get_elbow_joint_angle()
            wr = reader.get_wrist_roll_joint_angle()
            age = reader.seconds_since_last_packet()
            print(
                f"shoulder_pitch = {sp:+.4f} rad   "
                f"shoulder_roll = {sr:+.4f} rad   "
                f"shoulder_yaw = {sy:+.4f} rad   "
                f"elbow = {el:+.4f} rad   "
                f"wrist_roll = {wr:+.4f} rad   "
                f"last packet {age:.2f}s ago"
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
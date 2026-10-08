/*
  ESP32 + 5 potentiometers — direct right-arm joint teleoperation over WiFi (UDP)

  No IMU/orientation, no IK -- each potentiometer directly commands one
  joint's angle. This replaces the earlier MPU6050-based sketches
  entirely; the MPU6050 is no longer used or wired up.

  Wiring: each potentiometer's outer legs -> 3V3 and GND (either way
  round), wiper -> the pin listed below. All five pins are genuine
  ADC1 channels, so none of them become unreliable once WiFi is
  active (ADC2 pins do, on the ESP32 -- that's why these specific
  pins were picked).

    Pin : Joint
    32  : right_shoulder_pitch_joint
    33  : right_shoulder_roll_joint
    34  : right_shoulder_yaw_joint
    35  : right_elbow_joint
    36  : right_wrist_roll_joint

  Sends one UDP packet per reading (~50 Hz), payload =
  "q_shoulder_pitch,q_shoulder_roll,q_shoulder_yaw,q_elbow,q_wrist_roll"
  as plain text, ALL IN RADIANS. e.g.
  "-0.5000,-1.2000,0.3000,1.0000,-0.2000". Received by
  read_joints_esp32_wifi.py (JointsReaderWiFi) on the PC side.

  MAPPING: each pot's MECHANICAL CENTER = 0 rad for its joint, and the
  pot's own physical rotation angle IS the joint angle, 1:1 -- turn the
  pot 10 degrees off center, the joint moves 10 degrees off zero. This
  assumes a standard ~300 degree potentiometer (POT_SWEEP_DEG below),
  so the pot's full range is +-150 degrees from center. The result is
  then clipped to that joint's actual mechanical range (from the MJCF
  <joint range="..."> values) so a full pot sweep can't command past a
  real joint limit -- but note this means, for any joint whose real
  range is narrower than +-150 degrees, part of the pot's sweep does
  nothing (clipped flat at the limit), and for a joint whose range is
  wider than +-150 degrees in one direction, that extra range is simply
  unreachable by the pot. If your actual pot's mechanical sweep isn't
  ~300 degrees, update POT_SWEEP_DEG to match its datasheet/spec.

  SETUP REQUIRED BEFORE UPLOADING:
    1. Fill in WIFI_SSID / WIFI_PASSWORD below.
    2. Fill in PC_IP_ADDRESS with your computer's IP on the same WiFi
       network (find it with `ip addr` / `ifconfig` on Linux).
    3. Both the ESP32 and the PC must be on the same WiFi network/subnet.
*/

#include <WiFi.h>
#include <WiFiUdp.h>

// ---- Fill these in ----
const char* WIFI_SSID     = "Abcd";
const char* WIFI_PASSWORD = "bbbbbbbb";
const char* PC_IP_ADDRESS = "10.220.127.6";   // your PC's IP on the same network
const unsigned int UDP_PORT = 4210;           // must match read_joints_esp32_wifi.py
// ------------------------

WiFiUDP udp;

const unsigned long SEND_INTERVAL_MS = 20;  // ~50 Hz, matches sim control rate

// --- potentiometer pins (all ADC1 -- safe with WiFi active) ---
const int POT_SHOULDER_PITCH_PIN = 32;
const int POT_SHOULDER_ROLL_PIN  = 33;
const int POT_SHOULDER_YAW_PIN   = 34;
const int POT_ELBOW_PIN          = 35;
const int POT_WRIST_ROLL_PIN     = 36;
const int POT_GRIPPER_PIN        = 39; 

// --- joint ranges, in radians, from the MJCF <joint range="..."> values ---
const float SHOULDER_PITCH_MIN = -3.1416f, SHOULDER_PITCH_MAX = 2.0944f;
const float SHOULDER_ROLL_MIN  = -2.47849f, SHOULDER_ROLL_MAX = 0.2268f;
const float SHOULDER_YAW_MIN   = -1.9199f, SHOULDER_YAW_MAX  = 1.9199f;
const float ELBOW_MIN          = -0.97564f, ELBOW_MAX        = 2.1852f;
const float WRIST_ROLL_MIN     = -1.9199f, WRIST_ROLL_MAX    = 1.9199f;
const float GRIPPER_MIN = -0.02f, GRIPPER_MAX = 0.0251f;

// pot's real mechanical rotation, in degrees, center = 0 -- see mapping note above
const float POT_SWEEP_DEG = 300.0f;

unsigned long lastSend = 0;

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

// Reads a pot (0-4095), treats its mechanical CENTER as 0 degrees, converts
// the pot's own rotation angle directly to radians (1:1), then clips to
// [rangeMin, rangeMax] so it can't exceed that joint's real mechanical limit.
float readPotAsJointAngle(int pin, float rangeMin, float rangeMax) {
  int raw = analogRead(pin);
  float norm = (raw / 4095.0f) - 0.5f;      // -0.5 .. +0.5, center = 0
  float potDeg = norm * POT_SWEEP_DEG;      // pot's real rotation, degrees, center = 0
  float jointRad = radians(potDeg);         // 1:1 -- pot degrees become joint radians directly
  if (jointRad < rangeMin) jointRad = rangeMin;
  if (jointRad > rangeMax) jointRad = rangeMax;
  return jointRad;
}

float readPotAsLinear(int pin, float rangeMin, float rangeMax) {
  int raw = analogRead(pin);
  float t = raw / 4095.0f;  // 0..1
  return rangeMin + t * (rangeMax - rangeMin);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { }

  // No pinMode() needed for analogRead() on ADC1 pins.

  connectWiFi();

  Serial.println("Streaming 5 joint angles (radians) over UDP.");

  lastSend = millis();
}

void loop() {
  // Reconnect if WiFi drops
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  unsigned long now = millis();
  if (now - lastSend >= SEND_INTERVAL_MS) {
    lastSend = now;

    float qShoulderPitch = readPotAsJointAngle(POT_SHOULDER_PITCH_PIN, SHOULDER_PITCH_MIN, SHOULDER_PITCH_MAX);
    float qShoulderRoll  = readPotAsJointAngle(POT_SHOULDER_ROLL_PIN,  SHOULDER_ROLL_MIN,  SHOULDER_ROLL_MAX);
    float qShoulderYaw   = readPotAsJointAngle(POT_SHOULDER_YAW_PIN,   SHOULDER_YAW_MIN,   SHOULDER_YAW_MAX);
    float qElbow         = readPotAsJointAngle(POT_ELBOW_PIN,          ELBOW_MIN,          ELBOW_MAX);
    float qWristRoll     = readPotAsJointAngle(POT_WRIST_ROLL_PIN,     WRIST_ROLL_MIN,     WRIST_ROLL_MAX);
    float qGripper       = readPotAsLinear(POT_GRIPPER_PIN, GRIPPER_MIN, GRIPPER_MAX);

    char b0[16], b1[16], b2[16], b3[16], b4[16], b5[16];
    dtostrf(qShoulderPitch, 0, 5, b0);
    dtostrf(qShoulderRoll,  0, 5, b1);
    dtostrf(qShoulderYaw,   0, 5, b2);
    dtostrf(qElbow,         0, 5, b3);
    dtostrf(qWristRoll,     0, 5, b4);
    dtostrf(qGripper,       0, 5, b5);

    char packet[112];
    snprintf(packet, sizeof(packet), "%s,%s,%s,%s,%s,%s", b0, b1, b2, b3, b4, b5);

    udp.beginPacket(PC_IP_ADDRESS, UDP_PORT);
    udp.print(packet);
    udp.endPacket();
  }
}
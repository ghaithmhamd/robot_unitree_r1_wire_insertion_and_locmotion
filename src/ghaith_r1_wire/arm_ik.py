"""
Right-arm inverse kinematics (numerical, least-squares).

Chain (from your notebook, base frame = shoulder_pitch origin, parallel to waist):
    q1 -> right_shoulder_pitch
    q2 -> right_shoulder_roll
    q3 -> right_shoulder_yaw
    q4 -> right_elbow
    q5 -> right_wrist_roll
End effector: physical origin of right_wrist_roll_link (includes the fixed
112.18 mm tool offset along the local z5 axis, so it doesn't move with q5).

This is a 5-DOF arm. A fully general 6-DOF pose (x,y,z,roll,pitch,yaw) is NOT
always exactly reachable -- there's no closed-form IK for the general case.
Instead we minimize a weighted position + orientation error numerically. If a
target is reachable, the residual goes to ~0; if not, you get the closest
feasible pose (and solve_ik() tells you how close it got).

ASSUMPTIONS (change these if wrong for your setup):
  - RPY convention: R_target = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (intrinsic ZYX)
  - Joint limits: placeholder [-pi, pi] for all 5 joints -- replace with your
    actual MJCF/URDF ranges in JOINT_LIMITS below before trusting solutions
    near the edges.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rsc

# ---------------------------------------------------------------------------
# 1. Numeric DH constants, copied directly from your notebook's printed table
#    (i, a_i [m], alpha_i [rad], d_i [m], theta_off_i [rad])
# ---------------------------------------------------------------------------
DH = [
    # a,          alpha,        d,           theta_off
    (0.0,         1.83266600,   0.0,         3.14159265),
    (0.02569300,  1.57079633,  -0.04713200, -1.57079633),
    (0.00430000,  1.57079633,   0.0,        -1.30892665),
    (0.01619100, -1.57079633,  -0.19120800,  1.57079633),
    (0.01170200,  1.57079633,   0.00355900,  1.57079633),
]
TOOL_OFFSET = 0.11218000  # along local z5, doesn't change with q5

# TODO: replace with the real limits from your MJCF <joint range="...">
JOINT_LIMITS = [(-np.pi, np.pi)] * 5


def dh_matrix(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d],
        [0,        0,        0,      1],
    ])


def fk(q):
    """
    Forward kinematics.
    q: array-like of 5 joint values [q1..q5] (radians)
    returns: (p, R) -- end-effector position (3,) and rotation matrix (3,3)
             in the base frame (shoulder_pitch origin, parallel to waist)
    """
    T = np.eye(4)
    for (a, alpha, d, theta_off), qi in zip(DH, q):
        T = T @ dh_matrix(a, alpha, d, theta_off + qi)
    R = T[0:3, 0:3]
    p = T[0:3, 3] + R @ np.array([0.0, 0.0, TOOL_OFFSET])
    return p, R


def rpy_to_R(roll, pitch, yaw):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) -- intrinsic ZYX. See assumptions above."""
    return Rsc.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()


def R_to_rpy(R):
    yaw, pitch, roll = Rsc.from_matrix(R).as_euler('ZYX')
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# 2. IK residual + solver
# ---------------------------------------------------------------------------
def _residual_full(q, p_target, R_target, w_pos, w_rot):
    p, R = fk(q)
    pos_err = w_pos * (p - p_target)
    # orientation error as a rotation vector (axis * angle) of R_target * R_current^T
    rot_err = w_rot * Rsc.from_matrix(R_target @ R.T).as_rotvec()
    return np.concatenate([pos_err, rot_err])


def _residual_pos_only(q, p_target, w_pos):
    p, _R = fk(q)
    return w_pos * (p - p_target)


def solve_ik(x, y, z, roll=None, pitch=None, yaw=None,
             q_init=None, w_pos=1.0, w_rot=1.0,
             joint_limits=JOINT_LIMITS, n_restarts=8, tol=1e-6, verbose=False):
    """
    x, y, z         : target end-effector position in the base frame (m)
    roll,pitch,yaw  : target end-effector orientation (rad), see RPY convention above.
                       Pass all three, or leave all three as None for FREE
                       ORIENTATION mode: orientation is dropped from the
                       optimization entirely and only position is solved
                       for. With 5 joints and only 3 position constraints,
                       there are 2 redundant DOF -- in free-orientation mode
                       those just settle wherever the local solver's search
                       from q_init lands, so orientation is NOT controlled
                       or predictable, just whatever falls out. Useful when
                       a fixed target orientation (e.g. 0,0,0) has no
                       solution at some positions but you don't actually
                       need to constrain orientation.
    q_init          : optional (5,) warm-start guess -- pass your last solution
                       here in a control loop for smooth, continuous motion.
                       Matters even more in free-orientation mode, since it's
                       what determines which of the redundant solutions you get.
    w_pos, w_rot     : relative weights if position accuracy matters more than
                       orientation accuracy (or vice versa) when the target
                       isn't exactly reachable. w_rot is ignored in
                       free-orientation mode.
    n_restarts      : random restarts to try to escape local minima /
                       find a valid solution (ignored if q_init is given and works)
    returns: dict with 'q' (best solution), 'pos_error' (m), 'success' (bool).
             Also includes 'rot_error' (rad) unless in free-orientation mode,
             where there's no orientation target to measure error against.
    """
    free_orientation = roll is None and pitch is None and yaw is None
    if not free_orientation and (roll is None or pitch is None or yaw is None):
        raise ValueError(
            "Pass all three of roll, pitch, yaw, or leave all three as None "
            "for free-orientation mode -- partial orientation isn't supported."
        )

    p_target = np.array([x, y, z])
    R_target = None if free_orientation else rpy_to_R(roll, pitch, yaw)

    lo = np.array([lim[0] for lim in joint_limits])
    hi = np.array([lim[1] for lim in joint_limits])

    starts = []
    if q_init is not None:
        starts.append(np.asarray(q_init, dtype=float))
    rng = np.random.default_rng(0)
    starts += [rng.uniform(lo, hi) for _ in range(n_restarts)]

    best = None
    for q0 in starts:
        q0 = np.clip(q0, lo, hi)
        if free_orientation:
            res = least_squares(
                _residual_pos_only, q0, args=(p_target, w_pos),
                bounds=(lo, hi), method='trf', xtol=1e-12, ftol=1e-12,
            )
        else:
            res = least_squares(
                _residual_full, q0, args=(p_target, R_target, w_pos, w_rot),
                bounds=(lo, hi), method='trf', xtol=1e-12, ftol=1e-12,
            )
        p_sol, R_sol = fk(res.x)
        pos_err = np.linalg.norm(p_sol - p_target)

        if free_orientation:
            score = pos_err
            candidate = dict(q=res.x, pos_error=pos_err, score=score)
            good_enough = pos_err < tol
        else:
            rot_err = np.linalg.norm(Rsc.from_matrix(R_target @ R_sol.T).as_rotvec())
            score = pos_err + rot_err
            candidate = dict(q=res.x, pos_error=pos_err, rot_error=rot_err, score=score)
            good_enough = pos_err < tol and rot_err < tol

        if best is None or score < best['score']:
            best = candidate
        if good_enough:
            break  # good enough, stop early

    if free_orientation:
        best['success'] = bool(best['pos_error'] < tol)
    else:
        best['success'] = bool(best['pos_error'] < tol and best['rot_error'] < tol)
    del best['score']
    if verbose:
        print(f"q = {np.round(best['q'], 5)}")
        if free_orientation:
            print(f"pos_error = {best['pos_error']:.2e} m, success = {best['success']}")
        else:
            print(f"pos_error = {best['pos_error']:.2e} m, "
                  f"rot_error = {best['rot_error']:.2e} rad, success = {best['success']}")
    return best


# ---------------------------------------------------------------------------
# 3. Self-test: round-trip against the same test_q you used in the notebook
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_q = np.array([0.3, -0.4, 0.2, 0.6, 0.1])
    p, R = fk(test_q)
    print("FK sanity check (should match your notebook's p_EE_func(test_q)):")
    print("p =", p)
    print("expected ~[0.13777991, 0.11927662, -0.15571775]")
    print()

    roll, pitch, yaw = R_to_rpy(R)
    print("Now solving IK for that exact pose (should recover test_q, up to")
    print("redundancy/symmetry -- any q with the same p,R is a valid answer):")
    sol = solve_ik(*p, roll, pitch, yaw, verbose=True)
    print("original q1..q5:", np.round(test_q, 5))

    print()
    print("Free-orientation mode: solve for position only (roll/pitch/yaw")
    print("omitted), orientation left to settle wherever the solver lands:")
    sol_free = solve_ik(*p, verbose=True)
    p_check, _R_check = fk(sol_free['q'])
    print("position achieved:", p_check, " (target was:", p, ")")
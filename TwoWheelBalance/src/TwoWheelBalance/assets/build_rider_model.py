"""Build the chair + mannequin model for the rider task: low-poly meshes (OBJ) and two URDFs. Needs numpy and scipy.

  python build_rider_model.py <output folder>      (writes <out>/tw_balance_rider.urdf, <out>/tw_balance_rider_simple.urdf,
                                                   <out>/meshes/*.obj and <out>/rider_model_info.json)

Chair: shaped after the Genny Zero (Ottobock): seat 64 cm above the floor, seat 46 x 44 cm, segmented backrest,
central steering column ("tiller") with two handgrips, low side rails, footplate, chassis between the wheels. The PHYSICS of the chair and wheels is unchanged from
tw_balance.urdf (55 kg body, CoM 0.305 m above the axle, 2 x 5 kg wheels of radius 0.3 m, track 0.61 m).

Mannequin: 1.75 m adult, segment lengths and mass fractions from Winter (2009), "Biomechanics and Motor Control of
Human Movement", table 4.1 (Dempster's data). Seated with the hip joints 9 cm above the seat. Two links:
  rider_legs  pelvis + thighs + shanks + feet  = 46.4 % of the rider mass, fixed to the chair
  rider       head + neck + trunk + arms       = 53.6 % of the rider mass, on the hip joint "rider_joint" (leans)
Both links have 1 kg in the URDF; the rider_mass event adds fraction * rider mass at every reset and scales the
inertia with the mass.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

# ------------------------------------------------------------------------------------------------ anthropometry
H = 1.75                               # stature [m]
A2B = -0.305                           # z(base_link frame) = z(axle frame) + A2B
HIP_A = np.array([-0.10, 0.0, 0.43])   # hip joint centre in the axle frame (x fwd, z up): 9 cm above the seat
SEAT_TOP_A = 0.34                      # 0.64 m above the floor (wheel radius 0.30)

L_THIGH, L_SHANK = 0.245 * H, 0.246 * H            # 0.429, 0.431
L_UPARM, L_FOREARM = 0.186 * H, 0.146 * H          # 0.326, 0.256
ANKLE_H, FOOT_L = 0.039 * H, 0.152 * H             # 0.068, 0.266
HIP_W = 0.09                                       # half distance between the hip joints
HEAD_C = np.array([0.005, 0.0, 0.695])             # head centre relative to the hip; head 0.23 m high, 0.16 m wide
HEAD_SEMI = (0.10, 0.08, 0.115)
SHOULDER = np.array([-0.03, 0.185, 0.465])         # shoulder joint centre relative to the hip (seated upright)

# seated posture (relative to the hip joint, link frames sit at the hip)
th_thigh = math.radians(5.0)                       # thighs 5 deg below horizontal
th_shank = math.radians(10.0)                      # shanks 10 deg forward of vertical
KNEE = np.array([L_THIGH * math.cos(th_thigh), HIP_W, -L_THIGH * math.sin(th_thigh)])
ANKLE = KNEE + np.array([L_SHANK * math.sin(th_shank), 0.01, -L_SHANK * math.cos(th_shank)])
# hands on the two grips of the steering column (tiller): solve the elbow (2-link arm in the sagittal plane)
GRIP_A = np.array([0.37, 0.20, 0.585])             # grip centre in the axle frame (one per side, +-y)
HAND_OFF = 0.07                                    # wrist -> centre of the hand along the forearm


def _arm_ik():
    g = GRIP_A - HIP_A                             # grip relative to the hip
    w = g.copy()
    for _ in range(20):                            # wrist = grip - HAND_OFF * forearm direction (iterate)
        d = np.array([w[0] - SHOULDER[0], w[2] - SHOULDER[2]])
        L = np.linalg.norm(d)
        a, b = L_UPARM, L_FOREARM
        alpha = math.acos(np.clip((a * a + L * L - b * b) / (2 * a * L), -1, 1))
        ang = math.atan2(d[1], d[0]) - alpha        # elbow below the shoulder-wrist line
        e = np.array([SHOULDER[0] + a * math.cos(ang), 0.21, SHOULDER[2] + a * math.sin(ang)])
        fdir = np.array([w[0] - e[0], 0.0, w[2] - e[2]]); fdir /= np.linalg.norm(fdir)
        w = g - HAND_OFF * fdir
        w[1] = g[1]
    return e, w


ELBOW, WRIST = _arm_ik()

# mass fractions of body mass (Dempster via Winter)
FRAC = dict(head=0.081, thorax=0.216, abdomen=0.139, pelvis=0.142, upper_arm=0.028, forearm=0.016, hand=0.006,
            thigh=0.100, shank=0.0465, foot=0.0145)

# ------------------------------------------------------------------------------------------------ colours
COL = {
    "chassis": (0.17, 0.18, 0.20, 1.0), "accent": (0.13, 0.36, 0.69, 1.0), "cushion": (0.08, 0.08, 0.09, 1.0),
    "metal": (0.62, 0.64, 0.67, 1.0), "rubber": (0.06, 0.06, 0.06, 1.0), "tyre": (0.07, 0.07, 0.07, 1.0),
    "hub": (0.55, 0.57, 0.60, 1.0),
    "mannequin": (0.87, 0.73, 0.56, 1.0), "joint": (0.66, 0.50, 0.36, 1.0),
}


# ------------------------------------------------------------------------------------------------ low-poly meshes
class Mesh:
    def __init__(self):
        self.tris = []                                  # list of (3,3) arrays

    def add(self, verts, faces):
        v = np.asarray(verts, float)
        for f in faces:
            for k in range(1, len(f) - 1):              # fan-triangulate polygons
                self.tris.append(v[[f[0], f[k], f[k + 1]]])
        return self

    def extend(self, other):
        self.tris += other.tris
        return self

    def transformed(self, R=np.eye(3), t=np.zeros(3)):
        m = Mesh()
        m.tris = [(R @ tr.T).T + t for tr in self.tris]
        return m

    def write_obj(self, path):
        lines = ["# low-poly mesh written by build_rider_model.py (flat shading: one normal per face)"]
        vn = []
        for i, tr in enumerate(self.tris):
            n = np.cross(tr[1] - tr[0], tr[2] - tr[0])
            ln = np.linalg.norm(n)
            n = n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
            for p in tr:
                lines.append(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
            vn.append(n)
        for n in vn:
            lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
        for i in range(len(self.tris)):
            a = 3 * i + 1
            lines.append(f"f {a}//{i + 1} {a + 1}//{i + 1} {a + 2}//{i + 1}")
        open(path, "w").write("\n".join(lines) + "\n")


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def frame_from_axis(d):
    """rotation whose z axis is d"""
    z = d / np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def ellipsoid(c, semi, n_lon=10, n_lat=6, R=np.eye(3)):
    v, f = [], []
    for i in range(n_lat + 1):
        th = math.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * math.pi * j / n_lon
            v.append([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    v = np.array(v) * np.asarray(semi)
    v = (R @ v.T).T + np.asarray(c)
    for i in range(n_lat):
        for j in range(n_lon):
            a, b = i * n_lon + j, i * n_lon + (j + 1) % n_lon
            cc, d = a + n_lon, b + n_lon
            if i == 0:
                f.append([a, cc, d])
            elif i == n_lat - 1:
                f.append([a, cc, b])
            else:
                f.append([a, cc, d, b])
    return Mesh().add(v, f)


def capsule(p0, p1, r0, r1, n=10, rings=3):
    """tapered capsule: frustum between p0 (radius r0) and p1 (r1) with hemispherical end caps"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    R = frame_from_axis(p1 - p0)
    prof = []                                            # (z along axis, radius) from p0 cap tip to p1 cap tip
    for k in range(rings, 0, -1):
        a = math.pi / 2 * k / rings
        prof.append((-r0 * math.sin(a), r0 * math.cos(a)))
    prof.append((0.0, r0))
    prof.append((L, r1))
    for k in range(1, rings + 1):
        a = math.pi / 2 * k / rings
        prof.append((L + r1 * math.sin(a), r1 * math.cos(a)))
    v, f = [], []
    for z, r in prof:
        for j in range(n):
            ph = 2 * math.pi * j / n
            v.append([r * math.cos(ph), r * math.sin(ph), z])
    v = (R @ np.array(v).T).T + p0
    m = len(prof)
    for i in range(m - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            f.append([a, b, b + n, a + n])
    return Mesh().add(v, f)


def rbox(c, size, r=0.02, R=np.eye(3), n=3):
    """box with rounded edges (superellipsoid-like via clamped sphere mapping), low poly"""
    hx, hy, hz = np.asarray(size, float) / 2
    r = min(r, hx, hy, hz)
    inner = np.array([hx - r, hy - r, hz - r])
    m = ellipsoid((0, 0, 0), (r, r, r), n_lon=4 * n, n_lat=2 * n)
    tris = []
    for tr in m.tris:
        t2 = tr.copy()
        t2 += np.sign(tr) * inner                         # push the 8 octants of the sphere out to the box corners
        tris.append(t2)
    # faces between octants: rebuild as convex hull for a clean closed shape
    pts = np.unique(np.round(np.concatenate(tris), 6), axis=0)
    out = convex_hull_mesh(pts)
    return out.transformed(R, np.asarray(c, float))


def convex_hull_mesh(pts):
    from scipy.spatial import ConvexHull
    h = ConvexHull(pts)
    m = Mesh()
    cen = pts.mean(0)
    for s in h.simplices:
        tri = pts[s]
        nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.dot(nrm, tri.mean(0) - cen) < 0:
            tri = tri[[0, 2, 1]]
        m.tris.append(tri)
    return m


def cylinder_y(c, r, w, n=16):
    """cylinder with axis along y"""
    v, f = [], []
    for s in (-w / 2, w / 2):
        for j in range(n):
            ph = 2 * math.pi * j / n
            v.append([r * math.cos(ph), s, r * math.sin(ph)])
    v = np.array(v) + np.asarray(c)
    for j in range(n):
        a, b = j, (j + 1) % n
        f.append([a, a + n, b + n, b])
    f.append(list(range(n)))                             # cap at -y (normal -y)
    f.append(list(range(n, 2 * n))[::-1])                # cap at +y (normal +y)
    return Mesh().add(v, f)


def torus_y(c, R0, rr, ry, n=24, m=8):
    """tyre: torus around the y axis, tube radius rr (radial) and ry (axial)"""
    v, f = [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        for j in range(m):
            b = 2 * math.pi * j / m
            rad = R0 + rr * math.cos(b)
            v.append([rad * math.cos(a), ry * math.sin(b), rad * math.sin(a)])
    v = np.array(v) + np.asarray(c)
    for i in range(n):
        for j in range(m):
            a0, a1 = i * m + j, i * m + (j + 1) % m
            b0, b1 = ((i + 1) % n) * m + j, ((i + 1) % n) * m + (j + 1) % m
            f.append([a0, a1, b1, b0])
    return Mesh().add(v, f)


# ------------------------------------------------------------------------------------------------ parts
def to_b(p_axle):
    """axle frame -> base_link frame"""
    p = np.asarray(p_axle, float).copy()
    p[2] += A2B
    return p


def chair_parts():
    """visual meshes on base_link (base frame), grouped by colour"""
    P = {k: Mesh() for k in ("chassis", "accent", "cushion", "metal", "rubber")}
    # chassis between the wheels (motors on the axle, batteries below the seat)
    P["chassis"].extend(rbox(to_b([0.0, 0, 0.02]), (0.40, 0.44, 0.38), r=0.06, n=3))
    # blue side covers (Genny accent colour)
    for s in (1, -1):
        P["accent"].extend(rbox(to_b([0.0, s * 0.228, 0.02]), (0.34, 0.02, 0.30), r=0.01, n=2))
    # seat post and seat shell
    P["chassis"].extend(rbox(to_b([-0.01, 0, 0.235]), (0.22, 0.24, 0.08), r=0.02, n=2))
    P["chassis"].extend(rbox(to_b([0.0, 0, 0.278]), (0.48, 0.48, 0.045), r=0.018, n=2))
    # cushion: seat 46 x 44 cm, top 64 cm above the floor
    P["cushion"].extend(rbox(to_b([0.0, 0, SEAT_TOP_A - 0.028]), (0.44, 0.46, 0.056), r=0.025, n=3))
    # backrest: segmented "Tarta" spine behind a padded back
    rec = math.radians(-6.0)
    pad_c = np.array([-0.255, 0, SEAT_TOP_A + 0.30])
    P["cushion"].extend(rbox(to_b(pad_c), (0.05, 0.40, 0.44), r=0.02, R=rot_y(rec), n=3))
    for k in range(7):
        z = SEAT_TOP_A + 0.10 + k * 0.058
        x = -0.30 - (z - pad_c[2]) * math.tan(-rec)
        P["metal" if k % 2 else "chassis"].extend(rbox(to_b([x, 0, z]), (0.035, 0.11, 0.05), r=0.012, R=rot_y(rec), n=2))
    P["chassis"].extend(rbox(to_b([-0.27, 0, SEAT_TOP_A - 0.02]), (0.06, 0.12, 0.08), r=0.015, n=2))   # spine bracket
    # steering column ("tiller", turns left/right) from the front of the chassis up between the knees,
    # control centre and a bar with two handgrips (the hands of the mannequin hold them)
    top = np.array([GRIP_A[0] - 0.01, 0.0, GRIP_A[2] - 0.035])
    P["metal"].extend(capsule(to_b([0.17, 0, 0.12]), to_b(top), 0.022, 0.020, n=10, rings=2))
    P["chassis"].extend(rbox(to_b(top + [0.005, 0, 0.035]), (0.07, 0.13, 0.05), r=0.015, R=rot_y(math.radians(-20)), n=2))
    P["accent"].extend(rbox(to_b(top + [0.03, 0, 0.045]), (0.012, 0.10, 0.03), r=0.004, R=rot_y(math.radians(-20)), n=1))
    P["metal"].extend(capsule(to_b([GRIP_A[0], -0.15, GRIP_A[2]]), to_b([GRIP_A[0], 0.15, GRIP_A[2]]), 0.016, 0.016, n=8, rings=2))
    for s in (1, -1):
        P["rubber"].extend(capsule(to_b([GRIP_A[0], s * 0.15, GRIP_A[2]]), to_b([GRIP_A[0], s * 0.265, GRIP_A[2]]), 0.021, 0.021, n=10, rings=2))
        # low side rails on the seat shell (for transfers)
        for xx in (-0.10, 0.13):
            P["metal"].extend(capsule(to_b([xx, s * 0.25, SEAT_TOP_A - 0.05]), to_b([xx, s * 0.25, SEAT_TOP_A + 0.06]), 0.012, 0.012, n=8, rings=2))
        P["metal"].extend(capsule(to_b([-0.10, s * 0.25, SEAT_TOP_A + 0.06]), to_b([0.13, s * 0.25, SEAT_TOP_A + 0.06]), 0.012, 0.012, n=8, rings=2))
    # footplate under the feet (top = soles), bar to the chassis
    sole_z = (HIP_A + ANKLE)[2] - ANKLE_H
    P["chassis"].extend(rbox(to_b([0.42, 0, sole_z - 0.016]), (0.24, 0.34, 0.032), r=0.012, n=2))
    P["metal"].extend(capsule(to_b([0.17, 0, -0.04]), to_b([0.33, 0, sole_z - 0.02]), 0.022, 0.022, n=8, rings=2))
    return P


def wheel_parts():
    """visual meshes in the wheel frame (axis y): tyre, rim, hub cap"""
    tyre = torus_y((0, 0, 0), 0.255, 0.045, 0.034, n=24, m=8)
    rim = cylinder_y((0, 0, 0), 0.212, 0.05, n=24)
    cap = cylinder_y((0, 0, 0), 0.065, 0.072, n=12)
    return {"tyre": tyre, "hub": rim, "chassis": cap}


def legs_parts():
    """rider_legs link (frame at the hip joint): pelvis, thighs, shanks, feet"""
    body, joint = Mesh(), Mesh()
    body.extend(ellipsoid((-0.02, 0, 0.03), (0.11, 0.165, 0.10), n_lon=12, n_lat=7))
    body.extend(ellipsoid((-0.06, 0, -0.035), (0.10, 0.16, 0.06), n_lon=12, n_lat=6))
    for s in (1, -1):
        hip = np.array([0.0, s * HIP_W, 0.0])
        knee = KNEE * [1, s, 1]
        ankle = ANKLE * [1, s, 1]
        body.extend(capsule(hip, knee, 0.085, 0.056, n=10, rings=3))
        joint.extend(ellipsoid(knee, (0.058, 0.058, 0.058), n_lon=12, n_lat=8))
        body.extend(capsule(knee, ankle, 0.052, 0.034, n=10, rings=3))
        joint.extend(ellipsoid(ankle, (0.036, 0.036, 0.036), n_lon=8, n_lat=5))
        body.extend(ellipsoid(ankle + [0.075, 0, -0.035], (0.125, 0.047, 0.036), n_lon=10, n_lat=6))
    return {"mannequin": body, "joint": joint}


def upper_parts():
    """rider link (frame at the hip joint): abdomen, chest, neck, head, arms, hands"""
    body, joint = Mesh(), Mesh()
    body.extend(ellipsoid((-0.01, 0, 0.17), (0.10, 0.15, 0.13), n_lon=12, n_lat=7))
    body.extend(ellipsoid((-0.02, 0, 0.36), (0.11, 0.175, 0.15), n_lon=12, n_lat=7))
    body.extend(ellipsoid((-0.03, 0, 0.465), (0.085, 0.165, 0.055), n_lon=12, n_lat=6))      # shoulder girdle
    body.extend(capsule((-0.03, 0, 0.50), (-0.005, 0, 0.61), 0.056, 0.05, n=10, rings=2))
    body.extend(ellipsoid(HEAD_C, HEAD_SEMI, n_lon=12, n_lat=8))
    for s in (1, -1):
        sh, el, wr = SHOULDER * [1, s, 1], ELBOW * [1, s, 1], WRIST * [1, s, 1]
        joint.extend(ellipsoid(sh, (0.055, 0.055, 0.055), n_lon=12, n_lat=8))
        body.extend(capsule(sh, el, 0.045, 0.037, n=10, rings=3))
        joint.extend(ellipsoid(el, (0.039, 0.039, 0.039), n_lon=10, n_lat=6))
        body.extend(capsule(el, wr, 0.036, 0.028, n=10, rings=3))
        joint.extend(ellipsoid(wr, (0.029, 0.029, 0.029), n_lon=8, n_lat=5))
        d = (wr - el) / np.linalg.norm(wr - el)
        R = frame_from_axis(d)[:, [1, 2, 0]]             # local x of the ellipsoid along the forearm
        body.extend(ellipsoid(wr + d * 0.068 + [0, 0, -0.012], (0.072, 0.036, 0.024), n_lon=10, n_lat=6, R=R))
    return {"mannequin": body, "joint": joint}


# ------------------------------------------------------------------------------------------------ inertial data
def segment_masses():
    """point-ish segments (fraction, CoM relative to the hip, own inertia diag per unit fraction) for both links"""
    legs, upper = [], []
    rod = lambda L, r: np.array([0.25 * r * r + L * L / 12, 0.25 * r * r + L * L / 12, 0.5 * r * r])  # about own axes
    def add(lst, frac, com, I_local=np.zeros(3), R=np.eye(3)):
        lst.append((frac, np.asarray(com, float), R @ np.diag(I_local) @ R.T))
    add(legs, FRAC["pelvis"], (-0.03, 0, 0.0), 0.2 * np.array([0.165**2 + 0.10**2, 0.11**2 + 0.10**2, 0.11**2 + 0.165**2]))
    for s in (1, -1):
        hip, knee, ankle = np.array([0, s * HIP_W, 0]), KNEE * [1, s, 1], ANKLE * [1, s, 1]
        add(legs, FRAC["thigh"], hip + 0.433 * (knee - hip), rod(L_THIGH, 0.07), frame_from_axis(knee - hip))
        add(legs, FRAC["shank"], knee + 0.433 * (ankle - knee), rod(L_SHANK, 0.045), frame_from_axis(ankle - knee))
        add(legs, FRAC["foot"], ankle + [0.075, 0, -0.035], 0.2 * np.array([0.047**2 + 0.036**2, 0.125**2 + 0.036**2, 0.125**2 + 0.047**2]))
    add(upper, FRAC["head"], HEAD_C, 0.2 * np.array([0.08**2 + 0.115**2, 0.10**2 + 0.115**2, 0.10**2 + 0.08**2]))
    add(upper, FRAC["thorax"], (-0.02, 0, 0.36), 0.2 * np.array([0.175**2 + 0.15**2, 0.11**2 + 0.15**2, 0.11**2 + 0.175**2]))
    add(upper, FRAC["abdomen"], (-0.01, 0, 0.17), 0.2 * np.array([0.15**2 + 0.13**2, 0.10**2 + 0.13**2, 0.10**2 + 0.15**2]))
    for s in (1, -1):
        sh, el, wr = SHOULDER * [1, s, 1], ELBOW * [1, s, 1], WRIST * [1, s, 1]
        add(upper, FRAC["upper_arm"], sh + 0.436 * (el - sh), rod(L_UPARM, 0.04), frame_from_axis(el - sh))
        add(upper, FRAC["forearm"], el + 0.430 * (wr - el), rod(L_FOREARM, 0.032), frame_from_axis(wr - el))
        add(upper, FRAC["hand"], wr + 0.07 * (wr - el) / np.linalg.norm(wr - el), np.full(3, 0.0004))
    return legs, upper


def combine(segs):
    """total fraction, CoM, inertia about the CoM per kg of this link"""
    f = sum(s[0] for s in segs)
    com = sum(s[0] * s[1] for s in segs) / f
    I = np.zeros((3, 3))
    for fr, c, Il in segs:
        d = c - com
        I += fr * (Il + (d @ d) * np.eye(3) - np.outer(d, d))
    return f, com, I / f


# ------------------------------------------------------------------------------------------------ URDF
def urdf(mesh_dir, legs, upper, simple=False):
    fl, cl, Il = legs
    fu, cu, Iu = upper
    hip_b = to_b(HIP_A)
    mat = "\n".join(f'  <material name="{k}"><color rgba="{" ".join(f"{x:.3f}" for x in v)}"/></material>' for k, v in COL.items())
    def mesh_vis(name, colour):
        return f'    <visual><geometry><mesh filename="{mesh_dir}/{name}.obj"/></geometry><material name="{colour}"/></visual>'
    def inertial(m, com, I):
        return (f'    <inertial>\n      <origin xyz="{com[0]:.4f} {com[1]:.4f} {com[2]:.4f}" rpy="0 0 0"/>\n      <mass value="{m}"/>\n'
                f'      <inertia ixx="{I[0,0]:.5f}" iyy="{I[1,1]:.5f}" izz="{I[2,2]:.5f}" ixy="{I[0,1]:.5f}" ixz="{I[0,2]:.5f}" iyz="{I[1,2]:.5f}"/>\n    </inertial>')
    if simple:
        base_vis = simple_chair_visuals()
        legs_vis, up_vis, wheel_vis = simple_legs_visuals(), simple_upper_visuals(), simple_wheel_visuals()
    else:
        base_vis = "\n".join(mesh_vis(f"chair_{c}", c) for c in ("chassis", "accent", "cushion", "metal", "rubber"))
        legs_vis = "\n".join(mesh_vis(f"rider_legs_{c}", c) for c in ("mannequin", "joint"))
        up_vis = "\n".join(mesh_vis(f"rider_upper_{c}", c) for c in ("mannequin", "joint"))
        wheel_vis = "\n".join(mesh_vis(f"wheel_{c}", c) for c in ("tyre", "hub", "chassis"))
    chassis_col = to_b([0.0, 0, 0.085])
    wheel = lambda side, y: f"""
  <link name="{side}_wheel">
{wheel_vis}
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry><cylinder radius="0.3" length="0.04"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="5.0"/>
      <inertia ixx="0.113" iyy="0.225" izz="0.113" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="{side}_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="{side}_wheel"/>
    <origin xyz="0 {y} -0.305" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.0" friction="0.0"/>
    <limit effort="1.0" velocity="50.0"/>
  </joint>"""
    return f"""<?xml version="1.0"?>
<!--
  Self-balancing chair shaped after the Genny Zero, with a seated mannequin (rider task).
  Generated by build_rider_model.py ({'primitive shapes only' if simple else 'low-poly meshes in ' + mesh_dir + '/'}).
  Frame convention: +x forward, +y left, +z up. base_link origin 0.305 m above the wheel axle (as in tw_balance.urdf).

  Physics of base_link and the wheels is IDENTICAL to tw_balance.urdf (55 kg, 2 x 5 kg wheels, r = 0.3 m, track 0.61 m);
  only the base_link collision box changed shape (it never touches flat ground before the tilt termination).
  Rider (1.75 m mannequin, mass fractions after Dempster / Winter 2009):
    rider_legs : pelvis + legs, {fl*100:.1f} % of the rider mass, fixed to the chair (fixed joint, keep merge_fixed_joints=False)
    rider      : head + trunk + arms, {fu*100:.1f} % of the rider mass, on the hip joint rider_joint (axis y, + = lean forward)
  Both rider links weigh 1 kg here; the rider_mass event adds the fraction of the rider mass and scales the inertia.
  Seat top 0.64 m above the floor, hip joints {HIP_A[2]-SEAT_TOP_A:.2f} m above the seat and {-HIP_A[0]:.2f} m behind the axle.
-->
<robot name="balance_bot_rider">

{mat}

  <link name="base_link">
{base_vis}
    <collision>
      <origin xyz="{chassis_col[0]:.3f} 0 {chassis_col[2]:.3f}" rpy="0 0 0"/>
      <geometry><box size="0.46 0.44 0.51"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="55.0"/>
      <inertia ixx="3.506" iyy="3.506" izz="1.856" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
{wheel('left', 0.305)}
{wheel('right', -0.305)}

  <!-- pelvis and legs: fixed to the chair. Link frame = hip joint centre. -->
  <link name="rider_legs">
{legs_vis}
{inertial(1.0, cl, Il)}
  </link>

  <joint name="rider_legs_joint" type="fixed">
    <parent link="base_link"/>
    <child link="rider_legs"/>
    <origin xyz="{hip_b[0]:.3f} 0 {hip_b[2]:.3f}" rpy="0 0 0"/>
  </joint>

  <!-- head, trunk and arms: lean about the hip joint. Link frame = hip joint centre. -->
  <link name="rider">
{up_vis}
{inertial(1.0, cu, Iu)}
  </link>

  <joint name="rider_joint" type="revolute">
    <parent link="base_link"/>
    <child link="rider"/>
    <origin xyz="{hip_b[0]:.3f} 0 {hip_b[2]:.3f}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.0" friction="0.0"/>
    <limit lower="-0.40" upper="0.40" effort="1000.0" velocity="10.0"/>
  </joint>

</robot>
"""


# primitive-only fallback (boxes, cylinders, spheres) with the same proportions
def _box(c, s, col, rpy=(0, 0, 0)):
    return (f'    <visual><origin xyz="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}" rpy="{rpy[0]:.3f} {rpy[1]:.3f} {rpy[2]:.3f}"/>'
            f'<geometry><box size="{s[0]:.3f} {s[1]:.3f} {s[2]:.3f}"/></geometry><material name="{col}"/></visual>')


def _sph(c, r, col):
    return f'    <visual><origin xyz="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}"/><geometry><sphere radius="{r:.3f}"/></geometry><material name="{col}"/></visual>'


def _cyl(p0, p1, r, col):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    c = (p0 + p1) / 2
    pitch = math.atan2(d[0], d[2])                       # rotate z axis towards d (in x-z plane, then y)
    roll = -math.atan2(d[1], math.hypot(d[0], d[2]))
    return (f'    <visual><origin xyz="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}" rpy="{roll:.3f} {pitch:.3f} 0"/>'
            f'<geometry><cylinder radius="{r:.3f}" length="{L:.3f}"/></geometry><material name="{col}"/></visual>')


def _limb(p0, p1, r0, r1, col, jcol=None):
    s = [_cyl(p0, p1, (r0 + r1) / 2, col), _sph(p0, r0, jcol or col), _sph(p1, r1, jcol or col)]
    return "\n".join(s)


def simple_chair_visuals():
    v = [_box(to_b([0, 0, 0.02]), (0.40, 0.44, 0.38), "chassis"), _box(to_b([0, 0.228, 0.02]), (0.34, 0.02, 0.30), "accent"),
         _box(to_b([0, -0.228, 0.02]), (0.34, 0.02, 0.30), "accent"), _box(to_b([-0.01, 0, 0.235]), (0.22, 0.24, 0.08), "chassis"),
         _box(to_b([0, 0, 0.278]), (0.48, 0.48, 0.045), "chassis"), _box(to_b([0, 0, SEAT_TOP_A - 0.028]), (0.44, 0.46, 0.056), "cushion"),
         _box(to_b([-0.255, 0, SEAT_TOP_A + 0.30]), (0.05, 0.40, 0.44), "cushion", (0, -0.105, 0))]
    top = np.array([GRIP_A[0] - 0.01, 0.0, GRIP_A[2] - 0.035])
    v.append(_cyl(to_b([0.17, 0, 0.12]), to_b(top), 0.021, "metal"))
    v.append(_box(to_b(top + [0.005, 0, 0.035]), (0.07, 0.13, 0.05), "chassis"))
    v.append(_cyl(to_b([GRIP_A[0], -0.15, GRIP_A[2]]), to_b([GRIP_A[0], 0.15, GRIP_A[2]]), 0.016, "metal"))
    for s in (1, -1):
        v.append(_cyl(to_b([GRIP_A[0], s * 0.15, GRIP_A[2]]), to_b([GRIP_A[0], s * 0.265, GRIP_A[2]]), 0.021, "rubber"))
        v.append(_cyl(to_b([-0.10, s * 0.25, SEAT_TOP_A + 0.06]), to_b([0.13, s * 0.25, SEAT_TOP_A + 0.06]), 0.012, "metal"))
    sole_z = (HIP_A + ANKLE)[2] - ANKLE_H
    v.append(_box(to_b([0.42, 0, sole_z - 0.016]), (0.24, 0.34, 0.032), "chassis"))
    v.append(_cyl(to_b([0.17, 0, -0.04]), to_b([0.33, 0, sole_z - 0.02]), 0.022, "metal"))
    return "\n".join(v)


def simple_wheel_visuals():
    return ('    <visual><origin xyz="0 0 0" rpy="1.5708 0 0"/><geometry><cylinder radius="0.3" length="0.07"/></geometry><material name="tyre"/></visual>\n'
            '    <visual><origin xyz="0 0 0" rpy="1.5708 0 0"/><geometry><cylinder radius="0.212" length="0.075"/></geometry><material name="hub"/></visual>')


def simple_legs_visuals():
    v = [_sph((-0.02, 0, 0.02), 0.12, "mannequin")]
    for s in (1, -1):
        hip, knee, ankle = np.array([0, s * HIP_W, 0]), KNEE * [1, s, 1], ANKLE * [1, s, 1]
        v.append(_limb(hip, knee, 0.08, 0.056, "mannequin", "joint"))
        v.append(_limb(knee, ankle, 0.05, 0.035, "mannequin", "joint"))
        v.append(_box(ankle + [0.075, 0, -0.035], (0.25, 0.09, 0.07), "mannequin"))
    return "\n".join(v)


def simple_upper_visuals():
    v = [_cyl((-0.015, 0, 0.10), (-0.02, 0, 0.47), 0.13, "mannequin"), _cyl((-0.03, 0, 0.50), (-0.005, 0, 0.615), 0.048, "mannequin"),
         _sph(HEAD_C, 0.105, "mannequin")]
    for s in (1, -1):
        sh, el, wr = SHOULDER * [1, s, 1], ELBOW * [1, s, 1], WRIST * [1, s, 1]
        v.append(_limb(sh, el, 0.05, 0.038, "mannequin", "joint"))
        v.append(_limb(el, wr, 0.036, 0.029, "mannequin", "joint"))
        d = (wr - el) / np.linalg.norm(wr - el)
        v.append(_sph(wr + d * 0.06, 0.04, "mannequin"))
    return "\n".join(v)


# ------------------------------------------------------------------------------------------------ main
def main(out):
    mesh_dir = "meshes"
    os.makedirs(os.path.join(out, mesh_dir), exist_ok=True)
    groups = {}
    for c, m in chair_parts().items():
        groups[f"chair_{c}"] = m
    for c, m in wheel_parts().items():
        groups[f"wheel_{c}"] = m
    for c, m in legs_parts().items():
        groups[f"rider_legs_{c}"] = m
    for c, m in upper_parts().items():
        groups[f"rider_upper_{c}"] = m
    ntri = 0
    for name, m in groups.items():
        m.write_obj(os.path.join(out, mesh_dir, f"{name}.obj"))
        ntri += len(m.tris)
    legs_segs, upper_segs = segment_masses()
    legs, upper = combine(legs_segs), combine(upper_segs)
    open(os.path.join(out, "tw_balance_rider.urdf"), "w").write(urdf(mesh_dir, legs, upper))
    open(os.path.join(out, "tw_balance_rider_simple.urdf"), "w").write(urdf(mesh_dir, legs, upper, simple=True))
    info = dict(
        stature_m=H, legs_fraction=round(legs[0], 4), upper_fraction=round(upper[0], 4),
        legs_com_from_hip=np.round(legs[1], 4).tolist(), upper_com_from_hip=np.round(upper[1], 4).tolist(),
        legs_inertia_per_kg=np.round(np.diag(legs[2]), 5).tolist(), upper_inertia_per_kg=np.round(np.diag(upper[2]), 5).tolist(),
        hip_in_axle_frame=HIP_A.tolist(), seat_top_above_floor=SEAT_TOP_A + 0.30, knee=np.round(KNEE, 3).tolist(),
        ankle=np.round(ANKLE, 3).tolist(), elbow=np.round(ELBOW, 3).tolist(), wrist=np.round(WRIST, 3).tolist(),
        sole_above_floor=round(float((HIP_A + ANKLE)[2] - ANKLE_H + 0.30), 3),
        head_top_above_floor=round(float(HIP_A[2] + HEAD_C[2] + HEAD_SEMI[2] + 0.30), 3),
        sitting_height=round(float(HIP_A[2] - SEAT_TOP_A + HEAD_C[2] + HEAD_SEMI[2]), 3), triangles=ntri,
    )
    json.dump(info, open(os.path.join(out, "rider_model_info.json"), "w"), indent=1)
    return groups, info


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    _, info = main(out)
    print(json.dumps(info, indent=1))

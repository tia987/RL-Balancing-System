"""TwoWheelBalance environment = Isaac Lab's ManagerBasedRLEnv + an optional text overlay on recorded videos.

Put this file in  TwoWheelBalance/src/TwoWheelBalance/tasks/twb/twb_env.py
and in tasks/twb/config/twb/__init__.py change the entry point to
    entry_point="TwoWheelBalance.tasks.twb.twb_env:TwbEnv",

Training is unchanged. The overlay is only drawn when the environment variable TWB_OVERLAY=1 is set,
e.g.  TWB_OVERLAY=1 HEADLESS=1 ENABLE_CAMERAS=1 isaaclab play ... --video
It shows, for the robot the camera follows: rider mass, commanded vs actual velocity, pitch, wheel torques
and terrain level. Set TWB_OVERLAY_ENV=<index> to choose another robot.

How it works: Isaac Lab 3.0 records videos with env.video_recorders (VideoRecorder.step() is called inside
env.step() after physics/resets). We wrap each recorder's frame grabber and draw on the frame right there,
so the numbers are from exactly the same moment as the picture.
"""
from __future__ import annotations

import math
import os

import numpy as np

from isaaclab.envs import ManagerBasedRLEnv


def _t(x):
    """Isaac Lab 3.0 data fields are proxy arrays with a .torch view."""
    return x.torch if hasattr(x, "torch") else x


class TwbEnv(ManagerBasedRLEnv):
    def step(self, action):
        if not getattr(self, "_twb_overlay_ready", False):
            self._twb_overlay_ready = True
            if os.environ.get("TWB_OVERLAY", "0") == "1":
                for rec in getattr(self, "video_recorders", []):
                    grab = rec._get_frame
                    rec._get_frame = lambda grab=grab: self._twb_overlay(grab())
        return super().step(action)

    # ------------------------------------------------------------------ overlay
    def _twb_env_index(self) -> int:
        if "TWB_OVERLAY_ENV" in os.environ:
            return int(os.environ["TWB_OVERLAY_ENV"]) % self.num_envs
        viz = getattr(getattr(self.cfg, "sim", None), "default_visualizer_cfg", None)
        idx = getattr(viz, "origin_env_index", None)
        if idx is None:
            idx = getattr(getattr(self.cfg, "viewer", None), "env_index", 0)
        return int(idx) % self.num_envs

    def _twb_lines(self) -> list[str]:
        i = self._twb_env_index()
        robot = self.scene["robot"]
        lines = [f"robot #{i}"]
        if hasattr(self, "rider_mass"):
            lines.append(f"rider      {self.rider_mass[i].item():6.1f} kg")
        cmd = self.command_manager.get_command("base_velocity")[i]
        v = _t(robot.data.root_lin_vel_b)[i]
        w = _t(robot.data.root_ang_vel_b)[i]
        lines.append(f"cmd    vx {cmd[0].item():+5.2f} m/s   wz {cmd[2].item():+5.2f} rad/s")
        lines.append(f"actual vx {v[0].item():+5.2f} m/s   wz {w[2].item():+5.2f} rad/s")
        g = _t(robot.data.projected_gravity_b)[i]
        pitch = math.degrees(math.atan2(g[0].item(), -g[2].item()))
        lines.append(f"pitch      {pitch:+5.1f} deg")
        tau = _t(robot.data.applied_torque)[i]
        lines.append(f"torque  L {tau[0].item():+5.0f}  R {tau[1].item():+5.0f} N·m")
        terrain = getattr(self.scene, "terrain", None)
        levels = getattr(terrain, "terrain_levels", None)
        if levels is not None:
            lines.append(f"terrain level {int(levels[i].item())}")
        return lines

    def _twb_overlay(self, frame):
        if frame is None:
            return frame
        try:
            return draw_text_box(frame, self._twb_lines())
        except Exception as e:  # never lose a video because of the overlay
            if not getattr(self, "_twb_overlay_warned", False):
                print(f"[TWB overlay] disabled: {e!r}")
                self._twb_overlay_warned = True
            return frame


def draw_text_box(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw white text on a semi-transparent dark box in the top-left corner of an RGB(A) uint8 frame."""
    from PIL import Image, ImageDraw, ImageFont

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    img = Image.fromarray(arr[..., :3]).convert("RGBA")
    size = max(14, img.height // 32)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    box = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(box)
    line_h = int(size * 1.3)
    width = max(draw.textlength(s, font=font) for s in lines)
    draw.rectangle([10, 10, 30 + width, 20 + line_h * len(lines)], fill=(0, 0, 0, 150))
    for k, s in enumerate(lines):
        draw.text((20, 15 + k * line_h), s, font=font, fill=(255, 255, 255, 255))
    out = np.array(Image.alpha_composite(img, box).convert("RGB"))
    return out if arr.shape[-1] == 3 else np.concatenate([out, arr[..., 3:]], axis=-1)

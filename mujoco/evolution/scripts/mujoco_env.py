"""MuJoCo environment for the quadruped, matching the CreatureEnv interface.

The GA only needs four things from an environment:
    reset()  -> observation
    step(a)  -> (observation, reward, done, fallen)
So this class provides exactly that on top of mujoco, so run_episode and the
Population in src/ can drive it unchanged.

Observation layout (26 numbers):
    8  hinge angles
    8  hinge velocities
    1  torso height
    4  torso quaternion (w, x, y, z) -- lets the brain know if it is tilting
    3  torso angular velocity      -- lets the brain damp a wobble
    2  torso linear velocity x, y  -- lets the brain feel how fast it goes
"""

from pathlib import Path

import numpy as np

import mujoco

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "smoke.xml"
FALLEN_HEIGHT = 0.25

class MujocoDogEnv:

    def __init__(self, max_steps=600, xml_path=None):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.steps = 0
        self.start_x = 0.0
        self.angle_idx = []
        self.vel_idx = []
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                self.angle_idx.append(int(self.model.jnt_qposadr[i]))
                self.vel_idx.append(int(self.model.jnt_dofadr[i]))

        self.obs_size = 2 * len(self.angle_idx) + 1 + 4 + 3 + 2
        self.act_size = int(self.model.nu)
        self.legs = ["frontleft","frontright","backleft","backright"]
        self.foot_id = {
            leg: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,leg + "shin")
            for leg in self.legs
        }
        self.step_log = []
        self._down = {leg: False for leg in self.legs}
        self._lifted_at = {leg: -1.0 for leg in self.legs}

    def reset(self, pose_noise=None):
        # mj_resetData puts qpos back to model.qpos0, which is the standing pose
        # baked into the XML, and zeroes all velocities.
        mujoco.mj_resetData(self.model, self.data)

        if pose_noise:
            for a in self.angle_idx:
                self.data.qpos[a] += np.random.normal(0, pose_noise)
            for v in self.vel_idx:
                self.data.qvel[v] += np.random.normal(0, pose_noise)

        mujoco.mj_forward(self.model, self.data)  # send the new one out
        self.steps = 0
        self.start_x = float(self.data.qpos[0])
        self.step_log = []
        self._down = {leg: False for leg in self.legs}
        self._lifted_at = {leg: -1.0 for leg in self.legs}
        return self._observation()  # return the original


    def _observation(self):
        values = []
        for a in self.angle_idx:  # the current angles which we will take
            values.append(self.data.qpos[a])
        for v in self.vel_idx:  # angular velocity part same
            values.append(self.data.qvel[v])
        values.append(self.data.qpos[2])  # torso height

        # free joint layout: qpos[3:7] is the orientation quaternion,
        # qvel[0:3] is linear velocity, qvel[3:6] is angular velocity
        values.extend(self.data.qpos[3:7])  # quaternion
        values.extend(self.data.qvel[3:6])  # angular velocity
        values.extend(self.data.qvel[0:2])  # forward / sideways velocity
        return np.array(values, dtype=np.float32)

    # main part of the function
    def step(self, action):  # step function of the GA
        action = np.clip(action, -1.0, 1.0)  # clip the back and forth part
        self.data.ctrl[:] = action  # feeds the action into the body
        mujoco.mj_step(self.model, self.data)  # time step per thingy
        self.steps += 1
        now = float(self.data.time)
        for leg in self.legs:
            foot_z = self.data.xpos[self.foot_id[leg]][2] - 0.3
            down = foot_z < 0.03
            # only act on a change of state, and ignore contact chatter
            if down == self._down[leg] or now - self._lifted_at[leg] < 0.02:
                continue
            if down:  # touchdown: open a row, stance fills in at liftoff
                self.step_log.append({
                    "foot":leg,
                    "t_down":now,
                    "x":float(self.data.qpos[0]),
                    "stance":None,
                })
            else:  # liftoff: close this foot's newest open row, not step_log[-1]
                for i in range(len(self.step_log) - 1, -1, -1):
                    if self.step_log[i]["foot"] == leg and self.step_log[i]["stance"] is None:
                        self.step_log[i]["stance"] = now - self.step_log[i]["t_down"]
                        break
            self._down[leg] = down
            self._lifted_at[leg] = now
        reward = (self.data.qpos[0] - self.start_x) * 2.0
        fallen = bool(self.data.qpos[2] < FALLEN_HEIGHT)
        done = bool(fallen or self.steps >= self.max_steps)
        return self._observation(), reward, done, fallen

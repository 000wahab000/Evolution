import math

import numpy as np
import pymunk

from src.terrain import add_terrain, legacy_course

CPG_CLOCK = tuple(
    (math.sin(2.0 * math.pi * i / 30.0), math.cos(2.0 * math.pi * i / 30.0))
    for i in range(30)
)
CPG_MODES = ("cpg", "zeros", "noise", "shuffled")


def run_episode(env, brain, max_steps=600, pose_noise=None, on_step=None):
    """The one rollout loop: returns (training return incl. -100 on fall, steps, fallen).
    on_step(env, action) is called after every step, for recording."""
    state = env.reset(pose_noise=pose_noise)
    total_reward, steps, done, fallen = 0.0, 0, False, False
    while not done and steps < max_steps:
        action = brain.get_action(state)
        state, reward, done, fallen = env.step(action)
        total_reward += reward
        steps += 1
        if on_step:
            on_step(env, action)
    if fallen:
        total_reward -= 100.0
    return total_reward, steps, fallen


class CreatureEnv:
    def __init__(self, fitness_mode="baseline", terrain=None, cpg_mode="cpg", cpg_rng=None, substeps=1,
                 cpg_perm=None, knees=False, fall_rule="touch"):
        """terrain: TerrainSpec (default: the legacy course). cpg_mode replaces state[13:15]:
        cpg (clock), zeros, noise (U[-1,1] from cpg_rng), shuffled (the 30-step table read
        through cpg_perm, a fixed permutation chosen once per run). substeps>1 splits each step
        into that many physics steps of 3/60/substeps; all step counters still count brain steps.
        knees: each leg is thigh + shin with a one-way knee (8 motors, state grows by 8;
        knee action -1 bends the knee (foot moves forward of the knee), +1 straightens it). knees="spot": Spot-like knees (limit
        14-150 deg, bent 30 deg at reset, hips +-90 deg). fall_rule: "touch" (torso on the ground
        ends the episode) or "penalty" (reward -2 per touching step, no fall for touching)."""
        if knees not in (False, True, "spot"):
            raise ValueError("knees must be False, True or 'spot'")
        if fall_rule not in ("touch", "penalty"):
            raise ValueError("fall_rule must be 'touch' or 'penalty'")
        if cpg_mode not in CPG_MODES:
            raise ValueError(f"cpg_mode must be one of {CPG_MODES}")
        if cpg_mode == "noise" and cpg_rng is None:
            raise ValueError("cpg_mode='noise' needs cpg_rng")
        if cpg_mode == "shuffled" and (cpg_perm is None or sorted(cpg_perm) != list(range(30))):
            raise ValueError("cpg_mode='shuffled' needs cpg_perm, a permutation of range(30)")
        self.fitness_mode = fitness_mode
        self.terrain = terrain if terrain is not None else legacy_course()
        self.cpg_mode = cpg_mode
        self.cpg_rng = cpg_rng
        self.cpg_perm = list(cpg_perm) if cpg_mode == "shuffled" else None
        self.substeps = int(substeps)
        self.knees = bool(knees)
        self.spot = knees == "spot"
        self.fall_rule = fall_rule
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)
        self.create_ground()
        self.create_creature()
        self.steps = 0

    def create_ground(self):
        self.ground_shapes = add_terrain(self.space, self.terrain)

    def create_creature(self):
        mass = 10
        width = 60
        height = 20
        moment = pymunk.moment_for_box(mass, (width, height))
        self.torso = pymunk.Body(mass, moment)
        self.torso.position = (400,450)
        self.torso_shape = pymunk.Poly.create_box(self.torso, (width, height))
        self.torso_shape.friction = 0.5
        self.torso_shape.filter = pymunk.ShapeFilter(group=1)
        self.space.add(self.torso, self.torso_shape)
        self.legs = []
        self.leg_shapes = []
        self.joints = []
        self.motors = []
        leg_mass = 2
        leg_width = 10
        leg_length = 40
        leg_moment = pymunk.moment_for_box(leg_mass, (leg_width, leg_length))
        self.anchors = [
            (-22, height / 2),
            (-18, height / 2),
            (18, height / 2),
            (22, height / 2)]
        self.leg_length = leg_length
        if self.knees:
            self.create_kneed_legs(leg_mass=1, leg_width=10, seg=20)
            return
        for anchor in self.anchors:
            leg = pymunk.Body(leg_mass, leg_moment)
            leg.position = (
                self.torso.position.x + anchor[0],
                self.torso.position.y + anchor[1] + leg_length / 2,
            )
            leg_shape = pymunk.Poly.create_box(leg, (leg_width, leg_length))
            leg_shape.friction = 1.2
            leg_shape.filter = pymunk.ShapeFilter(group=1)
            self.space.add(leg, leg_shape)
            self.legs.append(leg)
            self.leg_shapes.append(leg_shape)
            pivot = pymunk.PivotJoint(self.torso, leg, anchor, (0, -leg_length / 2))
            limit = pymunk.RotaryLimitJoint(self.torso, leg, -np.pi / 3, np.pi / 3)
            motor = pymunk.SimpleMotor(self.torso, leg, 0)
            motor.max_force = 150000
            self.space.add(pivot, limit, motor)
            self.joints.append(pivot)
            self.motors.append(motor)

    def create_kneed_legs(self, leg_mass, leg_width, seg):
        self.seg_length = seg
        self.knee_range = (-math.radians(150), -math.radians(14)) if self.spot else (-np.pi / 2, 0.0)
        self.shins = []
        self.shin_shapes = []
        knee_motors = []
        moment = pymunk.moment_for_box(leg_mass, (leg_width, seg))
        for anchor in self.anchors:
            bodies = []
            for k in range(2):
                b = pymunk.Body(leg_mass, moment)
                b.position = (
                    self.torso.position.x + anchor[0],
                    self.torso.position.y + anchor[1] + seg / 2 + k * seg,
                )
                shape = pymunk.Poly.create_box(b, (leg_width, seg))
                shape.friction = 1.2
                shape.filter = pymunk.ShapeFilter(group=1)
                self.space.add(b, shape)
                bodies.append(b)
                (self.leg_shapes if k == 0 else self.shin_shapes).append(shape)
            thigh, shin = bodies
            self.legs.append(thigh)
            self.shins.append(shin)
            pivot = pymunk.PivotJoint(self.torso, thigh, anchor, (0, -seg / 2))
            hip_range = np.pi / 2 if self.spot else np.pi / 3
            limit = pymunk.RotaryLimitJoint(self.torso, thigh, -hip_range, hip_range)
            motor = pymunk.SimpleMotor(self.torso, thigh, 0)
            motor.max_force = 150000
            knee = pymunk.PivotJoint(thigh, shin, (0, seg / 2), (0, -seg / 2))
            knee_limit = pymunk.RotaryLimitJoint(thigh, shin, *self.knee_range)
            knee_motor = pymunk.SimpleMotor(thigh, shin, 0)
            knee_motor.max_force = 75000
            self.space.add(pivot, limit, motor, knee, knee_limit, knee_motor)
            self.joints.append(pivot)
            self.motors.append(motor)
            knee_motors.append(knee_motor)
        self.motors.extend(knee_motors)

    def clamp_knees(self):
        """The limit joint only reacts after a knee is already past its limit and the motor
        overpowers it (116-128 deg seen on a 90 deg limit). After each physics step, rotate any
        out-of-range shin back about its knee pivot and drop its outward relative spin."""
        lo, hi = self.knee_range
        for thigh, shin in zip(self.legs, self.shins):
            rel = shin.angle - thigh.angle
            if lo <= rel <= hi:
                continue
            d = (lo if rel < lo else hi) - rel
            knee = thigh.local_to_world((0, self.seg_length / 2))
            c, s_ = math.cos(d), math.sin(d)
            off = shin.position - knee
            shin.position = (knee.x + c * off.x - s_ * off.y, knee.y + s_ * off.x + c * off.y)
            shin.angle += d
            if (shin.angular_velocity - thigh.angular_velocity) * d < 0:
                shin.angular_velocity = thigh.angular_velocity

    def reset(self, pose_noise=None):
        """pose_noise: optional 4 leg angles (rad) applied by rotating each leg about its pivot."""
        self.torso.position = (400, 450)
        self.torso.angle = 0.0
        self.torso.velocity = (0.0, 0.0)
        self.torso.angular_velocity = 0.0
        for i, (leg, anchor) in enumerate(zip(self.legs, self.anchors)):
            leg.position = (
                400 + anchor[0],
                450 + anchor[1] + self.leg_length / 2,
            )
            leg.angle = 0.0
            if pose_noise is not None:
                th = float(pose_noise[i])
                h = self.leg_length / 2
                leg.position = (400 + anchor[0] - h * math.sin(th), 450 + anchor[1] + h * math.cos(th))
                leg.angle = th
            leg.velocity = (0.0, 0.0)
            leg.angular_velocity = 0.0
        if self.knees:
            h = self.seg_length / 2
            knee = -math.radians(30) if self.spot else 0.0
            thigh_tilt = -knee / 2  # spot: foot ends up roughly under the hip
            torso_y = 450.0
            if self.spot:  # feet touch the ground: vertical leg height = 20cos(thigh) + 20cos(thigh+knee)
                torso_y = 500.0 - 10.0 - 2 * h * (math.cos(thigh_tilt) + math.cos(thigh_tilt + knee)) / 2
                self.torso.position = (400, torso_y)
            for i, (leg, shin, anchor) in enumerate(zip(self.legs, self.shins, self.anchors)):
                th = thigh_tilt + (float(pose_noise[i]) if pose_noise is not None else 0.0)
                sh = th + knee
                hip = (400 + anchor[0], torso_y + anchor[1])
                leg.position = (hip[0] - h * math.sin(th), hip[1] + h * math.cos(th))
                kx, ky = hip[0] - 2 * h * math.sin(th), hip[1] + 2 * h * math.cos(th)
                shin.position = (kx - h * math.sin(sh), ky + h * math.cos(sh))
                leg.angle, shin.angle = th, sh
                for b in (leg, shin):
                    b.velocity = (0.0, 0.0)
                    b.angular_velocity = 0.0
        for motor in self.motors:
            motor.rate = 0.0
        self.steps = 0
        self.still_steps = 0
        return self.get_state()

    def step(self, actions):
        for i, (motor, action) in enumerate(zip(self.motors, actions)):
            motor.rate = float(action) * (-10.0 if self.knees and i >= 4 else 10.0)

        old_x = self.torso.position.x
        for _ in range(self.substeps):
            self.space.step(3 / 60.0 / self.substeps)
            if self.knees:
                self.clamp_knees()
        self.steps += 1
        state = self.get_state()
        displacement = self.torso.position.x - old_x
        reward = displacement * 2
        # leg_motion = sum(abs(leg.angular_velocity) for leg in self.legs)
        # reward += leg_motion * 0.0005 #(dont remove the bum ahh ## please )
        torso_angle = self.torso.angle
        torso_bottom = self.torso.position.y + 30.0 * abs(math.sin(torso_angle)) + 10.0 * abs(math.cos(torso_angle))
        if self.terrain.legacy:
            ground, deep = 495.0, 520.0
        else:  # ground height under the torso; off the course counts as falling
            top = self.terrain.ground_top(self.torso.position.x)
            ground = top if top is not None else -math.inf
            deep = ground + 25.0
        body_touched_ground = torso_bottom >= ground
        if self.fitness_mode in ["tilt_penalized", "all_penalties"]:
            reward -= 20 * abs(torso_angle) / 60.0
        if self.fitness_mode in ["energy_penalized", "all_penalties"]:
            energy_cost = sum(abs(float(a)) for a in actions)
            reward -= energy_cost * 0.05 / 60.0
        if self.fitness_mode in ["ground_termination", "touch_ground"] and body_touched_ground:
            reward -= 100.0
        if abs(self.torso.velocity.x) < 2.0 and self.steps > 45:
            self.still_steps += 1
            reward -= 0.1
        else:
            self.still_steps = 0
        if self.fall_rule == "penalty":
            if body_touched_ground:
                reward -= 2.0
            fallen = abs(torso_angle) > 1.2 or self.torso.position.y > deep
        elif self.fitness_mode in ["ground_termination", "touch_ground"]:
            fallen = body_touched_ground or abs(torso_angle) > 1.2 or self.torso.position.y > deep
        else:
            fallen = abs(torso_angle) > 1.2 or self.torso.position.y > deep or body_touched_ground
        stuck = self.still_steps > 60
        done = self.steps > 600 or fallen or stuck
        return state, reward, done, fallen

    def get_state(self):
        torso_angle = self.torso.angle
        if self.cpg_mode == "cpg":
            cpg_sin, cpg_cos = CPG_CLOCK[self.steps % 30]
        elif self.cpg_mode == "zeros":
            cpg_sin, cpg_cos = 0.0, 0.0
        elif self.cpg_mode == "noise":
            cpg_sin, cpg_cos = self.cpg_rng.uniform(-1.0, 1.0, 2)
        else:
            cpg_sin, cpg_cos = CPG_CLOCK[self.cpg_perm[self.steps % 30]]
        legs = self.legs

        values = [
            torso_angle,
            self.torso.angular_velocity * 0.1,
            self.torso.velocity.x * 0.01,
            self.torso.velocity.y * 0.01,
            (self.torso.position.y - 450.0) * 0.02,
            legs[0].angle - torso_angle,
            legs[0].angular_velocity * 0.1,
            legs[1].angle - torso_angle,
            legs[1].angular_velocity * 0.1,
            legs[2].angle - torso_angle,
            legs[2].angular_velocity * 0.1,
            legs[3].angle - torso_angle,
            legs[3].angular_velocity * 0.1,
            cpg_sin,
            cpg_cos,
        ]
        if self.knees:
            for leg, shin in zip(legs, self.shins):
                values += [shin.angle - leg.angle, shin.angular_velocity * 0.1]
        return np.array(values, dtype=np.float32)

    def get_render_state(self):
        """Returns coordinate dictionary for lightweight browser HTML5 canvas rendering."""
        return {
            "torso": {
                "x": float(self.torso.position.x),
                "y": float(self.torso.position.y),
                "angle": float(self.torso.angle),
                "width": 60,
                "height": 20,
            },
            "legs": [
                {
                    "x": float(leg.position.x),
                    "y": float(leg.position.y),
                    "angle": float(leg.angle),
                    "width": 10,
                    "height": 40,
                }
                for leg in self.legs
            ],
            "ground_y": 500,
            "steps": int(self.steps),
            "distance": float(self.torso.position.x - 400.0),
            "velocity_x": float(self.torso.velocity.x),
            "angle": float(self.torso.angle),
        }

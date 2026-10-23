import math

import numpy as np
import pymunk

try:
    from noise import pnoise1
except ImportError:
    from perlin_noise import PerlinNoise
    _perlin = PerlinNoise(octaves=4, seed=42)
    def pnoise1(x, octaves=4):
        return _perlin(x)

CPG_CLOCK = tuple(
    (math.sin(2.0 * math.pi * i / 30.0), math.cos(2.0 * math.pi * i / 30.0))
    for i in range(30)
)
class CreatureEnv:
    def __init__(self, fitness_mode="baseline"):
        self.fitness_mode = fitness_mode
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)
        self.create_ground()
        self.create_creature()
        self.steps = 0
        self.cpg_phase = 0.0

    def create_ground(self):
        body = self.space.static_body
        self.ground_shape = pymunk.Segment(body, (-10000, 500), (600, 500), 5.0)
        self.ground_shape.friction = 1.2
        self.space.add(self.ground_shape)
        self.terrain_changing()

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

    def reset(self):
        self.torso.position = (400, 450)
        self.torso.angle = 0.0
        self.torso.velocity = (0.0, 0.0)
        self.torso.angular_velocity = 0.0
        for leg, anchor in zip(self.legs, self.anchors):
            leg.position = (
                400 + anchor[0],
                450 + anchor[1] + self.leg_length / 2,
            )
            leg.angle = 0.0
            leg.velocity = (0.0, 0.0)
            leg.angular_velocity = 0.0
        for motor in self.motors:
            motor.rate = 0.0
        self.steps = 0
        self.still_steps = 0
        self.cpg_phase = 0.0
        return self.get_state()

    def step(self, actions):
        motor_actions = actions[:4]
        freq_mod = float(actions[4]) if len(actions) > 4 else 0.0
        for motor, action in zip(self.motors, motor_actions):
            motor.rate = float(action) * 10.0
        self.cpg_phase += 1.0 + freq_mod * 0.5

        old_x = self.torso.position.x
        self.space.step(3 / 60.0)
        self.steps += 1
        state = self.get_state()
        displacement = self.torso.position.x - old_x
        reward = displacement * 2
        # leg_motion = sum(abs(leg.angular_velocity) for leg in self.legs)
        # reward += leg_motion * 0.0005 #(dont remove the bum ahh ## please )
        torso_angle = self.torso.angle
        torso_bottom = self.torso.position.y + 30.0 * abs(math.sin(torso_angle)) + 10.0 * abs(math.cos(torso_angle))
        body_touched_ground = torso_bottom >= 495.0
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
        if self.fitness_mode in ["ground_termination", "touch_ground"]:
            fallen = body_touched_ground or abs(torso_angle) > 1.2 or self.torso.position.y > 520
        else:
            fallen = abs(torso_angle) > 1.2 or self.torso.position.y > 520 or body_touched_ground
        reached_goal = self.torso.position.x >= 4000
        if reached_goal:
            done = True

        stuck = self.still_steps > 60
        done = reached_goal or fallen or stuck or (self.steps > 2500)
        return state, reward, done, fallen


    def query_ground_height(self, world_x):
        hits = self.space.segment_query(
            (world_x, 300), (world_x, 600), 1.0, pymunk.ShapeFilter()
        )
        static_hits = [h for h in hits if h.shape.body.body_type == pymunk.Body.STATIC]
        if static_hits:
            return min(h.point.y for h in static_hits)
        return 500.0

    def get_state(self):
        torso_angle = self.torso.angle
        phase_idx = int(self.cpg_phase) % 30
        cpg_sin, cpg_cos = CPG_CLOCK[phase_idx]
        legs = self.legs

        torso_x = self.torso.position.x
        look1 = (self.query_ground_height(torso_x + 50) - 500.0) * 0.02
        look2 = (self.query_ground_height(torso_x + 100) - 500.0) * 0.02
        look3 = (self.query_ground_height(torso_x + 150) - 500.0) * 0.02

        return np.array([
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
            look1,
            look2,
            look3,
        ], dtype=np.float32)

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
    def terrain_changing(self):
        self.terrain_bodies = []
        self.terrain_shapes = []
        sample_xs = range(600, 5001, 30)
        points = []
        for x in sample_xs:
            frequency = 0.0008 + (x / 5000.0) * 0.002
            amp = 15.0 + (x / 5000.0) * 55.0
            noise_val = pnoise1(x * frequency, octaves=4)
            y = 500.0 + noise_val * amp
            points.append((x, y))
        for i in range(len(points) - 1):
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            seg = pymunk.Segment(body, points[i], points[i + 1], 3)
            seg.friction = 1.5
            self.space.add(body, seg)
            self.terrain_bodies.append(body)
            self.terrain_shapes.append(seg)

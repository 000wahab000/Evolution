import numpy as np
import pymunk


class CreatureEnv:
    def __init__(self, fitness_mode="baseline"):
        self.fitness_mode = fitness_mode
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)  # PyGame Y points down, so positive gravity is down
        self.create_ground()
        self.create_creature()
        self.steps = 0

    def create_ground(self):
        body = self.space.static_body
        self.ground_shape = pymunk.Segment(body, (-10000, 500), (10000, 500), 5.0)  # Ground at Y=500
        self.ground_shape.friction = 1.0
        self.space.add(self.ground_shape)

    def create_creature(self):
        # Torso
        mass = 10
        width = 60
        height = 20
        moment = pymunk.moment_for_box(mass, (width, height))
        self.torso = pymunk.Body(mass, moment)
        self.torso.position = (
            400,
            450,
        )  # Torso center at Y=450; with 40px legs, feet touch ground at Y=500
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

        # Attach 4 legs (2 rear legs, 2 front legs) to bottom of torso
        self.anchors = [
            (-22, height / 2),  # Rear-outer
            (-18, height / 2),  # Rear-inner
            (18, height / 2),   # Front-inner
            (22, height / 2),   # Front-outer
        ]
        self.leg_length = leg_length

        for anchor in self.anchors:
            leg = pymunk.Body(leg_mass, leg_moment)
            # Center of leg is 20px below anchor, so foot rests at Y=500
            leg.position = (
                self.torso.position.x + anchor[0],
                self.torso.position.y + anchor[1] + leg_length / 2,
            )
            leg_shape = pymunk.Poly.create_box(leg, (leg_width, leg_length))
            leg_shape.friction = 1.2  # High ground traction
            leg_shape.filter = pymunk.ShapeFilter(group=1)  # Don't self-collide
            self.space.add(leg, leg_shape)
            self.legs.append(leg)
            self.leg_shapes.append(leg_shape)

            # Pivot Joint at top of leg (0, -leg_length / 2)
            pivot = pymunk.PivotJoint(self.torso, leg, anchor, (0, -leg_length / 2))
            # Rotary Limit Joint (-60 to 60 degrees relative to torso)
            limit = pymunk.RotaryLimitJoint(self.torso, leg, -np.pi / 3, np.pi / 3)
            # Motor to apply torque
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
        return self.get_state()

    def step(self, actions):
        # Actions are desired velocities for the motors
        for i, motor in enumerate(self.motors):
            motor.rate = float(actions[i]) * 10.0  # Scale action to velocity

        old_x = self.torso.position.x
        self.space.step(1 / 60.0)
        self.steps += 1

        state = self.get_state()

        # Base displacement reward
        displacement = self.torso.position.x - old_x
        reward = displacement

        # Continuous leg activity bonus (encourages active cycling over freezing)
        leg_motion = sum(abs(float(leg.angular_velocity)) for leg in self.legs)
        reward += leg_motion * 0.005

        # Check if torso / body touches the ground (ground surface is at ~495-500)
        torso_bottom = max(
            self.torso.position.y + lx * np.sin(self.torso.angle) + ly * np.cos(self.torso.angle)
            for lx, ly in [(-30, -10), (-30, 10), (30, -10), (30, 10)]
        )
        body_touched_ground = bool(torso_bottom >= 495.0)

        # Tilt penalty only in tilt-penalized modes (not in baseline or ground_termination)
        if self.fitness_mode in ["tilt_penalized", "all_penalties"]:
            reward -= 20 * abs(self.torso.angle) / 60.0

        if self.fitness_mode in ["energy_penalized", "all_penalties"]:
            energy_cost = sum(abs(float(a)) for a in actions)
            reward -= energy_cost * 0.05 / 60.0

        # Ground termination penalty when body touches the ground
        if self.fitness_mode in ["ground_termination", "touch_ground"] and body_touched_ground:
            reward -= 100.0

        # Standstill / stagnation tracking (terminate if freezing in place)
        if abs(self.torso.velocity.x) < 2.0 and self.steps > 45:
            self.still_steps += 1
            reward -= 0.1
        else:
            self.still_steps = 0

        # Fall / termination condition: touching the ground terminates the episode
        if self.fitness_mode in ["ground_termination", "touch_ground"]:
            fallen = body_touched_ground or abs(self.torso.angle) > 1.2 or self.torso.position.y > 520
        else:
            fallen = abs(self.torso.angle) > 1.2 or self.torso.position.y > 520 or body_touched_ground

        stuck = self.still_steps > 60
        done = self.steps > 600 or fallen or stuck

        return state, reward, done, fallen

    def get_state(self):
        state = []
        state.append(float(self.torso.angle))
        state.append(float(self.torso.angular_velocity) / 10.0)
        state.append(float(self.torso.velocity.x) / 100.0)
        state.append(float(self.torso.velocity.y) / 100.0)
        state.append((float(self.torso.position.y) - 450.0) / 50.0)

        for leg in self.legs:
            state.append(float(leg.angle - self.torso.angle))
            state.append(float(leg.angular_velocity) / 10.0)

        # Central Pattern Generator (CPG) internal rhythm: sin/cos phase clock (stride cycle ~0.5s)
        phase = 2.0 * np.pi * (self.steps % 30) / 30.0
        state.append(float(np.sin(phase)))
        state.append(float(np.cos(phase)))

        return np.array(state, dtype=np.float32)

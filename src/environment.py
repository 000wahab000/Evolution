import math
import numpy as np
import pymunk

# Precomputed 30-step rhythmic CPG phase clock lookup table
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

    def create_ground(self):
        body = self.space.static_body
        self.ground_shape = pymunk.Segment(body, (-10000, 500), (2000, 500), 5.0)  
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
        return self.get_state()

    def step(self, actions):
        for motor, action in zip(self.motors, actions):
            motor.rate = float(action) * 10.0

        old_x = self.torso.position.x
        self.space.step(3 / 60.0)
        self.steps += 1
        state = self.get_state()
        displacement = self.torso.position.x - old_x
        reward = displacement * 2
        # leg_motion = sum(abs(leg.angular_velocity) for leg in self.legs)
        # reward += leg_motion * 0.0005
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

        stuck = self.still_steps > 60
        done = self.steps > 600 or fallen or stuck

        return state, reward, done, fallen

    def get_state(self):
        torso_angle = self.torso.angle
        cpg_sin, cpg_cos = CPG_CLOCK[self.steps % 30]
        legs = self.legs

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
        #*ok so firstly the ground is NOT super flat but kinda
        # and then it will chanage like the more far we go the harder it is
        # the body spawns at 400 so we need to add the uneven grounds 
        # after the 600 and then we can do soo
        # no obstacles btw
       start_x = 600
       end_x = 1000
       step_size = 15
        
       for x in np.arange(start_x, end_x, step_size):
            progress = (x - 600) * 0.02
            hill_height = min(40.0 , 10.0 + progress )

            y1 = abs(500 - math.sin(x * 0.1) * hill_height)
            y2 = abs(500 - math.sin((x + step_size) * 0.1) * hill_height)
            segment = pymunk.Segment(
                self.space.static_body,
                (x, y1),
                (x + step_size, y2),
                2.0
            )
            segment.friction = 1.6
            self.space.add(segment)

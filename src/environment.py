import pymunk
import numpy as np

class CreatureEnv:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)  # PyGame Y points down, so positive gravity is down
        self.create_ground()
        self.create_creature()
        self.steps = 0

    def create_ground(self):
        body = self.space.static_body
        shape = pymunk.Segment(body, (-10000, 500), (10000, 500), 5.0)  # Ground at Y=500
        shape.friction = 1.0
        self.space.add(shape)

    def create_creature(self):
        # Torso
        mass = 10
        width = 60
        height = 20
        moment = pymunk.moment_for_box(mass, (width, height))
        self.torso = pymunk.Body(mass, moment)
        self.torso.position = (400, 440)  # Spawn standing on the ground instead of dropping from the sky
        shape = pymunk.Poly.create_box(self.torso, (width, height))
        shape.friction = 0.5
        shape.filter = pymunk.ShapeFilter(group=1)
        self.space.add(self.torso, shape)

        self.legs = []
        self.joints = []
        self.motors = []

        leg_mass = 2
        leg_width = 10
        leg_length = 40
        leg_moment = pymunk.moment_for_box(leg_mass, (leg_width, leg_length))

        # Attach 4 legs: front-left, front-right, back-left, back-right
        anchors = [
            (-width/2, -height/2), (width/2, -height/2),
            (-width/2, height/2), (width/2, height/2) 
        ]
        
        for anchor in anchors:
            leg = pymunk.Body(leg_mass, leg_moment)
            leg.position = (self.torso.position.x + anchor[0], self.torso.position.y + anchor[1] - leg_length/2)
            leg_shape = pymunk.Poly.create_box(leg, (leg_width, leg_length))
            leg_shape.friction = 0.8
            leg_shape.filter = pymunk.ShapeFilter(group=1) # Don't collide with torso or other legs
            self.space.add(leg, leg_shape)
            self.legs.append(leg)

            # Pivot Joint
            pivot = pymunk.PivotJoint(self.torso, leg, self.torso.local_to_world(anchor))
            # Rotary Limit Joint (-60 to 60 degrees relative to torso to prevent full splits)
            limit = pymunk.RotaryLimitJoint(self.torso, leg, -np.pi/3, np.pi/3)
            # Motor to apply torque
            motor = pymunk.SimpleMotor(self.torso, leg, 0)
            motor.max_force = 150000 # Lower max torque so it doesn't violently flip itself over
            
            self.space.add(pivot, limit, motor)
            self.joints.append(pivot)
            self.motors.append(motor)

    def reset(self):
        self.__init__()
        return self.get_state()

    def step(self, actions):
        # Actions are desired velocities for the motors
        for i, motor in enumerate(self.motors):
            motor.rate = float(actions[i]) * 10.0 # Scale action to velocity
        
        old_x = self.torso.position.x
        self.space.step(1/60.0)
        self.steps += 1
        
        state = self.get_state()
        
        # Calculate incremental reward
        forward_progress = self.torso.position.x - old_x
        # Reduce energy penalty heavily so it isn't afraid to move its legs
        energy_penalty = sum(abs(float(a)) for a in actions) * 0.005
        # Penalize tilting too much so it tries to stay upright
        tilt_penalty = abs(self.torso.angle) * 0.05
        
        reward = forward_progress - energy_penalty - tilt_penalty
        
        # Check if the creature has fallen over (angle > 1.2 rad or < -1.2 rad)
        fallen = abs(self.torso.angle) > 1.2
        done = self.steps > 600 or fallen
            
        return state, reward, done, fallen

    def get_state(self):
        state = []
        state.append(self.torso.angle)
        state.append(self.torso.angular_velocity)
        state.append(self.torso.velocity.x)
        state.append(self.torso.velocity.y)
        state.append(self.torso.position.y)
        
        for leg in self.legs:
            state.append(leg.angle - self.torso.angle)
            state.append(leg.angular_velocity)
            
        return np.array(state, dtype=np.float32)

import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Define target position and range
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        
        # define range of speed rotors
        speed = 400
        self.action_low = 0.9 * speed
        self.action_high = 1.1 * speed
        
        # define action and state size
        self.action_size = 1
        self.state_size = 3 #for position, velocity and accelaration on z axis

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-abs(self.sim.pose[2] - self.target_pos[2])
        return reward
    
    def get_state(self):
        position_error = (self.sim.pose[2] - self.target_pos[2])

        return np.array([ position_error, self.sim.v[2], self.sim.linear_accel[2] ])

    
    def convert_action(self, action):
        """ convert action between -1 and 1 to low and high """
        return 0.5*(self.action_high*(1+action) + self.action_low*(1-action))

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        
        speed_of_rotors = self.convert_action(action)
#        print(speed_of_rotors)
        is_done = self.sim.next_timestep(speed_of_rotors*np.ones(4))
        next_state = self.get_state()
        reward = self.get_reward()

        if reward <= 0:
            is_done = True

        return next_state, reward, is_done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = self.get_state()
        return state
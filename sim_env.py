# FILE: sim_env.py
import numpy as np
import mujoco
import mediapy as media

# A simple MuJoCo XML definition for a 4-DOF arm and a generic object (cube)
XML_SCENE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="1 1 1"/>
    <body name="floor" pos="0 0 -0.1">
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
    </body>

    <!-- The Robot Arm (Simplified 4-joint arm + gripper) -->
    <body name="base" pos="0 0 0">
        <geom type="cylinder" size="0.1 0.1" rgba=".5 .5 .5 1"/>
        <body name="link1" pos="0 0 0.1">
            <joint name="joint1" axis="0 0 1" range="-3 3" damping="1.0"/>
            <geom type="capsule" fromto="0 0 0 0.2 0 0.2" size="0.04" rgba="0.2 0.6 0.8 1"/>
            <body name="link2" pos="0.2 0 0.2">
                <joint name="joint2" axis="0 1 0" range="-3 3" damping="1.0"/>
                <geom type="capsule" fromto="0 0 0 0.2 0 0.2" size="0.04" rgba="0.2 0.6 0.8 1"/>
                <body name="link3" pos="0.2 0 0.2">
                    <joint name="joint3" axis="0 1 0" range="-3 3" damping="1.0"/>
                    <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.04" rgba="0.2 0.6 0.8 1"/>
                    <body name="hand" pos="0.1 0 0">
                        <joint name="joint4" axis="0 1 0" range="-3 3" damping="1.0"/>
                        <geom type="box" size="0.05 0.05 0.05" rgba="0.8 0.2 0.2 1"/>
                        <camera name="wrist_cam" pos="0.1 0 0" fovy="60"/>
                    </body>
                </body>
            </body>
        </body>
    </body>

    <!-- The Object (Cube) -->
    <body name="cube" pos="0.3 0.1 0">
        <joint type="free" name="cube_joint"/>
        <geom type="box" size="0.03 0.03 0.03" rgba="0 1 0 1" mass="0.1"/>
    </body>

    <!-- Static Camera -->
    <camera name="front" pos="0.8 0 0.4" xyaxes="0 1 0 -0.5 0 1"/>
  </worldbody>
  
  <actuator>
    <position joint="joint1" kp="50"/>
    <position joint="joint2" kp="50"/>
    <position joint="joint3" kp="50"/>
    <position joint="joint4" kp="50"/>
  </actuator>
</mujoco>
"""

class SimEnv:
    def __init__(self):
        # Load the model from the XML string
        self.model = mujoco.MjModel.from_xml_string(XML_SCENE)
        self.data = mujoco.MjData(self.model)
        
        # Dimensions for images (matches your config.py)
        self.width = 640
        self.height = 480
        
        # Renderer
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        
        # Internal state
        self.actuator_ids = [0, 1, 2, 3] # Indices of joints
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Randomize cube position slightly
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        # Set cube position (x, y, z)
        self.data.qpos[-7:-4] = [0.3 + np.random.uniform(-0.05, 0.05), 
                                 0.1 + np.random.uniform(-0.05, 0.05), 
                                 0.05]
        mujoco.mj_forward(self.model, self.data)
        
    def step(self, action):
        """
        Mimics the hardware set_goal_pos logic.
        Action: numpy array of target joint positions (radians).
        """
        self.data.ctrl[self.actuator_ids] = action
        # Step physics multiple times to stabilize
        for _ in range(50): 
            mujoco.mj_step(self.model, self.data)

    def get_obs(self):
        """
        Returns observation dict matching robot.py/record_episodes.py structure
        """
        # Get Joint Positions (qpos)
        qpos = np.array([self.data.qpos[i] for i in range(len(self.actuator_ids))])
        
        # Get Joint Velocities (qvel)
        qvel = np.array([self.data.qvel[i] for i in range(len(self.actuator_ids))])
        
        # Render Camera
        self.renderer.update_scene(self.data, camera="front")
        img_front = self.renderer.render()
        
        return {
            'qpos': qpos,
            'qvel': qvel,
            'images': {'front': img_front}
        }
import numpy as np
import mujoco
import mediapy as media

XML_SCENE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="1 1 1"/>
    <body name="floor" pos="0 0 -0.1">
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
    </body>
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
                        <site name="end_effector" pos="0.05 0 0" size="0.01"/>
                    </body>
                </body>
            </body>
        </body>
    </body>
    <body name="cube" pos="0.3 0.1 0">
        <joint type="free" name="cube_joint"/>
        <geom type="box" size="0.03 0.03 0.03" rgba="0 1 0 1" mass="0.1"/>
    </body>
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
        self.model = mujoco.MjModel.from_xml_string(XML_SCENE)
        self.data = mujoco.MjData(self.model)
        self.width = 640
        self.height = 480
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.actuator_ids = [0, 1, 2, 3] 

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_cube_pos(self, pos):
        start_idx = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")]
        self.data.qpos[start_idx:start_idx+3] = pos
        mujoco.mj_forward(self.model, self.data)

    def get_ee_pos(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        return self.data.site_xpos[site_id]
        
    def step(self, action):
        self.data.ctrl[self.actuator_ids] = action
        for _ in range(50): 
            mujoco.mj_step(self.model, self.data)

    def get_obs(self):
        qpos = np.array([self.data.qpos[i] for i in range(len(self.actuator_ids))])
        qvel = np.array([self.data.qvel[i] for i in range(len(self.actuator_ids))])
        self.renderer.update_scene(self.data, camera="front")
        img_front = self.renderer.render()
        return {'qpos': qpos, 'qvel': qvel, 'images': {'front': img_front}}
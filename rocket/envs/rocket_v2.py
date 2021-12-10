# **********************************************************************
#
# Copyright (c) 2019, Autonomous Systems Lab
# Author: Inkyu Sa <enddl22@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# *************************************************************************

import numpy as np
from numpy import linalg
from gym import utils

from gym.envs.mujoco import mujoco_env
from scipy.spatial.transform import Rotation as R

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class RocketV2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.log_cnt = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, '/home/aswerdlow/EE209AS/RL/mujoco/rocket/envs/rocket_v2.xml', 5)
        

    def step(self, action):
        output = 0 if np.isnan(action[0]) else np.clip(action[0], a_min=-1, a_max=1)
        self.do_simulation(np.array([np.random.normal(10.22, 0.06), output]), self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:3]
        quat = ob[3:7]
        slider_pos = ob[7]
        lin_vel = ob[8:11]
        ang_vel = ob[11:14]
        slider_vel = ob[14]

        reward_slider_position = -slider_pos*10
        reward_slider_velocity = 0 #-slider_vel
        reward_ctrl = -np.square(output) # -np.sum(np.square(action))
        reward_position = -linalg.norm(pos - np.array([0, 0, 1.0]))
        reward_linear_velocity = 0 #-linalg.norm(lin_vel)
        reward_angular_velocity = -2 * linalg.norm(ang_vel)
        reward_orientation = -2 * linalg.norm(angle_between(R.from_quat(quat).as_rotvec(), np.array([np.pi, 0, 0])))
        
        reward = reward_position + reward_linear_velocity + reward_angular_velocity + reward_slider_position + reward_slider_velocity + reward_orientation
        # print(reward)
        done = bool(np.any(pos, where=abs(pos) > 2).astype('bool') or pos[2] <= 0.1)
        # breakpoint()
        if type(done) != bool:
            print(type(done))
        info = {
            'rp': reward_position,
            'rlv': reward_linear_velocity,
            'rav': reward_angular_velocity,
            'rctrl': reward_ctrl,
            'obx': pos[0],
            'oby': pos[1],
            'obz': pos[2],
            'obvx': lin_vel[0],
            'obvy': lin_vel[1],
            'obvz': lin_vel[2],
            'slp': reward_slider_position,
            'slv': reward_slider_velocity,
            'ro': reward_orientation,
            'o' : R.from_quat(quat).as_rotvec()
        }
        

        # if self.log_cnt % 10000 == 0:
        #     print(f"x={pos[0]},y={pos[1]},z={pos[2]}")
        #     print("slider={}".format(action[0]))
        #     self.log_cnt = 0
        # else:
        #     self.log_cnt += 1

        return ob, reward, done, info

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def reset_model(self):
        qpos = self.init_qpos
        qpos[7] += self.np_random.uniform(size=1, low=-0.001, high=0.001)
        qvel = self.init_qvel # + self.np_random.uniform(size=self.model.nv, low=-0.05, high=0.05)
        # print(qpos, qvel)
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 1
        v.cam.distance = self.model.stat.extent * 20
        v.cam.azimuth = 132.
        v.cam.elevation += 5

import registration
import gymnasium as gym
import sys
import cv2
import numpy as np
from lerobot_data_collection import DataCollection
import time
from PS5_controller_driver import ps5_controller_driver

from collections import deque
import numpy as np

class MovingAverageFilter:
    def __init__(self, window=5):
        self.window = window
        self.buffer = deque(maxlen=window)

    def filter(self, action):
        self.buffer.append(action)
        # Compute moving average: axis=0 to average over timesteps
        return np.mean(self.buffer, axis=0).tolist()

class HealImitation():
    def __init__(self):
        self._data_collection = DataCollection()
        self._data_collection.episode_no = 0
        self._data_collection.global_frame_idx = 0
        self.action_scaling = 100

        self._joy = ps5_controller_driver()
        self._joy.controller.activate()

        self.joy_filter = MovingAverageFilter(window=30)

        self.state_filter = MovingAverageFilter(window=30)
        self.action_filter = MovingAverageFilter(window=30)


        self.main_env = gym.make("Heal-v1", render_mode="rgb_array")
        self.main_env.reset()

        self._env = self.main_env.env.env.env

        self.action_trajectory = []

        self.start_state = [-1.5730853391174673, 0.15484474915392638, -0.051969636236800565,
                             -0.24907753891966092, 0.2833406040512775, -6.95920944727502e-07, 
                             1.4214986626857298]


    def run(self):
        print("Started data collection starting with" \
            f"episode no: {0.0}")
        episode_idx = 0
        state = self.start_state
        action = None

        while "circle" not in self._joy.get_pressed_buttons():
            time_stamp = 0.0
            frame_idx = 0
            buffer_limit = 50
            count = 0
            state = self.start_state
            prev_state = state
            self._data_collection.episode_no = episode_idx
            self._data_collection.reset_recorder()
            action = [0, 0, 0, 0, 0]

            for i in range(30):
                self.state_filter.filter(state)
                self.action_filter.filter([0, 0, 0, 0, 0, 0, 0])

            while True:
                if count == buffer_limit:
                    action = self._joy.get_joystick_values() + [self._joy.get_gripper_value()]
                    action = self.joy_filter.filter(action)

                obs, reward , terminated, truncated, info = self._env.step(action)

                frame = self._env.render()
                state = obs[:6].tolist() 
                state.append(obs[-2])

                state = self.state_filter.filter(state)

                img_frame = np.hstack([frame['wrist_cam'], frame['bev_cam'], frame['front']])
                cv2.imshow(f'Frame', img_frame)
                cv2.waitKey(1)

                if count < buffer_limit:
                    count = count + 1
                    prev_state = state
                    continue

            
                self._data_collection.record_video(camera="top", img=frame['bev_cam'] )
                self._data_collection.record_video(camera="wrist", img=frame['wrist_cam'])

                
                self._data_collection.frame_idx = frame_idx

                next_state = state[:6]
                next_state.append(int(255 / 2 * (action[-1] + 1))/255)

                self.action_trajectory.append(next_state)

                if "cross" in self._joy.get_pressed_buttons():
                    self._data_collection.record_state(state=prev_state, action=next_state, time_stamp=time_stamp, done=1)
                    break
                else:
                    self._data_collection.record_state(state=prev_state, action=next_state, time_stamp=time_stamp, done=0)

                time_stamp +=0.02
                frame_idx+=1
                prev_state = state

            self._data_collection.global_frame_idx = frame_idx + 1

            print("episode ended ...")
            print("saving episode data")
            if self._data_collection.stop_recording():
                episode_idx+=1
            time.sleep(2)
            self._data_collection.data.clear()
            self.main_env.reset()
            
            

if __name__ == '__main__':
    heal = HealImitation()
    heal.run()
    np.save('action_trajectory.npy', np.array(heal.action_trajectory))






            







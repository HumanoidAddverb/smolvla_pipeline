import socket
import pickle
import struct
import cv2
import numpy as np

import torch
import time
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from transformers import AutoProcessor

import registration
import gymnasium as gym
import cv2
import numpy as np
import time
from collections import deque

import os
import cv2
import torch
import numpy as np
import pandas as pd
from transformers import AutoProcessor
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import matplotlib.pyplot as plt
from collections import deque
import sys
import time 
 
# --- SmolVLA Inference Class ---
class SmolVLAInference:
    def __init__(self, checkpoint_path, device="cuda"):
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint_path).to(device)
        self.policy.eval()
        self.policy.language_tokenizer = AutoProcessor.from_pretrained(
            self.policy.config.vlm_model_name
        ).tokenizer
        self.device = device
        self.state_dim = self.policy.normalize_inputs.buffer_observation_state.mean.shape[-1]
        print(f"[INFO] State dimension: {self.state_dim}")
 
    def infer_action(self, top_image_np, wrist_image_np, state_np, task_text, prediction_horizon):
        top_image_tensor = torch.tensor(top_image_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        wrist_image_tensor = torch.tensor(wrist_image_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_tensor = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
 
        dummy_batch = {
            "observation.images.top": top_image_tensor,
            "observation.images.wrist": wrist_image_tensor,
            "observation.state": state_tensor,
            "task": [task_text],
        }
 
        normalized_batch = self.policy.normalize_inputs(dummy_batch)
        images, img_masks = self.policy.prepare_images(normalized_batch)
        state = self.policy.prepare_state(normalized_batch)
        lang_tokens, lang_masks = self.policy.prepare_language(normalized_batch)
 
        with torch.no_grad():
            action = self.policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
            action_batch = {"action": action[0, :prediction_horizon, 0:7]}
            print(action_batch)
            unnormalized_action = self.policy.unnormalize_outputs(action_batch)["action"]
 
        return unnormalized_action

class MovingAverageFilter:
    def __init__(self, window=5):
        self.window = window
        self.buffer = deque(maxlen=window)

    def filter(self, action):
        self.buffer.append(action)
        # Compute moving average: axis=0 to average over timesteps
        return np.mean(self.buffer, axis=0).tolist()
    

class HealInference():

    def __init__(self):
        self.main_env = gym.make("Heal-v1", render_mode="rgb_array")
        self.main_env.reset()

        self._env = self.main_env.env.env.env

        self.initial_action = [-1.5730853391174673, 0.15484474915392638, -0.051969636236800565, -0.24907753891966092, 0.2833406040512775, -6.95920944727502e-07, 127.5]
        self.state_filter = MovingAverageFilter(window=3)

        self.model = SmolVLAInference(checkpoint_path="lerobot/ouputs/train/heal_imitation_sim_5/checkpoints/last/pretained_model")

        self.action_horizon = 50
        self.action_steps = 30
        self.action_filer = MovingAverageFilter(window=3)

        self.task_text = "Place cube inside the bin"


        height, width, _ = (480, 1920, 3)

        video_path = "replay_inference_3.mp4"
        fps = 30

        self.frame_size = (width, height)
        self.bending_factor = 0.5

        self.writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            self.frame_size
        )

    def run(self):
        action = self.initial_action
        action_chunk = []

        for episode in range(10):
            action = self.initial_action
            prev_action_chunk = None

            for i in range(30):
                obs, reward , terminated, truncated, info = self._env.step(action)
                frame = self._env.render()
                self.action_filer.filter(action)
                state = obs[:6].tolist() 
                state.append(float(obs[-2]))

                state = self.state_filter.filter(state)
                frame = self._env.render()
                
    
                img_frame = np.hstack([frame['wrist_cam'], frame['bev_cam'], frame['front']])
                self.writer.write(img_frame)

                cv2.imshow(f'Frame', img_frame)
                cv2.waitKey(1)
    
            input("test")
            for i in range(600):

                state = obs[:6].tolist() 
                state.append(float(obs[-2]))

                state = self.state_filter.filter(state)

                if not action_chunk:
                    action_chunk = self.inference_model(state, frame['bev_cam'], frame['wrist_cam'])
    
                elif len(action_chunk) == (self.action_horizon - self.action_steps):
                    action_chunk = self.inference_model(state, frame['bev_cam'], frame['wrist_cam'])

                    if prev_action_chunk is not None:
                        overlapping_horizon = self.action_horizon - self.action_steps
                        action_chunk = self.bend_chunks(action_chunk, prev_action_chunk, overlapping_horizon)
                            
                    prev_action_chunk = action_chunk

                action = action_chunk.pop(0)
                action[-1] = 255*action[-1]
    
                action = self.action_filer.filter(action)
    
                obs, reward , terminated, truncated, info = self._env.step(action)
    
                frame = self._env.render()
                
    
                img_frame = np.hstack([frame['wrist_cam'], frame['bev_cam'], frame['front']])
                self.writer.write(img_frame)
    
                cv2.imshow(f'Frame', img_frame)
                cv2.waitKey(1)
            self.main_env.reset()
        self.writer.release()

    def inference_model(self, state, top_image, wrist_image):
        top_img_rgb = cv2.cvtColor(top_image, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        top_img_rgb = np.transpose(top_img_rgb, (2, 0, 1))
 
        wrist_img_rgb = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        wrist_img_rgb = np.transpose(wrist_img_rgb, (2, 0, 1))
        
        state = np.array(state, dtype=np.float32)

        predicted_action = self.model.infer_action(wrist_image_np=wrist_img_rgb, top_image_np=top_img_rgb, state_np=state, 
                                task_text=self.task_text, prediction_horizon=self.action_horizon)
        
        action = predicted_action.cpu().numpy().tolist()


        return action
    
    def bend_chunks(self, chunk1, chunk2, overlapping_horizon):
        chunk1_np = np.array(chunk1)
        chunk2_np = np.array(chunk2)
        
        print(chunk2_np.shape)
        chunk1_np[:overlapping_horizon] = self.bending_factor*chunk1_np[:overlapping_horizon] +\
                                           (1 - self.bending_factor)*chunk2_np[self.action_steps:]
        
        return chunk1_np.tolist()
    
if __name__ == '__main__':
    env = HealInference()

    env.run()







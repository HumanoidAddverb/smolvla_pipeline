import pandas as pd 
import os
import json
import numpy as np 

import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = current_dir + "/data/chunk-000/"
meta_dir = current_dir + "/meta"

epsiode_json_path = meta_dir + "/episodes.jsonl"

global_index = 0

def change_time_stamp(episode):
    '''
    change time stamps of each episode 
    make them consistent across episode'''
    
    print(f"episode no: {episode}")
    episode_path = data_dir + episode

    global global_index

    print(f"global index is {global_index}")
    try:
        df = pd.read_parquet(episode_path)
    except:
        return

    time_stamps = len(df['timestamp'])

    for time_stamp in range(time_stamps):
        df.loc[time_stamp, 'index'] = global_index
        global_index= global_index + 1

    obs = df['observation.state'].to_numpy()
    obs_state = np.zeros((obs.shape[0], 7))

    for t in range(obs_state.shape[0]):
        obs_state[t, :] = obs[t] 

    plt.plot(obs_state[:, 0])
    plt.xlabel('Timestep')
    plt.ylabel('Feature 0')
    plt.title('First feature over time')
    plt.show()
    get_episodes_info(df.loc[0, 'episode_index'], time_stamps)
    
    df.to_parquet(episode_path, index=False, engine='pyarrow')

def get_episodes_info(episode_idex, episode_length):
    with open(epsiode_json_path, "a") as f:
            # Adjust field names to match your Parquet file
            json_obj = {
                "episode_index": int(episode_idex),
                "tasks": "Place two cube boxes inside the bin",
                "length": int(episode_length)
            }

            f.write(json.dumps(json_obj) + "\n")

episodes = len(os.listdir(data_dir))

for episode in range(episodes):
    change_time_stamp(f"episode_{episode:06d}.parquet")


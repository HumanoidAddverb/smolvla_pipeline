import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import imageio.v3 as iio

def compute_stats(arr):
    arr = np.array(arr)
    return {
        "min": arr.min(axis=0).tolist() if arr.size > 0 else [0],
        "max": arr.max(axis=0).tolist() if arr.size > 0 else [0],
        "mean": arr.mean(axis=0).tolist() if arr.size > 0 else [0.0],
        "std": arr.std(axis=0).tolist() if arr.size > 0 else [0.0],
        "count": [len(arr)]
    }

def compute_image_stats_last_frames(video_path, max_frames=100):
    try:
        frames_iter = iio.imiter(str(video_path))
        frames = []
        for frame in frames_iter:
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
    except Exception as e:
        print(f"Warning: Failed to read video {video_path} with imageio: {e}")
        zero_stat = [[[0.0]], [[0.0]], [[0.0]]]
        return {
            "min": zero_stat,
            "max": zero_stat,
            "mean": zero_stat,
            "std": zero_stat,
            "count": [0]
        }

    frames = np.stack(frames, axis=0)  # shape (num_frames, H, W, 3)
    frames = frames[-max_frames:]  # last max_frames frames

    min_vals = []
    max_vals = []
    mean_vals = []
    std_vals = []

    for c in range(3):
        channel_data = frames[..., c].reshape(-1)
        min_vals.append([[float(np.min(channel_data))]])
        max_vals.append([[float(np.max(channel_data))]])
        mean_vals.append([[float(np.mean(channel_data))]])
        std_vals.append([[float(np.std(channel_data))]])

    return {
        "min": min_vals,
        "max": max_vals,
        "mean": mean_vals,
        "std": std_vals,
        "count": [len(frames)]
    }

def compute_stats_list(arr):
    arr = np.array(arr)
    if arr.size == 0:
        return {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [0]}
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [len(arr)]
    }


def main():
    base_dir = Path('heal_pnp')
    videos_dir = base_dir / 'videos' / 'chunk-000'
    meta_dir = base_dir / 'meta'
    data_dir = base_dir / 'data' / 'chunk-000'
    meta_dir.mkdir(parents=True, exist_ok=True)

    top_dir = videos_dir / 'observation.images.top'
    wrist_dir = videos_dir / 'observation.images.wrist'
    episode_indices = sorted([int(f.stem.split('_')[1]) for f in top_dir.glob('*.mp4')])

    with open(meta_dir / 'episodes_stats.jsonl', 'w') as f:
        for episode_index in tqdm(episode_indices, desc="Processing episodes"):
            stats = {}

            parquet_file = data_dir / f'episode_{episode_index:06d}.parquet'
            df = pd.read_parquet(parquet_file)

            # action & observation.state (full stats)
            stats["action"] = compute_stats(df["action"].tolist()) if "action" in df.columns else {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [0]}
            stats["observation.state"] = compute_stats(df["observation.state"].tolist()) if "observation.state" in df.columns else {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [0]}

            # observation.images.top and wrist last 100 frames
            top_video = top_dir / f'episode_{episode_index:06d}.mp4'
            wrist_video = wrist_dir / f'episode_{episode_index:06d}.mp4'

            stats["observation.images.top"] = compute_image_stats_last_frames(top_video, max_frames=100)
            stats["observation.images.wrist"] = compute_image_stats_last_frames(wrist_video, max_frames=100)

            # other scalar keys - timestamp, frame_index, episode_index, index, task_index
            for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
                if key in df.columns:
                    stats[key] = compute_stats_list(df[key].tolist())
                else:
                    stats[key] = {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [0]}

            episode_json = {
                "episode_index": episode_index,
                "stats": stats
            }
            f.write(json.dumps(episode_json) + "\n")


if __name__ == "__main__":
    main()

# "observation.images.top": {"min": [[[0.062745101749897]], [[0.0941176488995552]], [[0.03921568766236305]]], "max": [[[1.0]], [[1.0]], [[1.0]]], "mean": [[[0.49355241656303406]], [[0.5032037496566772]], [[0.5096811056137085]]], "std": [[[0.23679950833320618]], [[0.2114802449941635]], [[0.17989380657672882]]], "count": [100]}, "observation.images.wrist": {"min": [[[0.0]], [[0.0]], [[0.0]]], "max": [[[1.0]], [[1.0]], [[1.0]]], "mean": [[[0.5903795957565308]], [[0.540744960308075]], [[0.5339071154594421]]], "std": [[[0.2598201632499695]], [[0.25439774990081787]], [[0.25117114186286926]]], "count": [100]}, "timestamp": {"min": [0.0], "max": [14.866666793823242], "mean": [7.4333333352151465], "std": [4.301248743187641], "count": [447]},

# "observation.images.top": {"min": [[[0.06666666666666667]], [[0.09019607843137255]], [[0.0]]], "max": [[[1.0]], [[1.0]], [[1.0]]], "mean": [[[0.4998802920751634]], [[0.5033802879901961]], [[0.5087131147875817]]], "std": [[[0.2363218740821347]], [[0.2092585693168179]], [[0.17750579628174187]]], "count": [100]}, "observation.images.wrist": {"min": [[[0.0]], [[0.0]], [[0.0]]], "max": [[[1.0]], [[1.0]], [[1.0]]], "mean": [[[0.586954087009804]], [[0.5323829207516341]], [[0.5309204146241829]]], "std": [[[0.2677416344558895]], [[0.25789101345432675]], [[0.25153248107418824]]], "count": [100]}, "timestamp": {"min": [0.0], "max": [14.866666666666667], "mean": [7.433333333333333], "std": [4.301248742021407], "count": [447]}, 

# "observation.images.top": {"min": [[[0.03921568766236305]], [[0.04313725605607033]], [[0.0]]], "max": [[[1.0]], [[1.0]], [[1.0]]], "mean": [[[0.5010221004486084]], [[0.5058926939964294]], [[0.5100165009498596]]], "std": [[[0.23849886655807495]], [[0.21122726798057556]], [[0.18007060885429382]]], "count": [447]}, "observation.images.wrist": {"min": [[[0.0]], [[0.0]], [[0.0]]], "max": [[[1.0]], [[1.0]], [[1.0]]], "mean": [[[0.5854718089103699]], [[0.5316318273544312]], [[0.5300702452659607]]], "std": [[[0.26759272813796997]], [[0.2583354413509369]], [[0.25290346145629883]]], "count": [447]}, "timestamp": {"min": [0.0], "max": [14.866666793823242], "mean": [7.4333333352151465], "std": [4.301248743187641],
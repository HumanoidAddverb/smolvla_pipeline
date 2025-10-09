
import cv2
import pandas as pd
import time
import os


class DataCollection():
    def __init__(self, ):
        self.parent_dir = "training_script/dataset"

        self.dataset_dir = self.parent_dir + "/heal_pnp_4"
        self.chunk_no = "000/"

        self.video_dir = self.dataset_dir + "/videos" + "/chunk-" + self.chunk_no

        self.data_dir = self.dataset_dir + "/data" + "/chunk-" + self.chunk_no

        episodes = len(os.listdir(self.data_dir))
        self.episode_no = episodes

        self.writers = {"wrist": None, "top": None}

        self.start_time = None
        self.frame_idx = 0
        self.global_frame_idx = 0
        self.data = []
        self.fps = 50

        print("Data collection node initialized.")

    def reset_recorder(self):
        self.writers = {"wrist": None, "top": None}


    def record_video(self, camera: str, img):
        if self.writers[camera] is None:
                if camera == "top":
                    camera_dir = "observation.images.top"
                else:
                    camera_dir = "observation.images.wrist"

                output_file = self.video_dir + camera_dir + f"/episode_{self.episode_no:06d}" + ".mp4"

                print(output_file)

                height, width, _ = img.shape
                self.frame_size = (width, height)
                self.writers[camera] = cv2.VideoWriter(
                    output_file,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    self.frame_size
                )

                print(f'Video writer initialized: {output_file} at {self.fps} FPS')

        self.writers[camera].write(img)

    def record_state(self, state, action, time_stamp, done=0):
        row = {
            'observation.state': state ,
            'action': action,
            'timestamp': time_stamp,
            'episode_index': self.episode_no,
            'frame_index': self.frame_idx,
            'index': self.frame_idx + self.global_frame_idx,
            'done': done,
            'task_index': 0
            }
        
        self.data.append(row)
        self.frame_idx = self.frame_idx + 1

    def stop_recording(self):
        print(f'Episode: {self.episode_no}, final_index: {self.frame_idx + self.global_frame_idx}')
        print('Want to save episode? [Y/n]')
        save = input("Enter: ")

        if (save == 'Y'):
            output_file = self.data_dir + f"episode_{self.episode_no:06d}" + ".parquet"
            print(output_file)
        
            self.data[-1]["done"] = 1
            self.df = pd.DataFrame(self.data)
            self.df.to_parquet(output_file, index=False, engine='pyarrow')
            
            for name in self.writers.keys():
                
                if self.writers[name] is not None:
                    print("saving videos")
                    self.writers[name].release()
                else:
                    print(f"no writer present for camera {name}")
                    print(f"consider deleting this episode ...")
            return True
        else:
            print(f'not saving the episode: {self.episode_no}, final_frame_idx {self.global_frame_idx}')
            return False

    def on_shutdown(self):
        self.stop_recording()

def main(args=None):
    data_collection = DataCollection()


if __name__ == '__main__':
    main()

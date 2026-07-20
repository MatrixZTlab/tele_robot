import os
import cv2
import json
import datetime
import shutil
import numpy as np
import time
from .rerun_visualizer import RerunLogger
from queue import Queue, Empty
from threading import Lock, Thread
import logging_mp
logger_mp = logging_mp.getLogger(__name__)

class EpisodeWriter():
    def __init__(self, task_dir, task_goal=None, task_desc = None, task_steps = None,
                 frequency=30, image_size=[640, 480], rerun_log = True,
                 tolerance_s=1e-4):
        """
        image_size: [width, height]
        """
        logger_mp.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "task description",
            "steps":"step1: do this; step2: do that; ...",
        }
        if task_goal is not None:
            self.text['goal'] = task_goal
        if task_desc is not None:
            self.text['desc'] = task_desc
        if task_steps is not None:
            self.text['steps'] = task_steps

        self.frequency = float(frequency)
        self.image_size = image_size
        self.tolerance_s = float(tolerance_s)

        self.rerun_log = rerun_log
        if self.rerun_log:
            logger_mp.info("==> RerunLogger initializing...\n")
            self.rerun_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit = "300MB")
            logger_mp.info("==> RerunLogger initializing ok.\n")
        
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir and not episode_dir.endswith('.zip')]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info(f"==> episode directory does not exist, now create one.\n")
        self.data_info()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.need_discard = False
        self._discard_episode_dir = None
        self._action_in_progress = None
        self._lifecycle_lock = Lock()
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.\n")
    
    def is_ready(self):
        with self._lifecycle_lock:
            return self.is_available

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "unitree" if author is None else author,
                "fps": self.frequency,
                "tolerance_s": self.tolerance_s,
                "alignment": {
                    "scheme": "lerobot_fps_grid",
                    "timestamp_formula": "timestamp = frame_index / fps",
                    "actual_sensor_timestamps": "stored per frame under alignment.actual_timestamps_ns",
                },
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   [],
                    "left_ee":  [],
                    "right_arm":  [],
                    "right_ee": [],
                    "body":       [],
                },

                "tactile_names": {
                    "left_ee": [],
                    "right_ee": [],
                }, 
                "sim_state": ""
            }

 
    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_ready():
            logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write('{\n')
            f.write('"info": ' + json.dumps(self.info, ensure_ascii=False, indent=4) + ',\n')
            f.write('"text": ' + json.dumps(self.text, ensure_ascii=False, indent=4) + ',\n')
            f.write('"data": [\n')
        self.first_item = True   # Flag to handle commas in JSON array

        if self.rerun_log:
            self.online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit="300MB")

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None,
                 audios=None, sim_state=None, alignment=None):
        # Increment the item ID
        self.item_id += 1
        frame_index = self.item_id
        timestamp = frame_index / self.frequency if self.frequency > 0 else 0.0
        # Create the item data dictionary
        item_data = {
            'idx': self.item_id,
            'frame_index': frame_index,
            'timestamp': timestamp,
            'episode_index': self.episode_id,
            'colors': colors,
            'depths': depths,
            'states': states,
            'actions': actions,
            'tactiles': tactiles,
            'audios': audios,
            'sim_state': sim_state,
            'alignment': alignment or {},
        }
        # Enqueue the item data
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    logger_mp.info(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            if self.item_data_queue.empty():
                self._finish_pending_episode_action()

    def _finish_pending_episode_action(self):
        with self._lifecycle_lock:
            if self.need_discard:
                self.need_discard = False
                self.need_save = False
                episode_dir = self._discard_episode_dir
                self._discard_episode_dir = None
                action = "discard"
            elif self.need_save:
                self.need_save = False
                episode_dir = None
                action = "save"
            else:
                return
            self._action_in_progress = action

        if action == "discard":
            self._discard_episode(episode_dir)
        else:
            self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})

        # Save images
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f'{str(idx).zfill(6)}_{color_key}.jpg'
                if not cv2.imwrite(os.path.join(self.color_dir, color_name), color):
                    logger_mp.info(f"Failed to save color image.")
                item_data['colors'][color_key] = os.path.join('colors', color_name)

        # Save depths
        if depths:
            for idx_depth, (depth_key, depth) in enumerate(depths.items()):
                depth_name = f'{str(idx).zfill(6)}_{depth_key}.png'
                if not cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth):
                    logger_mp.info(f"Failed to save depth image.")
                item_data['depths'][depth_key] = os.path.join('depths', depth_name)

        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # Update episode data
        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self.first_item:
                f.write(",\n")
            f.write(json.dumps(item_data, ensure_ascii=False, indent=4))
            self.first_item = False

        # Log data if necessary
        if self.rerun_log:
            curent_record_time = time.time()
            logger_mp.info(f"==> episode_id:{self.episode_id}  item_id:{idx}  current_time:{curent_record_time}")
            self.rerun_logger.log_item_data(item_data)

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        with self._lifecycle_lock:
            self.need_save = True
        logger_mp.info(f"==> Episode saved start...")

    def discard_episode(self, episode_index=None):
        """Discard the current or most recently saved episode."""
        target_index = self.episode_id if episode_index is None else int(episode_index)
        episode_dir = os.path.join(
            self.task_dir,
            f"episode_{str(target_index).zfill(4)}",
        )
        if target_index < 0 or not os.path.exists(episode_dir):
            return None

        with self._lifecycle_lock:
            self.need_discard = True
            self.need_save = False
            self._discard_episode_dir = episode_dir
            self.is_available = False
        logger_mp.warning(f"==> Episode discard queued: {episode_dir}")
        return episode_dir

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        try:
            with open(self.json_path, "a", encoding="utf-8") as f:
                f.write("\n]\n}")      # Close the JSON array and object
            logger_mp.info(f"==> Episode saved successfully to {self.json_path}.")
        finally:
            with self._lifecycle_lock:
                self._action_in_progress = None
                if not self.need_discard:
                    self.is_available = True

    def _discard_episode(self, episode_dir):
        try:
            if os.path.exists(episode_dir):
                shutil.rmtree(episode_dir)
            logger_mp.warning(f"==> Episode discarded: {episode_dir}")
        except Exception:
            logger_mp.exception(f"==> Failed to discard episode: {episode_dir}")
        finally:
            with self._lifecycle_lock:
                self._action_in_progress = None
                self.is_available = True

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        with self._lifecycle_lock:
            discard_pending = (
                self.need_discard or self._action_in_progress == "discard"
            )
            save_pending = self.need_save or self._action_in_progress == "save"
        if not self.is_ready() and not discard_pending and not save_pending:
            self.save_episode()
        while not self.is_ready():
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()

import cv2
import numpy as np
import subprocess as sp
import random
import gym
from MAMEToolkit.sf_environment import Environment
from MAMEToolkit.sf_environment.Environment import index_to_attack_action, index_to_move_action
from macro import index_to_comb, MACRO_NUMS

import os
os.system("Xvfb :1 -screen 0 800x600x16 +extension RANDR &")
os.environ["DISPLAY"] = ":1"


class Monitor(object):
    def __init__(self, width, height, saved_path):

        self.command = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s",
            "{}X{}".format(width, height), "-pix_fmt", "rgb24", "-r", "30",
            "-i", "-", "-an", "-vcodec", "mpeg4", saved_path
        ]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (128, 128)) / 255.
        return frame.transpose((2, 0, 1))
    else:
        return np.zeros((128, 128))


class StreetFighterEnv(gym.Env):
    def __init__(self, index, difficulty, monitor=None):
        roms_path = "roms/"
        self.env = Environment("env{}".format(index),
                               roms_path,
                               difficulty=difficulty)
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None
        self.env.start()

        self.action_space = gym.spaces.Discrete(90)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(3 + self.action_space.n, 128, 128),
                                                dtype=np.float32)

    def step(self, action):
        move_action = action // 10
        attack_action = action % 10
        frames, reward, round_done, stage_done, game_done = self.env.step(
            move_action, attack_action)

        if self.monitor:
            for frame in frames:
                self.monitor.record(frame)

        states = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        if not (round_done or stage_done or game_done):
            states[:3, :] = process_frame(frames[-1])
        else:
            self.env.reset()
            action = 80
        states[action + 3, :] = 1

        reward = reward["P1"] / 10
        if stage_done:
            reward += 3
        elif game_done:
            reward -= 5

        info = {
            'stage_done': stage_done,
            'round_done': round_done,
            'stage': self.env.stage
        }
        return states, reward, game_done, info

    def reset(self):
        self.env.new_game()
        
        states = np.zeros(self.observation_space.shape, dtype=np.float32)
        states[80 + 3, :] = 1
        return states

    def __exit__(self, *args):
        return self.env.close()


class MacroStreetFighterEnv(gym.Env):
    def __init__(self, index, difficulty, monitor=None):
        roms_path = "roms/"
        self.env = Environment("env{}".format(index),
                               roms_path,
                               difficulty=difficulty)
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None
        self.env.start()
        
        self.action_space = gym.spaces.Discrete(18 + MACRO_NUMS)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(3 + self.action_space.n, 128, 128),
                                                dtype=np.float32)

    def step(self, action):
        frames, reward, round_done, stage_done, game_done = self.step_(action)

        if self.monitor:
            for frame in frames:
                self.monitor.record(frame)

        states = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        if not (round_done or stage_done or game_done):
            states[:3, :] = process_frame(frames[-1])
        else:
            self.env.reset()
            action = 8
            
        states[action + 3, :] = 1

        reward = reward["P1"] / 10
        if stage_done:
            reward += 3
        elif game_done:
            reward -= 5

        info = {
            'stage_done': stage_done,
            'round_done': round_done,
            'stage': self.env.stage
        }
        return states, reward, game_done, info

    def step_(self, action):
        if self.env.started:
            if not self.env.round_done and not self.env.stage_done and not self.env.game_done:

                if action < 9:
                    actions = index_to_move_action(action)
                elif action < 18:
                    actions = index_to_attack_action(action - 9)
                elif action < 18 + MACRO_NUMS:
                    actions = index_to_comb[action - 18]()
                else:
                    raise EnvironmentError("Action out of range")
                
                if action < 18:
                    data = self.env.gather_frames(actions)
                else:
                    data = self.sub_step_(actions)

                data = self.env.check_done(data)
                return data["frame"], data[
                    "rewards"], self.env.round_done, self.env.stage_done, self.env.game_done
            else:
                raise EnvironmentError(
                    "Attempted to step while characters are not fighting")
        else:
            raise EnvironmentError("Start must be called before stepping")

    def sub_step_(self, actions):
        frames = []
        for step in actions:
            for i in range(step["hold"]):
                data = self.env.emu.step(
                    [action.value for action in step["actions"]])
                frames.append(data['frame'])
        data = self.env.emu.step([])
        frames.append(data['frame'])

        p1_diff = (self.env.expected_health["P1"] - data["healthP1"])
        p2_diff = (self.env.expected_health["P2"] - data["healthP2"])
        self.env.expected_health = {
            "P1": data["healthP1"],
            "P2": data["healthP2"]
        }

        rewards = {"P1": (p2_diff - p1_diff), "P2": (p1_diff - p2_diff)}

        data["rewards"] = rewards
        data["frame"] = frames
        return data

    def reset(self):
        self.env.new_game()
        
        states = np.zeros(self.observation_space.shape, dtype=np.float32)
        states[8 + 3, :] = 1
        return states

    def __exit__(self, *args):
        return self.env.close()


def create_train_env(index, difficulty, macro, output_path=None):
    if output_path:
        monitor = Monitor(384, 224, output_path)
    else:
        monitor = None

    if macro:
        print('Use macro')
        env = MacroStreetFighterEnv(index, difficulty, monitor)
    else:
        env = StreetFighterEnv(index, difficulty, monitor)
    return env

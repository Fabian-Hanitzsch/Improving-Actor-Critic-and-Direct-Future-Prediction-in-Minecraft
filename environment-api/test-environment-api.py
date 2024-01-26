import json
import os
import uuid
import traceback

import time
from datetime import datetime

import pickle
import asyncio
import threading

from httpx import Client
import socket
from copy import deepcopy

from PIL import Image
import cv2

import torch
import numpy as np

from utils import socket_functions
from utils.Reward import Reward
from utils.StateStorage import StateStorage


class EnvironmentApi:

    def __init__(self):
        self._init_configs()
        self._init_run_configurations()
        self._init_sockets_and_clients()

        self._init_models(base_model_path = "../Trained_Models/data2.pt",weight_path="../Trained_Models/data5.pt")


    def _init_configs(self):
        with open('../Configs/model-parameters.json') as f:
            self.model_parameter_config = json.load(f)
        with open('../Configs/image-stream.json') as f:
            self.image_stream_config = json.load(f)
        with open('../Configs/trainer-parameters.json') as f:
            self.trainer_config = json.load(f)
        with open('../Configs/environment-mineflayer-connection.json') as f:
            self.environment_mineflayer_config = json.load(f)
        with open('../Configs/run-parameters.json') as f:
            self.run_parameter_config = json.load(f)

    def _init_run_configurations(self):
        # General

        self.timestep_delay = self.run_parameter_config["timestep_delay"]
        self.simplified_environment = self.run_parameter_config["simplified_environment"]

        # TCP mineflayer (get State + post Action)
        self.api_prefix_mineflayer: str = '/mineflayer'
        self.http_mineflayer_address = self.environment_mineflayer_config["http_mineflayer_ip_addr"]
        self.http_mineflayer_port = self.environment_mineflayer_config["http_mineflayer_port"]

        # TCP prismarine viewer (The image of the bot view)
        self.prismarine_viewer_port = self.image_stream_config["port"]
        self.prismarine_viewer_ip_addr = self.image_stream_config["ip_addr"]

        self.repeated_dig_commands = 0

        self.simple_dig_action = self.zero_action_state = {
            "forward": False,
            "left": False,
            "right": False,
            "sneak": False,
            "sprint": False,
            "back": False,
            "jump": False,
            "attack": True,
            "use": False,
            "insert": "none",
            "take": "none",
            "craft": "none",
            "equip": "none",
            "camera": [
                0,
                0
            ]
        }

        self.bot_just_died = False

        self.first_image = False
        self.started_digging = False

    def _init_sockets_and_clients(self):
        # Clients for non time critical operations, might have even been fine for time critical operations but in the
        # implementations were many latency spikes from different sources so most time critical operations are written
        # with TCP connections and I see no reason to change it
        self.http_post_client_to_mineflayer = Client()

        # Using the IPC protocol in Linux. The IP address of the socket for mineflayer will be interpreted as a filepath
        # should improve latency by a very small amount (e.g. possible filepath: /share/test.sock)
        self.IPC = self.image_stream_config["IPC"]

        if self.IPC:
            self.socket_to_mineflayer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket_to_mineflayer.bind("/share/echo.sock")

            self.socket_to_viewer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket_to_viewer.bind("/share/viewer.sock")
        else:
            self.socket_to_mineflayer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_to_mineflayer.bind((self.http_mineflayer_address, 5400))

            self.socket_to_viewer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_to_viewer.bind((self.prismarine_viewer_ip_addr, self.prismarine_viewer_port))

        self.socket_to_mineflayer.listen()
        self.socket_to_viewer.listen()


    def _init_models(self, base_model_path="", weight_path=""):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")
        #self.model = torch.load(base_model_path, map_location=self.device)
        #self.model.set_train_mode(False)
        model_parameter = torch.load(weight_path, map_location=self.device)
        #self.model.update_relevant_parameters(model_parameter["model_specific_parameter"])
        #self.model.load_state_dict(model_parameter["model_state_dict"])
        #self.model.device=torch.device("cpu")

        #self.img_dimensions = (self.model.image_width, self.model.image_height, self.model.image_channels)

        self.Reward = Reward()
        self.Reward.define_mission("../Missions/dirt")

        #self.state_storage: StateStorage = StateStorage(self.img_dimensions, self.Reward.get_goal_as_numerical_list(), train_mode=False)
        self.state_storage: StateStorage = StateStorage((84,84,3), self.Reward.get_goal_as_numerical_list(), train_mode=False)
        self.all_losses = {}



    async def update(self, conn):
        state_storage_to_use = self.state_storage

        socket_functions.send_json({"type": "get", "start_time": (time.time_ns() / 1000 / 1000)}, conn)
        answer, error = socket_functions.listen_conn(conn=conn,
                                                     message_target=socket_functions.process_msg_as_json_string,
                                                     threaded=False, repeat=False)
        if error:
            raise Exception

        answer_state = answer["state"]
        if len(answer_state.keys()) == 0:  # bot not ready yet
            return

        if answer_state["just_died"]:
            self.bot_just_died = True

        if not self.first_image:
            return

        action_chosen = False
        if self.repeated_dig_commands > 30:
            self.should_help = False
            self.repeated_dig_commands = 0
            self.started_digging = False

        if self.started_digging:  # and self.should_help:
            block_name = answer_state["block_name"]
            if answer_state["should_dig"] and not answer_state["destroyed_block"] and answer_state[
                "expected_dig_time"] < 5000 \
                    and answer_state["can_harvest_block"] and not answer_state[
                "stopped_digging"] and self.Reward.is_goal_block(block_name):
                action_dict = self.simple_dig_action
                action_dict["repeated"] = True
                action_chosen = True
                self.repeated_dig_commands += 1

            else:
                self.started_digging = False
                self.repeated_dig_commands = 0

        self.repeated_dig_commands = 0
        _ = state_storage_to_use.add_state(answer_state)

        if not action_chosen:
            action_dict = self.predict_action()
            action_dict["repeated"] = False
        cv2.imshow('image', self.state_storage.newest_image)
        cv2.waitKey(1)
        state_storage_to_use.add_action(action_dict)

        socket_functions.send_json({"type": "action", "start_time": (time.time_ns() / 1000 / 1000), "action": action_dict},
                                   conn)
        time_needed, error = socket_functions.listen_conn(conn, socket_functions.process_msg_as_json_string,
                                                          repeat=False)
        if error:
            raise Exception


    def predict_action(self) -> dict:
        action_dict = self.simple_dig_action
        return action_dict

        goal = self.Reward.get_goal_as_numerical_list()
        self.state_storage.add_goal(goal)
        state_with_pov = self.state_storage.get_last_game_state_without_action()
        action_dict, time_needed = self.model.choose_action(state_with_pov)
        self.started_digging = action_dict["attack"]
        action_dict = self.simple_dig_action
        return action_dict


    async def update_loop(self):
        print("Waiting for Mineflayer to start")
        conn, addr = self.socket_to_mineflayer.accept()
        self.test_start_time = time.time_ns()
        while True:
            update_step_start_time = time.time_ns()
            try:
                await self.update(conn)
            except Exception as e:
                print("Error happened in main Loop, reconnecting socket")
                print(e)
                traceback.print_exc()
                conn, addr = self.socket_to_mineflayer.accept()
                continue
            update_step_end_time = time.time_ns()
            update_step_time_needed_s = (update_step_end_time - update_step_start_time) / 1000 / 1000 / 1000
            wait_time = self.timestep_delay - update_step_time_needed_s

            if self.bot_just_died:
                self.bot_just_died = False
                time.sleep(self.timestep_delay)
            elif wait_time > 0:
                time.sleep(wait_time)

    def listen_viewer_tcp(self):
        while True:
            conn, addr = self.socket_to_viewer.accept()
            print("Viewer Socket Connected")

            while True:
                try:
                    message = socket_functions.recv_msg(conn)
                except Exception as e:
                    print("Connection lost for some reason")
                    print(e)
                    break

                if message is None:
                    break

                image = cv2.imdecode(np.fromstring(message, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if not self.first_image:
                    self.first_image = True

                # cv2.imshow('image', image)
                # cv2.waitKey(1)
                self.state_storage.add_img(image)

    def main(self):
        # print("create bucket in Database if it did not already exist")
        # create_bucket()

        thread = threading.Thread(target=self.listen_viewer_tcp)
        thread.daemon = True
        thread.start()

        asyncio.run(self.update_loop())


if __name__ == "__main__":
    Environment = EnvironmentApi()
    Environment.main()

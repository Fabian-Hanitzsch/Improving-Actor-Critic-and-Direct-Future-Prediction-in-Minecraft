import json
import os
import random
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
        self._init_environment_variables()
        self._init_run_configurations()
        self._init_sockets_and_clients()
        self._init_logging_variables()


        self._init_models()
        self.first_full_connection = False



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
    def _init_environment_variables(self):
        self.state_buffer_count = 0 # will be transfered through get model
        self.trainer_ip_address = os.environ.get("trainer_ip_addr")
        if self.trainer_ip_address is None:
            self.trainer_ip_address = "127.0.0.1"
        self.seed = os.environ.get("seed")
        if self.seed is None:
            self.seed = 0
        self.seed = int(self.seed)

        self.destruction_factor = os.environ.get("destruction_factor")
        if self.destruction_factor is None:
            self.destruction_factor = "0.1"
        self.destruction_factor = float(self.destruction_factor)

        self.mission_type = os.environ.get("mission_type")
        if self.mission_type is None:
            self.mission_type = "dirt"


        self.trainer_tcp_port = os.environ.get("trainer_tcp_port")
        if self.trainer_tcp_port is None:
            self.trainer_tcp_port = self.run_parameter_config["port_for_models"]
        self.trainer_tcp_port = int(self.trainer_tcp_port)

        self.session_id = os.environ.get("session_id")
        if self.session_id is None:
            print("No session ID in the environment, creating a unique on the fly")
            self.session_id = datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + "." + str(uuid.uuid1().hex)

        print("Session Id ist: ", self.session_id)

    def _init_run_configurations(self):
        # General

        self.max_episode_count = self.run_parameter_config["max_episode_count"]
        self.current_episode = 1
        self.current_episode_model_step = 1

        self.timestep_delay = self.run_parameter_config["timestep_delay"]
        self.simplified_environment = self.run_parameter_config["simplified_environment"]

        self.state_save_directory = self.run_parameter_config["state_save_directory"]
        self.seconds_between_model_updates = self.run_parameter_config["seconds_between_model_updates"]
        self.save_path = self.state_save_directory + "/" + str(self.session_id) + "/"
        self.state_save_path = self.save_path + "states/"
        self.socket_host = socket.gethostbyname(socket.gethostname())

        #self.socket_host = "127.0.0.1"
        #print(self.socket_host)

        # HTTP Trainer (get model + post logging values)
        self.api_prefix_trainer: str = '/trainer'
        self.trainer_port = self.trainer_config["port"]

        # TCP Trainer (get Model parameters)
        self.port_for_model = int(self.trainer_tcp_port) + 1

        # TCP mineflayer (get State + post Action)
        self.api_prefix_mineflayer: str = '/mineflayer'
        self.http_mineflayer_address = self.environment_mineflayer_config["http_mineflayer_ip_addr"]
        self.http_mineflayer_port = self.environment_mineflayer_config["http_mineflayer_port"]

        # TCP prismarine viewer (The image of the bot view)
        self.prismarine_viewer_port = self.image_stream_config["port"]
        self.prismarine_viewer_ip_addr = self.image_stream_config["ip_addr"]

        self.staggered_model_update_count = self.run_parameter_config["staggered_model_update_count"]
        self.previous_state = {}

        # Flags to control the program flow
        self.shutdown = False

        self.started_digging = False
        self.updating_gradients = 0
        self.current_batch_size = 0
        self.bot_just_died = False
        self.updates_running = False
        self.use_buffer = False
        self.socket_lock = False
        self.first_image = False
        self.repeat_images_in_a_row = 0
        self.max_repeat_images_in_a_row = 10
        self.action_length = 0
        self.repeated_dig_commands = 0
        self.should_help = False
        self.world_resetted = False


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

    def _init_sockets_and_clients(self):
        # Clients for non time critical operations, might have even been fine for time critical operations but in the
        # implementations were many latency spikes from different sources so most time critical operations are written
        # with TCP connections and I see no reason to change it
        self.http_post_client_to_mineflayer = Client()
        self.http_client_to_trainer = Client()

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

        #self.post_gradient_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.post_gradient_socket.connect((self.trainer_ip_address, self.trainer_tcp_port))

        self.socket_to_get_model = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_to_get_model.bind((self.socket_host, self.port_for_model))
        self.socket_to_get_model.listen()




    def _init_logging_variables(self):
        # Time Measurements (mainly for Debugging logging)
        self.last_time_since_image = time.time_ns()
        self.times_since_last_image = []
        self.times_to_upload_gradients = []
        self.times_to_calculate_gradients = []
        self.times_to_get_model_from_url = []
        self.times_to_get_model_from_file = []
        self.times_needed_to_send_get_model_request = []
        self.times_needed_to_update_state = []

        self.times_needed_to_send_get_state_request = []
        self.times_needed_to_send_action = []
        self.times_to_choose_action = []
        self.max_time_to_choose_action = 0
        self.times_needed_to_update_model = []
        self.digging_repetitions = []

        # More measurements for logging
        self.connections_lost = 0
        self.bot_deaths = 0
        self.state_counter = 0
        self.executed_actions = 0

        self.total_reward = 0
        self.total_reward_clipped = 0

        self.total_positive_reward = 0
        self.total_positive_reward_clipped = 0
        self.total_negative_reward = 0
        self.total_negative_reward_clipped = 0

        self.total_reused_images = 0
        self.failed_digging_count = 0

        # More debugging values
        self.current_upload_thread_counter = 0
        self.thread_counter_warning = 10
        self.upload_gradient_warning = 3

    def _init_models(self):
        self.img_dimensions = (self.model_parameter_config["image_width"], self.model_parameter_config["image_height"],
                               self.model_parameter_config["image_channels"])
        self.sequence_length = self.model_parameter_config["sequence_length"]
        self.batch_size = self.run_parameter_config["batch_size"]
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")

        print(self.device)


        self.newest_model = None
        self.newest_model_lock = False

        self.all_losses = {}
        self.Reward = Reward()

        self.get_model()



    def get_new_model(self):
        time_start = time.time_ns()

        # tell trainer that we want the newest model over our tcp port
        url = "http://" + str(self.trainer_ip_address) + ":" + str(self.trainer_port) + str(
            self.api_prefix_trainer) + "/get_model_with_tcp"

        try:
            response_raw = self.http_client_to_trainer.post(url, json={"ip": self.socket_host,
                                                                   "port": self.port_for_model,
                                                                       "current_episode_model_step": self.current_episode_model_step})
            response = pickle.loads(response_raw.content)
            if not response["new_model_available"]:
                print("waiting for new model to be ready")
                time.sleep(5)
                self.get_new_model()
                return

        except Exception as e:
            print("Failed getting response from trainer to get the new model")
            print(e)
            traceback.print_exc()
            return

        try:
            conn, addr = self.socket_to_get_model.accept()
            print('Socket connected')
            my_message = socket_functions.recv_msg(conn)
            self.load_model_from_package(my_message)

        except Exception as e:
            print(e)

        time_end = time.time_ns()
        time_needed = (time_end - time_start) / 1000 / 1000 / 1000
        print("Time needed to get newest Model:", str(time_needed) + "s")
        self.times_needed_to_update_model.append(time_needed)


    def load_model_from_package(self, package):
        model_data = pickle.loads(package)
        for param, new_param in zip(self.newest_model.parameters(), model_data["model_param"]):
            param.data = new_param.to(self.device)

        self.newest_model.update_relevant_parameters(model_data["extra_param"])
        print("Updated model")


    async def update(self, conn):
        if self.shutdown:
            return


        socket_functions.send_json({"type": "get", "start_time":(time.time_ns() / 1000 / 1000)}, conn)
        answer, error = socket_functions.listen_conn(conn=conn, message_target=socket_functions.process_msg_as_json_string, threaded=False, repeat=False)
        if self.shutdown:
            return

        if error:
            raise Exception

        answer_state = answer["state"]
        time_needed = answer["time_needed"]

        self.times_needed_to_send_get_state_request.append(time_needed)

        if len(answer_state.keys()) == 0:  # bot not ready yet
            return

        if not self.first_full_connection:
            self.get_new_model()
            print("updated model for the first episode")

        self.first_full_connection = True
        if answer_state["just_died"]:
            self.bot_just_died = True

        # if this happens this run might be dead, because usually there is enough buffer time to get the first image
        # Meaning if this condition happens something unaccounted went wrong in prismarine-viewer
        if not self.first_image:
            return

        self.action_length += 1
        self.state_counter += 1

        self.action_length += 1
        self.current_batch_size += 1

        answer_state["action_length"] = self.action_length
        self.failed_digging_count = answer_state["failed_digging_count"]
        self.action_length = 0
        self.extract_reward(answer_state, self.state_storage)
        self.previous_state = answer_state
        image_reused = self.state_storage.add_state(answer_state)

        if image_reused:
            self.total_reused_images += 1
            self.repeat_images_in_a_row += 1
            if self.repeat_images_in_a_row > self.max_repeat_images_in_a_row:
                critical_warning_message = "CRITICAL ERROR: The same image got reused " + str(self.max_repeat_images_in_a_row) + "times in a row, shutting Pod down so that it does not influence the model negatively"
                self.send_critical_warning_message(critical_warning_message)
                self.updates_running = False
                #self.system_shutdown()
                os._exit(1)
        else:
            self.repeat_images_in_a_row = 0


        if self.current_batch_size >= self.batch_size:
            if self.current_episode > self.max_episode_count:
                print("System should shutdown soon")
                self.shutdown = True


            print("sending clear message to mineflayer")
            url = "http://" + self.http_mineflayer_address + ":" + str(self.http_mineflayer_port) + str(
                self.api_prefix_mineflayer) + "/reset"
            try:
                response = self.http_post_client_to_mineflayer.post(url)
                print("mineflayer answered")


            except Exception as e:
                print("POST request " + str(url) + " failed\n" + str(e))
            self.world_resetted = True
            print("Batch full. Size:", str(self.current_batch_size))
            self.current_batch_size = 0
            block_name = answer_state["block_name"]
            forced_continued_digging = self.started_digging and (answer_state["should_dig"] and not answer_state["destroyed_block"] and answer_state[
                    "expected_dig_time"] < 5000 and answer_state["can_harvest_block"] and not answer_state[
                    "stopped_digging"] and self.Reward.is_goal_block(block_name)) and self.repeated_dig_commands <= 30
            self.state_storage.forced_continued_digging = forced_continued_digging

            for i in range(self.staggered_model_update_count):
                self.update_gradients(update_type="staggered")
                self.current_episode_model_step += 1
                self.get_new_model()

            self.current_episode += 1
            self.update_gradients(update_type="normal")
            self.current_episode_model_step += 1
            self.get_new_model()
            self.state_storage.reset()
            self.get_new_model()
            self.first_image = False
            self.previous_state = {}

            if self.shutdown:
                print("System shutdown initialized")
                self.system_shutdown()
            return


        action_chosen = False
        dig_block = answer_state["block_digging"]
        if self.repeated_dig_commands > 30:
            critical_warning = "CRITICAL ERROR, BOT WAS UNABLE TO DESTROY BLOCK, UNSURE WHY: "
            if dig_block:
                critical_warning += str(dig_block["name"])

            critical_warning += "   mining_progress: " + str(
                answer_state["mining_progress"]) + "    expected_dig_time: " + str(answer_state["expected_dig_time"])
            critical_warning += "   repeated_dig_commands: " + str(self.repeated_dig_commands)

            self.send_critical_warning_message(critical_warning)
            self.should_help = False
            self.digging_repetitions.append(self.repeated_dig_commands)
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
                if self.repeated_dig_commands > 0:
                    self.digging_repetitions.append(self.repeated_dig_commands)
                self.repeated_dig_commands = 0

        self.executed_actions += 1

        if not action_chosen:
            action_dict = self.predict_action(self.state_storage)
            action_dict["repeated"] = False

        self.state_storage.add_action(action_dict)
        socket_functions.send_json({"type": "action", "start_time": (time.time_ns() / 1000 / 1000), "action":action_dict}, conn)
        time_needed, error = socket_functions.listen_conn(conn, socket_functions.process_msg_as_json_string, repeat=False)
        if self.shutdown:
            return
        if error:
            raise Exception

        self.times_needed_to_send_action.append(time_needed)

    def send_critical_warning_message(self, critical_warning_message):
        url = "http://" + str(self.trainer_ip_address) + ":" + str(self.trainer_port) + str(
            self.api_prefix_trainer) + "/critical_warnings"

        try:
            response = self.http_client_to_trainer.post(url, json={"critical warning": critical_warning_message})
        except Exception as e:
            print("Sending critical warning failed (" + str(url) + ") " + str(e))
        print(critical_warning_message)

    def update_gradients(self, update_type="normal"):
        self.updating_gradients += 1

        if self.updating_gradients >= self.upload_gradient_warning:
            critical_warning_message = "CRITICAL ERROR, many gradients being uploaded at the same time (" + str(self.upload_gradient_warning) +"), network might be overloaded"
            self.send_critical_warning_message(critical_warning_message)

        if len(self.state_storage.actions) == 0:
            self.updating_gradients -= 1
            print("no states for gradient update")
            return

        try:
            start_time = time.time_ns()
            losses = self.newest_model.calculate_gradients(self.state_storage, update_type=update_type)
            end_time = time.time_ns()
            time_needed = (end_time - start_time) / 1000 / 1000
            self.times_to_calculate_gradients.append(time_needed)
        except Exception as e:
            self.updating_gradients -= 1
            critical_warning_message = "CRITICAL ERROR, calculation of loss failed, probably took too long and model changed during calculation"
            self.send_critical_warning_message(critical_warning_message + str(e))
            traceback.print_exc()
            return

        if losses is None:
            self.updating_gradients -= 1
            print("something went wrong in the gradient calculation")
            return


        for key in losses:
            if key in self.all_losses:
                self.all_losses[key].append(losses[key].item())
            else:
                self.all_losses[key] = [losses[key].item()]

        self.post_gradients(losses, update_type=update_type)
        self.updating_gradients -= 1

    def post_gradients(self, losses, update_type="normal"):
        start_time = time.time_ns()
        all_grads = []

        for parameter in self.newest_model.parameters():
            all_grads.append(parameter.grad)

        package = {"losses": losses, "gradients": all_grads, "update_type":update_type,
                   "start_time": start_time, "mission_type": self.mission_type,
                   "destruction_factor": self.destruction_factor}

        while self.socket_lock:
            time.sleep(1)
            print("waiting for previous upload to finish")

        self.socket_lock = True
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.trainer_ip_address, self.trainer_tcp_port))
        #socket_functions.send_message(package, sock=self.post_gradient_socket)
        socket_functions.send_message(package, sock=sock)
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        self.socket_lock = False


        end_time = time.time_ns()
        time_needed = (end_time - start_time) / 1000 / 1000 / 1000
        self.times_to_upload_gradients.append(time_needed)

    def predict_action(self, state_storage_to_use) -> dict:
        state = state_storage_to_use.get_last_game_state_without_action()
        image_time_needed = (time.time_ns() - self.last_time_since_image) / 1000 / 1000
        self.times_since_last_image.append(image_time_needed)

        action_dict, time_needed = self.newest_model.choose_action(state)
        self.times_to_choose_action.append(time_needed)
        self.started_digging = action_dict["attack"]

        if time_needed > self.max_time_to_choose_action:
            self.max_time_to_choose_action = time_needed

        return action_dict

    def extract_reward(self, new_state:dict, state_storage_to_use:StateStorage):
        goal = self.Reward.get_goal_as_numerical_list()
        state_storage_to_use.add_goal(goal)

        clipped_reward, un_clipped_reward = self.Reward.extract_reward(self.previous_state, new_state)

        # The first state of the episode is empty and as such does not have a reward
        # only with the second state we can calculate a reward
        if clipped_reward is None:
            return

        state_storage_to_use.add_rewards(un_clipped_reward, clipped_reward)

        self.total_reward += un_clipped_reward
        self.total_reward_clipped += clipped_reward

        if un_clipped_reward > 0:
            self.total_positive_reward += un_clipped_reward
        else:
            self.total_negative_reward += un_clipped_reward

        if clipped_reward > 0:
            self.total_positive_reward_clipped += clipped_reward
        else:
            self.total_negative_reward_clipped += clipped_reward


    def system_shutdown(self):
        print("sending shutdown message to mineflayer")
        url = "http://" + self.http_mineflayer_address + ":" + str(self.http_mineflayer_port) + str(
            self.api_prefix_mineflayer) + "/end"
        try:
            response = self.http_post_client_to_mineflayer.post(url)
            print("mineflayer shut down")

        except Exception as e:
            print("POST request " + str(url) + " failed\n" + str(e))

        print("wait for gradient calculations to end")
        while self.updating_gradients > 0:
            time.sleep(1)
            print("still waiting", str(self.updating_gradients))

        print("sending total reward for logging")
        avg_losses = {}
        for key in self.all_losses:
            avg_losses[key] = np.mean(self.all_losses[key])

        blocks_collected, blocks_collected_clipped = self.Reward.get_blocks_collected()
        blocks_destroyed = self.Reward.get_blocks_destroyed()


        print("Total reused images:", str(self.total_reused_images))
        print("Total images:", str(self.state_counter))

        url = "http://" + str(self.trainer_ip_address) + ":" + str(self.trainer_port) + str(
            self.api_prefix_trainer) + "/post_logging_values"
        try:
            reward_per_step = 0
            reward_per_step_clipped = 0
            positive_reward_per_step = 0
            positive_reward_per_step_clipped = 0
            negative_reward_per_step = 0
            negative_reward_per_step_clipped = 0


            if self.state_counter > 0:
                reward_per_step = float(self.total_reward)/float(self.state_counter)
                reward_per_step_clipped = float(self.total_reward_clipped)/float(self.state_counter)
                positive_reward_per_step = float(self.total_positive_reward) / float(self.state_counter)
                positive_reward_per_step_clipped = float(self.total_positive_reward_clipped) / float(self.state_counter)
                negative_reward_per_step = float(self.total_negative_reward) / float(self.state_counter)
                negative_reward_per_step_clipped = float(self.total_negative_reward_clipped) / float(self.state_counter)



            print("send logging Values")
            response = self.http_client_to_trainer.post(url, json={"mission_type": self.mission_type, "destruction_factor": self.destruction_factor,
                "reward_per_state": reward_per_step, "reward_per_state_clipped": reward_per_step_clipped,
                                                                   "avg_losses": avg_losses,
                                                                   "positive_reward_per_step": positive_reward_per_step, "negative_reward_per_step":negative_reward_per_step,
                                                                   "positive_reward_per_step_clipped": positive_reward_per_step_clipped,
                                                                   "negative_reward_per_step_clipped": negative_reward_per_step_clipped,
                                                                   "pod_name": os.environ.get("pod_name"), "reused_images": self.total_reused_images,
                                             "state_counter":self.failed_digging_count,
                                                                   "action_counter": self.executed_actions,
                                             "times_to_calculate_gradients": self.times_to_calculate_gradients,
                                             "times_to_upload_gradients": self.times_to_upload_gradients,
                                             "max_time_to_choose_action": self.max_time_to_choose_action,
                                             "times_to_get_model_from_url": self.times_to_get_model_from_url,
                                             "times_to_get_model_from_file":self.times_to_get_model_from_file,
                                             "times_needed_to_send_get_model_request": self.times_needed_to_send_get_model_request,

                                           "times_needed_to_send_action": self.times_needed_to_send_action,
                                           "times_needed_to_send_get_state_request":self.times_needed_to_send_get_state_request,
                                           "bot_deaths": self.bot_deaths,
                                           "connections_lost": self.connections_lost,
                                           "times_since_last_image": self.times_since_last_image,
                                            "times_to_choose_action": self.times_to_choose_action,
                                            "times_needed_to_update_model": self.times_needed_to_update_model,
                                            "most_valuable_item_gotten": self.Reward.most_valuable_item_gotten,
                                            "blocks_collected": blocks_collected,
                                            "blocks_collected_clipped": blocks_collected_clipped,
                                            "blocks_destroyed": blocks_destroyed,
                                            "digging_repetitions":self.digging_repetitions,
                                            "seed": self.seed,
                                            "times_needed_to_update_state":self.times_needed_to_update_state})
        except Exception as e:
            print("failure to send logging values")
            print(e)
            traceback.print_exc()
            print("retrying in 5 seconds")
            time.sleep(5)
            self.system_shutdown()

        time.sleep(2) # making sure mineflayer had time to shut down
        os._exit(0)


    def get_mission(self):
        url = "http://" + str(self.trainer_ip_address) + ":" + str(self.trainer_port) + str(
            self.api_prefix_trainer) + "/get_mission"
        response_raw = self.http_client_to_trainer.post(url, json={"mission_type": self.mission_type})
        try:
            mission_json = pickle.loads(response_raw.content)
            mission_path = mission_json["mission_path"]
            self.Reward.define_mission(mission_path, destruction_weight=self.destruction_factor)
            print("Reward function successfully loaded")

        except Exception as e:
            print("POST request " + str(url) + " failed\n" + str(e))
            print(e)
            traceback.print_exc()

    def get_model(self):
        start_time = time.time_ns()

        url = "http://" + str(self.trainer_ip_address) + ":" + str(self.trainer_port) + str(
            self.api_prefix_trainer) + "/get_model"
        response_raw = self.http_client_to_trainer.get(url)


        try:
            model_parameter = pickle.loads(response_raw.content)

            end_time = time.time_ns()
            time_needed = (end_time - start_time) / 1000 / 1000 / 1000

            self.times_to_get_model_from_url.append(time_needed)
            time_start = model_parameter["time_start"]
            time_needed = (end_time - time_start) / 1000 / 1000 / 1000

            self.times_needed_to_send_get_model_request.append(time_needed)

            print("get mission")
            self.get_mission()
            print("create Model")
            self.state_buffer_count = model_parameter["state_buffer_count"]
            if self.state_buffer_count < 0:
                self.state_buffer_count = 0

            self.state_storage: StateStorage = StateStorage(self.img_dimensions, self.Reward.goal_items)

            self.newest_model = model_parameter["full_model"]
            self.newest_model = self.newest_model.to(self.device)
            print("Got model")
            #self.get_new_model()


        except Exception as e:
            print("POST request " + str(url) + " failed\n" + str(e))
            print(e)
            traceback.print_exc()
        print("Base models successfully loaded")

    def create_run_directory(self, retry_count):
        print("create directory:", str(self.save_path))
        if not os.path.exists(self.save_path):
            os.mkdir(str(self.save_path))
            os.mkdir(str(self.state_save_path))
        else:
            self.save_path = self.state_save_directory  +  "/" + str(self.session_id) + "nr_" + str(retry_count) + "/"
            self.state_save_path = self.save_path + "states/"
            print("Directory with the session ID already exists, trying next one (" + str(self.save_path) +")")
            retry_count += 1
            self.create_run_directory(retry_count)

    async def update_loop(self):
        print("Waiting for Mineflayer to start")
        conn, addr = self.socket_to_mineflayer.accept()
        self.test_start_time = time.time_ns()
        while not self.shutdown:
            update_step_start_time = time.time_ns()
            try:
                self.updates_running = True
                await self.update(conn)
                self.updates_running = False
            except Exception as e:
                print("Error happened in main Loop, reconnecting socket")
                print(e)
                traceback.print_exc()
                if self.first_full_connection:
                    self.connections_lost += 1
                self.updates_running = False
                conn, addr = self.socket_to_mineflayer.accept()
                continue
            update_step_end_time = time.time_ns()
            update_step_time_needed = (update_step_end_time - update_step_start_time) / 1000 / 1000
            if not self.world_resetted:
                self.times_needed_to_update_state.append(update_step_time_needed)
            else:
                self.world_resetted = False

            #wait_time = self.timestep_delay - update_step_time_needed_s
            time.sleep(self.timestep_delay)

            if self.bot_just_died:
                self.bot_just_died = False
                self.bot_deaths += 1
            #    time.sleep(self.timestep_delay)
            #elif wait_time > 0:
            #    time.sleep(wait_time)
            #else:
            #    print("update needed too long for:", str(wait_time) + "s")

        self.updates_running = False

    def listen_viewer_tcp(self):
        while not self.shutdown:
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

                if self.shutdown:
                    break

                image = cv2.imdecode(np.fromstring(message, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                self.first_image = True

                #cv2.imshow('image', image)
                #cv2.waitKey(1)
                self.state_storage.add_img(image)

                self.last_time_since_image = time.time_ns()


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

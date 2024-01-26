import time
from datetime import datetime
import json
import os
from copy import copy, deepcopy

import uvicorn
from fastapi import Body, FastAPI, Response, APIRouter
from httpx import AsyncClient
from starlette.middleware.cors import CORSMiddleware

import socket

import threading
import pickle

from kubernetes import client, config, utils
import yaml

import torch
import wandb
import numpy as np

from utils import socket_functions
from utils.custom_rmsprop import customRMSProp
from utils.WandbLogger import WandbLogger
from utils.Reward import Reward
from Models.DFP import DFP
from Models.A3C import A3C
from Models.DDFP import DDFP
from Models.LEHDDFP import LEHDDFP
from Models.ACDFP import ACDFP

from Models.EACDFP import EACDFP
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
import sys

def handle_exception(exc_type, exc_value, exc_traceback):
    print(exc_type)
    print(exc_value)
    print(exc_traceback)

sys.excepthook = handle_exception


wandb_key = os.getenv("WANDB_LOGIN_KEY")
if not wandb_key is None:
    try:
        wandb.login(key=str(wandb_key))
    except Exception as e:
        print(e)
        print("failed to login with wandb")

app = FastAPI()
# constraints for rest api
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])



class Trainer:

    def _init_configs(self):
        with open('../Configs/model-parameters.json') as f:
            self.model_parameter_config = json.load(f)
        with open('../Configs/trainer-parameters.json') as f:
            self.trainer_config = json.load(f)
        with open('../Configs/run-parameters.json') as f:
            self.run_config = json.load(f)

        self.seeds_to_use = np.loadtxt("../Configs/forest-seeds.txt")

        self.in_cluster = False
        try:
            config.load_incluster_config()
            self.k8s_client = client.api_client.ApiClient()
            self.in_cluster = True

        except Exception as e:
            print(e)

    def _init_generation_variables(self):
        self.model_generation = 0
        self.failed_pods_in_generation = 0
        self.rewards_steps = 0
        self.pods_started = 0
        self.pods_finished = 0
        self.pods_running = 0
        self.started_pod_names = []
        self.best_avg_reward_generation = 0
        self.step_counter = 0
        self.max_simultaneous_pods = self.trainer_config["max_simultaneous_pods"]

        self.generations_per_checkpoint = 10

        #self.model_update_method = self.trainer_config["update_method"]
        self.gradients_needed_for_step = self.trainer_config["gradients_needed_for_step"]
        self.model_update_method = os.environ.get("model_update_method")
        if self.model_update_method is None:
            self.model_update_method = "step"

        if self.model_update_method == "step":
            self.gradients_needed_for_step = 1



        self.max_pods = self.trainer_config["max_pods"]

        self.current_episode_model_step = 1
        self.recently_updated = True

        self.rewards_of_generation_per_state = {}
        self.rewards_of_generation_per_state_clipped = {}

        self.received_states_of_generation = {}
        self.positive_rewards_of_generation_per_state = {}
        self.negative_rewards_of_generation_per_state = {}

        self.losses_of_generation = {}


        self.emergency_pod_deletion_thread = None
        self.failed_received_msg_this_generation = 0

    def _init_model_parameters(self):
        self.model_type = os.environ.get("model_type")
        if self.model_type is None:
            self.model_type = "A3C"

        self.load_from_checkpoint = os.environ.get("load_from_checkpoint")
        if self.load_from_checkpoint is None:
            self.load_from_checkpoint = "False"

        if self.load_from_checkpoint == "True":
            self.load_from_checkpoint = True
        else:
            self.load_from_checkpoint = False


        self.checkpoint_path = os.environ.get("checkpoint_path")
        if self.checkpoint_path is None:
            self.checkpoint_path = ""
            self.load_from_checkpoint = False

        self.session_id = self.model_type + "_" + datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        if self.load_from_checkpoint:
            self.session_id = "TMP" + self.session_id

        self.dirt_mission_path = self.model_parameter_config["dirt_mission_path"]
        self.wood_mission_path = self.model_parameter_config["wood_mission_path"]

        #self.max_grad_norm = self.trainer_config["max_grad_norm"]
        self.max_grad_norm = os.environ.get("max_grad_norm")
        if self.max_grad_norm is None:
            self.max_grad_norm = 100
        self.max_grad_norm = int(self.max_grad_norm)

        self.rsm_alpha = os.environ.get("rsm_alpha")
        if self.rsm_alpha is None:
            self.rsm_alpha = "0.99"
        self.rsm_alpha = float(self.rsm_alpha)

        self.rsm_beta = os.environ.get("rsm_beta")
        if self.rsm_beta is None:
            self.rsm_beta = "0.99"
        self.rsm_beta = float(self.rsm_beta)


        self.clipped_reward = os.environ.get("clipped_reward")
        if self.clipped_reward is None:
            self.clipped_reward = "True"

        if self.clipped_reward == "True":
            self.clipped_reward = True
        else:
            self.clipped_reward = False

        self.wait_for_new_model = os.environ.get("wait_for_new_model")
        if self.wait_for_new_model is None:
            self.wait_for_new_model = "True"

        if self.wait_for_new_model == "False":
            self.wait_for_new_model = False
        else:
            self.wait_for_new_model = True


        self.last_saved_models = []

        self.gradients_to_update = []
        self.gradients_to_update_buffer = []
        self.use_buffer = False
        self.updating_model = False
        self.model_lock = False

        self.model_step = 0
        self.end_learning_rate_factor = self.model_parameter_config["end_learning_rate_factor"]
        self.expected_batches = int((self.run_config["run_time_seconds"] * self.max_pods)/(self.run_config["timestep_delay"] * self.run_config["batch_size"]))
        self.batch_size = self.run_config["batch_size"]
        print("Expected batches:", str(self.expected_batches))

        self.learning_rate = os.environ.get("learning_rate")
        if self.learning_rate is None:
            self.learning_rate = "0.001"
        self.learning_rate = float(self.learning_rate)

        #self.learning_rate = self.model_parameter_config["learning_rate"]


        if self.in_cluster:
            self.model_directory_path = self.trainer_config["models_path_kubernetes"]
        else:
            self.model_directory_path = self.trainer_config["models_path_local"]
        #self.model_directory_path = self.trainer_config["models_path_local"]

        if not os.path.exists(self.model_directory_path):
            os.mkdir(self.model_directory_path)
        self.model_path = self.model_directory_path + "/" + self.session_id
        self.best_model_path = self.model_path
        self.model_name = self.trainer_config["model_name"]

        os.mkdir(self.model_path)
        os.mkdir(self.model_path + "/best")


    def _init_model(self):
        print("create model")
        #self.model = DFP(self.model_parameter_config)


        if self.model_type == "A3C":
            self.model = A3C(self.model_parameter_config)
        elif self.model_type == "DFP":
            self.model = DFP(self.model_parameter_config)
        elif self.model_type == "DDFP":
            self.model = DDFP(self.model_parameter_config)
        elif self.model_type == "LEHDDFP":
            self.model = LEHDDFP(self.model_parameter_config)
        elif self.model_type == "ACDFP":
            self.model = ACDFP(self.model_parameter_config)
        elif self.model_type == "EACDFP":
            self.model = EACDFP(self.model_parameter_config)
        else:
            print("no valid model was submitted, stopping program")
            os._exit(1)

        self.model.clipped_reward = self.clipped_reward
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        #self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
        #                                     alpha=self.rsm_alpha, eps=1e-08)
        self.optimizer = customRMSProp(self.model.parameters(), lr=self.learning_rate,
                                             alpha=self.rsm_alpha, beta=self.rsm_beta, eps=1e-08)

        print("model created")


        self.base_model_path = str(self.model_path) + "/base_" + self.model_name + ".pt"
        torch.save(self.model, self.base_model_path)
        print("saved entire model at:", str(self.base_model_path))

    def _init_clients(self):
        self.api_prefix: str = '/trainer'  # url prefix for http requests
        self.uvicorn_port = self.trainer_config["port"]
        self.socket_connections = {}
        self.trainer_ip_addr = socket.gethostbyname(socket.gethostname())
        self.http_client_to_agents = AsyncClient()
        self.router = APIRouter()

    def __init__(self):
        self._init_configs()
        self._init_generation_variables()
        self._init_model_parameters()
        self._init_model()
        self._init_clients()

        self.debug_mode = self.trainer_config["Debug"]
        self.base_port = self.run_config["port_for_models"]
        self.current_base_port = self.base_port

        self.run_name = os.environ.get("run_name")
        if self.run_name is None:
            self.run_name = "Test"

        if not self.in_cluster:
            self.gradients_needed_for_step = 1

        wandb_name = self.run_name + str(self.model_type) + str(self.session_id)
        if self.load_from_checkpoint:
            wandb_name = "2nd" + wandb_name

        run = wandb.init(
            project="minecraft-rl",
            notes="First Dirt Run",
            tags=[str(self.model_type), "minecraft", "rl", "Wood", "mixed"],
            name= str(wandb_name)
        )

        if self.load_from_checkpoint:
            self.load_checkpoint(self.checkpoint_path)
            print("model loaded from checkpoint:", str(self.checkpoint_path))
        else:
            self.save_checkpoint()


        @self.router.post(self.api_prefix + "/get_model_with_tcp")
        def get_model_with_tpc(connection_args=Body(...)):

            ip = connection_args["ip"]
            port = connection_args["port"]

            time_start = time.time_ns()
            answer_json = {"time_start":time_start}
            if (self.current_episode_model_step == connection_args["current_episode_model_step"]
                and not self.updating_model and self.recently_updated) or not self.wait_for_new_model:

                answer_json["new_model_available"] = True
                new_thread = threading.Thread(target=self.send_model_over_tcp, args=[ip, port])
                new_thread.start()
            else:
                answer_json["new_model_available"] = False

            pickle_object = pickle.dumps(answer_json)
            return Response(content=pickle_object)


        @self.router.get(self.api_prefix + "/get_model")
        async def get_model():
            time_start = time.time_ns()

            answer_json = {"full_model_path": self.base_model_path,
                           "full_model": self.model,
                    "time_start":time_start,
                    "state_buffer_count": self.model.state_buffer_count
                    }

            pickle_object = pickle.dumps(answer_json)
            return Response(content=pickle_object)

        @self.router.post(self.api_prefix + "/get_mission")
        async def get_mission(mission_type_json=Body(...)):
            mission_type = mission_type_json["mission_type"]
            if mission_type == "dirt" or mission_type == "zero-shot":
                mission_path = self.dirt_mission_path
            else:
                mission_path = self.wood_mission_path

            answer_json = {"mission_path": mission_path}

            pickle_object = pickle.dumps(answer_json)
            return Response(content=pickle_object)

        @self.router.post(self.api_prefix + "/post_logging_values")
        def log_run_values(log_values=Body(...)):
            new_thread = threading.Thread(target=self.process_logging_values, args=[log_values])
            new_thread.start()
            return Response(content=None)

        @self.router.post(self.api_prefix + "/critical_warnings")
        def log_critical_warning(critical_warning=Body(...)):
            print("got the following critical warning from a run:", str(critical_warning))
            return



    def process_logging_values(self, log_values):
        mission_type = log_values["mission_type"]
        destruction_factor = format(log_values["destruction_factor"], ".1f")

        wandbLogger = WandbLogger(mission_type, destruction_factor)

        avg_losses = log_values["avg_losses"]

        for key in avg_losses:
            wandbLogger.add_logging_value("epsiodic avg " + str(key), avg_losses[key])

        # tons of single values to be logged. Mostly used for debugging and checking if all runs are running smoothly (like received states and reused images)
        wandbLogger.add_logging_value(" episodic reward per step", log_values["reward_per_state"])
        wandbLogger.add_logging_value(" episodic reward per step clipped", log_values["reward_per_state_clipped"])
        wandbLogger.add_logging_value(" positive reward per step", log_values["positive_reward_per_step"])
        wandbLogger.add_logging_value(" positive reward per step clipped", log_values["positive_reward_per_step_clipped"])
        wandbLogger.add_logging_value(" negative reward per step", log_values["negative_reward_per_step"])
        wandbLogger.add_logging_value(" negative reward per step clipped", log_values["negative_reward_per_step_clipped"])
        wandbLogger.add_logging_value(" reused images", log_values["reused_images"])
        wandbLogger.add_logging_value(" received states", log_values["state_counter"])
        wandbLogger.add_logging_value(" action counter", log_values["action_counter"])
        wandbLogger.add_logging_value(" bot deaths", log_values["bot_deaths"])
        wandbLogger.add_logging_value(" most valuable item gotten", log_values["most_valuable_item_gotten"])
        wandbLogger.add_logging_value(" seed", log_values["seed"])

        for block_collected, block_collected_clipped, blocks_destroyed in zip(log_values["blocks_collected"], log_values["blocks_collected_clipped"], log_values["blocks_destroyed"]):
            wandbLogger.add_logging_value(str(block_collected) + " collected", log_values["blocks_collected"][block_collected])
            wandbLogger.add_logging_value(str(block_collected_clipped) + " collected clipped", log_values["blocks_collected_clipped"][block_collected_clipped])
            wandbLogger.add_logging_value(str(blocks_destroyed) + " destroyed", log_values["blocks_destroyed"][blocks_destroyed])

        # debug values do not need to be further seperated and this mode should only be started if the environment needs to be debugged
        # in which case the readability of the data is not that important
        if self.debug_mode:
            wandb.log({"connections lost": log_values["connections_lost"],
                       "max time to choose action in milliseconds": log_values["max_time_to_choose_action"]})

            for digging_repetition in log_values["digging_repetitions"]:
                wandb.log({"digging repetitions": digging_repetition})

            for time_to_upload_gradient, time_to_calculate_gradient, time_needed_to_send_get_model_request in zip(
                    log_values["times_to_upload_gradients"], log_values["times_to_calculate_gradients"],
                    log_values["times_needed_to_send_get_model_request"]):
                wandb.log({"times to upload gradients in seconds": time_to_upload_gradient,
                           "times to calculate gradients in milliseconds": time_to_calculate_gradient})

            for time_needed_to_send_get_model_request, time_to_get_model_from_url in zip(
                    log_values["times_needed_to_send_get_model_request"], log_values["times_to_get_model_from_url"]):
                wandb.log({"times needed to send get model request": time_needed_to_send_get_model_request,
                           "times to get model from url": time_to_get_model_from_url})

            for time_needed_to_update_model in log_values["times_needed_to_update_model"]:
                wandb.log({"times needed to update model": time_needed_to_update_model})

            for time_needed_to_update_state in log_values["times_needed_to_update_state"]:
                wandb.log({"times needed to update state": time_needed_to_update_state})

            for time_needed_to_send_action, time_needed_to_send_get_state_request, time_to_choose_action in zip(
                    log_values["times_needed_to_send_action"], log_values["times_needed_to_send_get_state_request"],
                    log_values["times_to_choose_action"]):
                wandb.log({"times needed to send action": time_needed_to_send_action,
                           "times needed to send get state request": time_needed_to_send_get_state_request,
                           "times to choose action": time_to_choose_action})

            for time_since_last_image in log_values["times_since_last_image"]:
                wandb.log({"times since last image update": time_since_last_image})

        wandb.log(wandbLogger.get_logging_values())

        self.add_value_to_generation_dict(self.rewards_of_generation_per_state_clipped, mission_type, destruction_factor, log_values["reward_per_state"])
        self.add_value_to_generation_dict(self.rewards_of_generation_per_state, mission_type, destruction_factor, log_values["reward_per_state_clipped"])
        self.add_value_to_generation_dict(self.received_states_of_generation, mission_type, destruction_factor, log_values["state_counter"])
        self.add_value_to_generation_dict(self.positive_rewards_of_generation_per_state, mission_type, destruction_factor, log_values["positive_reward_per_step"])
        self.add_value_to_generation_dict(self.negative_rewards_of_generation_per_state, mission_type, destruction_factor, log_values["negative_reward_per_step"])

        pod_name = log_values["pod_name"]
        shutdown_thread = threading.Thread(target=self.shutdown_pod, args=[pod_name])
        shutdown_thread.start()


    def add_value_to_generation_dict(self, generation_dict, mission_type, destruction_factor, value):
        if not mission_type in generation_dict:
            generation_dict[mission_type] = {}

        if not destruction_factor in generation_dict[mission_type].keys():
            generation_dict[mission_type][destruction_factor] = [value]
        else:
            generation_dict[mission_type][destruction_factor].append(value)


    def send_model_over_tcp(self, target_ip, target_port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((target_ip, target_port))
        start_time = time.time_ns()

        # Do not send parameters during an update to ensure they do not overwrite each other
        while self.updating_model:
            time.sleep(0.05)

        self.model_lock = True
        model_param = []

        for param in self.model.parameters():
            model_param.append(deepcopy(param))
        self.model_lock = False

        msg = {"model_param": model_param, "extra_param":self.model.get_relevant_parameters()}
        socket_functions.send_message(msg, sock=sock)

        end_time = time.time_ns()
        time_needed = (end_time - start_time) / 1000 / 1000
        #print("Time needed to send Model over TCP:", str(time_needed) + "ms")
        if self.debug_mode:
            wandb.log({"Time needed to send Model as TCP message": time_needed})

        sock.shutdown(socket.SHUT_RDWR)
        sock.close()

    def process_gradient_package(self, package_bytes):
        try:
            package = pickle.loads(package_bytes)
        except Exception as e:
            self.failed_received_msg_this_generation += 1
            print("Failed message len:", str(len(package_bytes)))
            print(e)
            return
        # for syncing the models. Only update the step counter after a normal update (the last update that gets executed)
        self.step_counter += 1
        if self.step_counter >= self.max_simultaneous_pods:
            self.step_counter = 0
            self.current_episode_model_step += 1
            if len(self.gradients_to_update_buffer) > 0 or len(self.gradients_to_update) > 0:
                self.recently_updated = False

        gradients = package["gradients"]
        self.rewards_steps += 1

        losses = package["losses"]
        time_start = package["start_time"]
        time_end = time.time_ns()
        time_needed = (time_end - time_start) / 1000 / 1000
        if self.debug_mode:
            wandb.log({"time needed to fully send gradients over TCP": time_needed})

        mission_type = package["mission_type"]
        destruction_factor = format(package["destruction_factor"], ".1f")

        if package["update_type"] == "normal":
            wandbLogger = WandbLogger(mission_type, destruction_factor)
            for key in losses.keys():
                loss = losses[key].item()
                wandbLogger.add_logging_value(str(key), loss)

                if not mission_type in self.losses_of_generation.keys():
                    self.losses_of_generation[mission_type] = {}

                if not destruction_factor in self.losses_of_generation[mission_type].keys():
                    self.losses_of_generation[mission_type][destruction_factor] = {}

                if not key in self.losses_of_generation[mission_type][destruction_factor].keys():
                    self.losses_of_generation[mission_type][destruction_factor][key] = [loss]
                else:
                    self.losses_of_generation[mission_type][destruction_factor][key].append(loss)

            wandb.log(wandbLogger.get_logging_values())

        if package["mission_type"] == "zero-shot":
            return

        if self.use_buffer:
            self.gradients_to_update_buffer.append(gradients)
        else:
            self.gradients_to_update.append(gradients)

        if not self.updating_model and (len(self.gradients_to_update) >= self.gradients_needed_for_step or
                                        len(self.gradients_to_update_buffer) >= self.gradients_needed_for_step):
            self.updating_model = True
            new_thread = threading.Thread(target=self.update_model)
            new_thread.start()

    def training_thread(self):
        if self.in_cluster:

            # Create TCP connections to listen to for the gradients of each pod
            for i in range(self.max_simultaneous_pods):
                hostname = socket.gethostname()
                socket_host = socket.gethostbyname(hostname)
                new_tcp_thread = threading.Thread(target=self.listen_tcp_for_gradients, args=[socket_host, self.base_port + i * 2])
                new_tcp_thread.start()

            self.create_pods()
        else:
            # Create TCP connections to listen to for the local instance of the agent (manually started)
            print("probably not in Kubernetes, you now need to start the other programs manually")
            if not os.path.exists(self.model_directory_path):
                os.mkdir(self.model_directory_path)
            self.max_simultaneous_pods = 1

            listen_thread = threading.Thread(target=self.listen_tcp_for_gradients, args=["localhost", self.base_port])
            listen_thread.start()


    def train(self):
        training_thread = threading.Thread(target=self.training_thread)
        training_thread.start()
        uvicorn.run(app, host="0.0.0.0", port=self.uvicorn_port, log_level="critical")

    def reset_values_of_generation(self):
        self.losses_of_generation = {}

        self.rewards_of_generation_per_state = {}
        self.rewards_of_generation_per_state_clipped = {}
        self.received_states_of_generation = {}
        self.positive_rewards_of_generation_per_state = {}
        self.negative_rewards_of_generation_per_state = {}
        self.failed_received_msg_this_generation = 0
        self.failed_pods_in_generation = 0

    def save_checkpoint(self):
        self.current_model_path = self.save_model()
        important_data = {"pods_started": self.pods_started,
                          "reward_steps": self.rewards_steps,
                          "model_path": self.current_model_path,
                          "pods_finished": self.pods_finished}
        print("Saved Generation model at:" , str(self.current_model_path))
        print("Gradients/rewards gotten:", str(self.rewards_steps))
        json_save_path = str(self.model_path) + "/" + self.model_name + "_step_" + str(self.model_step) + ".json"
        print("save json at:", str(json_save_path))
        with open(str(json_save_path), "w") as f:
            json.dump(important_data, f)

    def load_checkpoint(self, checkpoint_path):
        with open(str(checkpoint_path)) as f:
            checkpoint_json = json.load(f)
        model_path = checkpoint_json["model_path"]
        self.pods_started = checkpoint_json["pods_started"]
        self.rewards_steps = checkpoint_json["reward_steps"]
        self.pods_finished = checkpoint_json["pods_finished"]
        print("starting training from gradient step:", str(self.rewards_steps))

        model_checkpoint = torch.load(model_path)
        self.model.load_state_dict(model_checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(model_checkpoint["model_optimizer_dict"])
        self.model.update_relevant_parameters(model_checkpoint["model_specific_parameter"])

    def finish_generation(self):
        if self.pods_running < 0:
            print("some error happened when scheduling pods")
            self.pods_running = 0

        if self.model_generation % self.generations_per_checkpoint == 0:
            self.save_checkpoint()

        # if we got any gradients over we will delete them to avoid "hidden" generations (when a pod fails but
        # the other pods were able to send gradients). It also helps to better sync the models
        self.gradients_to_update_buffer = []
        self.gradients_to_update = []

        self.current_episode_model_step = 1
        self.recently_updated = True
        self.step_counter = 0

        wandb.log({"failed pods in generation:": self.failed_pods_in_generation})
        if self.debug_mode:
            wandb.log({"failed loading of gradients (tcp) in this generation": self.failed_received_msg_this_generation})


        # individual pods
        for mission_key in self.rewards_of_generation_per_state.keys():
            for destruction_factor_key in self.rewards_of_generation_per_state[mission_key].keys():
                wandbLogger = WandbLogger(mission_key, destruction_factor_key)
                for loss_key in self.losses_of_generation[mission_key][destruction_factor_key].keys():
                    avg_loss = 0
                    for loss in self.losses_of_generation[mission_key][destruction_factor_key][loss_key]:
                        avg_loss += loss
                    avg_loss = avg_loss / len(self.losses_of_generation[mission_key][destruction_factor_key][loss_key])
                    wandbLogger.add_logging_value_single("avg " + str(loss_key) + " of generation", avg_loss)

                generation_reward = sum(self.rewards_of_generation_per_state[mission_key][destruction_factor_key]) / len(self.rewards_of_generation_per_state[mission_key][destruction_factor_key])
                generation_reward_clipped = sum(self.rewards_of_generation_per_state_clipped[mission_key][destruction_factor_key]) / len(self.rewards_of_generation_per_state_clipped[mission_key][destruction_factor_key])
                generation_positive_reward = sum(self.positive_rewards_of_generation_per_state[mission_key][destruction_factor_key]) / len(self.positive_rewards_of_generation_per_state[mission_key][destruction_factor_key])
                generation_negative_reward = sum(self.negative_rewards_of_generation_per_state[mission_key][destruction_factor_key]) / len(self.negative_rewards_of_generation_per_state[mission_key][destruction_factor_key])

                wandbLogger.add_logging_value_single("avg reward of generation per state", generation_reward)
                wandbLogger.add_logging_value_single("avg reward of generation per state clipped", generation_reward_clipped)
                wandbLogger.add_logging_value_single("avg positive reward of generation per state", generation_positive_reward)
                wandbLogger.add_logging_value_single("avg negative reward of generation per state", generation_negative_reward)
                wandb.log(wandbLogger.get_logging_values())

        # per mission type (dirt, wood, zero-shot)
        for mission_key in self.rewards_of_generation_per_state.keys():
            pod_counter = 0
            generation_reward = 0
            generation_reward_clipped = 0
            generation_positive_reward = 0
            generation_negative_reward = 0

            avg_losses = {}
            wandbLogger = WandbLogger(mission_key)

            for destruction_factor_key in self.rewards_of_generation_per_state[mission_key].keys():
                for loss_key in self.losses_of_generation[mission_key][destruction_factor_key].keys():
                    avg_loss = 0
                    for loss in self.losses_of_generation[mission_key][destruction_factor_key][loss_key]:
                        avg_loss += loss
                    avg_loss = avg_loss / len(self.losses_of_generation[mission_key][destruction_factor_key][loss_key])
                    if not loss_key in avg_losses.keys():
                        avg_losses[loss_key] = [avg_loss]
                    else:
                        avg_losses[loss_key].append(avg_loss)

                pod_counter += 1
                generation_reward += sum(
                    self.rewards_of_generation_per_state[mission_key][destruction_factor_key]) / len(
                    self.rewards_of_generation_per_state[mission_key][destruction_factor_key])
                generation_reward_clipped += sum(
                    self.rewards_of_generation_per_state_clipped[mission_key][destruction_factor_key]) / len(
                    self.rewards_of_generation_per_state_clipped[mission_key][destruction_factor_key])
                generation_positive_reward += sum(
                    self.positive_rewards_of_generation_per_state[mission_key][destruction_factor_key]) / len(
                    self.positive_rewards_of_generation_per_state[mission_key][destruction_factor_key])
                generation_negative_reward += sum(
                    self.negative_rewards_of_generation_per_state[mission_key][destruction_factor_key]) / len(
                    self.negative_rewards_of_generation_per_state[mission_key][destruction_factor_key])

            if pod_counter <= 0:
                print("Missing values for logging, somehow an entry exists for the mission but no values were logged")
            else:
                generation_reward /= pod_counter
                generation_reward_clipped /= pod_counter
                generation_positive_reward /= pod_counter
                generation_negative_reward /= pod_counter

            wandbLogger.add_logging_value_single("avg reward of generation per state", generation_reward)
            wandbLogger.add_logging_value_single("avg reward of generation per state clipped",
                                                     generation_reward_clipped)
            wandbLogger.add_logging_value_single("avg positive reward of generation per state",
                                                     generation_positive_reward)
            wandbLogger.add_logging_value_single("avg negative reward of generation per state",
                                                     generation_negative_reward)

            for loss_key in avg_losses.keys():
                avg_loss = sum(avg_losses[loss_key]) / len(avg_losses[loss_key])
                wandbLogger.add_logging_value_single("avg " + str(loss_key) + " of generation", avg_loss)

            if pod_counter > 0:
                wandb.log(wandbLogger.get_logging_values())

        # combined (Dirt + Wood, but without zero-shot)
        pod_counter = 0
        generation_reward = 0
        generation_reward_clipped = 0
        generation_positive_reward = 0
        generation_negative_reward = 0
        avg_losses = {}
        wandbLogger = WandbLogger("combined")
        for mission_key in self.rewards_of_generation_per_state.keys():
            if mission_key == "zero-shot":
                continue

            for destruction_factor_key in self.rewards_of_generation_per_state[mission_key].keys():
                for loss_key in self.losses_of_generation[mission_key][destruction_factor_key].keys():
                    avg_loss = 0
                    for loss in self.losses_of_generation[mission_key][destruction_factor_key][loss_key]:
                        avg_loss += loss
                    avg_loss = avg_loss / len(self.losses_of_generation[mission_key][destruction_factor_key][loss_key])
                    if not loss_key in avg_losses.keys():
                        avg_losses[loss_key] = [avg_loss]
                    else:
                        avg_losses[loss_key].append(avg_loss)

                pod_counter += 1
                generation_reward += sum(
                    self.rewards_of_generation_per_state[mission_key][destruction_factor_key]) / len(
                    self.rewards_of_generation_per_state[mission_key][destruction_factor_key])
                generation_reward_clipped += sum(
                    self.rewards_of_generation_per_state_clipped[mission_key][destruction_factor_key]) / len(
                    self.rewards_of_generation_per_state_clipped[mission_key][destruction_factor_key])
                generation_positive_reward += sum(
                    self.positive_rewards_of_generation_per_state[mission_key][destruction_factor_key]) / len(
                    self.positive_rewards_of_generation_per_state[mission_key][destruction_factor_key])
                generation_negative_reward += sum(
                    self.negative_rewards_of_generation_per_state[mission_key][destruction_factor_key]) / len(
                    self.negative_rewards_of_generation_per_state[mission_key][destruction_factor_key])

        for loss_key in avg_losses:
            avg_loss = sum(avg_losses[loss_key]) / len(avg_losses[loss_key])
            wandbLogger.add_logging_value_single("avg " + str(loss_key) + " of generation", avg_loss)

        if pod_counter <= 0:
            print("Missing values for logging, somehow an entry exists for the mission but no values were logged")
        else:
            generation_reward/= pod_counter
            generation_reward_clipped/= pod_counter
            generation_positive_reward/= pod_counter
            generation_negative_reward/= pod_counter

        wandbLogger.add_logging_value_single("avg reward of generation per state", generation_reward)
        wandbLogger.add_logging_value_single("avg reward of generation per state clipped",
                                             generation_reward_clipped)
        wandbLogger.add_logging_value_single("avg positive reward of generation per state",
                                             generation_positive_reward)
        wandbLogger.add_logging_value_single("avg negative reward of generation per state",
                                             generation_negative_reward)

        if pod_counter > 0:
            wandb.log(wandbLogger.get_logging_values())

        if generation_reward > self.best_avg_reward_generation:
            self.best_avg_reward_generation = generation_reward
            self.best_model_path = str(self.model_path) + "/best/" + self.model_name + ".pt"
            model_dict = {"model_state_dict": self.model.state_dict(),
                          "model_optimizer_dict": self.optimizer.state_dict(),
                          "model_specific_parameter": self.model.get_relevant_parameters()}

            torch.save(model_dict, self.best_model_path)
            print("saved new best model with a avg reward of generation of:", str(generation_reward),
                  "at:" + str(self.best_model_path))

        self.reset_values_of_generation()

        if self.pods_finished >= self.max_pods:
            print("Training finished")
            print("Uploading base model to wandb")
            wandb.save(self.base_model_path)
            print("Uploading best model to wandb")
            wandb.save(self.best_model_path)
            print("Uploading newest model to wandb")
            wandb.save(self.current_model_path)
            wandb.finish()
            os._exit(0)
        else:
            self.create_pods()

    def shutdown_pod(self, pod_name, repeat=False, successful=True):
        # only valid pods should reach this (only connected in the kubernet) but to make sure the api was not called accidentally:
        if not pod_name in self.started_pod_names and not repeat:
            return

        if not repeat:
            self.started_pod_names.remove(pod_name)
        time.sleep(1) # allowing pod internal shutdown

        # Delete pod
        try:
            api_instance = client.CoreV1Api(self.k8s_client)
            api_instance.delete_namespaced_pod(pod_name, namespace="mineflayer-stack")
        except Exception as e:
            print(e)
            time.sleep(5)
            self.shutdown_pod(pod_name, repeat=True, successful=successful)
            return

        self.pods_running -= 1
        if successful:
            self.pods_finished += 1

        # new generation
        if self.pods_running <= 0:
            self.finish_generation()

    def update_model_aggregate_method(self, gradients_to_use, method_type="mean"):
        gradient_means = []
        gradient_layer_count = len(gradients_to_use[0])
        total_gradient_count = len(gradients_to_use)

        # We take gradients by layer and take the sum of all the runs with the layer
        for i in range(gradient_layer_count):
            gradient_sum = None
            for gradient_step in gradients_to_use:
                if gradient_sum is None:
                    gradient_sum = gradient_step[i]
                elif gradient_step[i] is not None:
                    gradient_sum = gradient_sum + gradient_step[i]
            if method_type=="mean" and gradient_sum is not None:
                gradient_means.append(gradient_sum/total_gradient_count)  # normalizing by the batches we used
            else:
                gradient_means.append(gradient_sum)

        self.update_model_single_step(gradient_means)


    def update_model_single_step(self, gradient_step):
        self.optimizer.zero_grad() # should not be needed since we load the gradient from what was send
        for parameter, gradient in zip(self.model.parameters(), gradient_step):
            if gradient is None:
                parameter._grad = None
            else:
                parameter._grad = gradient.to(self.device)
        gradient_size = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm, norm_type=2)
        wandb.log({"gradient size": gradient_size})
        self.optimizer.step()
        self.model.step()
        #self.scheduler.step()
        relevant_parameters = self.model.get_relevant_parameters()
        if len(relevant_parameters.keys()) > 0:
            wandb.log(relevant_parameters)
        wandb.log({"learning rate": self.optimizer.param_groups[0]["lr"]})


    def update_model_step_method(self, gradients_to_use):
        for gradient_step in gradients_to_use:
            self.update_model_single_step(gradient_step)

    def update_model(self):
        self.updating_model = True
        while self.model_lock:
            time.sleep(0.01)


        time_start = time.time_ns()
        if self.use_buffer:
            self.use_buffer = False
            used_buffer = True
            gradients_to_use = self.gradients_to_update_buffer

        else:
            self.use_buffer = True
            used_buffer = False
            gradients_to_use = self.gradients_to_update

        gradient_count = len(gradients_to_use)
        if gradient_count <= 0:
            print("somehow an empty list was sent to update gradients")
            if len(self.gradients_to_update) >= self.gradients_needed_for_step or \
                    len(self.gradients_to_update_buffer) >= self.gradients_needed_for_step:
                self.update_model()
            else:
                self.updating_model = False
            return

        if self.model_update_method == "step":
            self.update_model_step_method(gradients_to_use)
        elif self.model_update_method == "mean":
            self.update_model_aggregate_method(gradients_to_use, method_type="mean")
        else:
            self.update_model_aggregate_method(gradients_to_use, method_type="sum")

        self.model_step += 1
        self.recently_updated = True
        #self.current_model_path = self.save_model()


        time_end = time.time_ns()
        time_needed = (time_end - time_start) / 1000 / 1000
        #print("time needed to update gradients of the model:", str(time_needed)+ "ms with", str(len(gradients_to_use)) , "gradients")
        wandb.log({"time needed to update model":time_needed,
                   "gradient count": len(gradients_to_use)})

        if used_buffer:
            self.gradients_to_update_buffer = []
        else:
            self.gradients_to_update = []

        if len(self.gradients_to_update) >= self.gradients_needed_for_step or \
                len(self.gradients_to_update_buffer) >= self.gradients_needed_for_step:
            self.update_model()
        else:
            self.updating_model = False




    # In the case some pods are unable to stop themself (~ 5% of all pods through unable to start server, disconnections from server, tcp connection fails between environment and mineflayer
    def emergency_pod_deletion(self, generation_start):
        time.sleep(self.run_config["batch_size"] * self.run_config["timestep_delay"] * self.run_config["max_episode_count"] + 1800) # 30 Minutes should cover even the worst startup and shutdown times
        if generation_start != self.model_generation:
            print("every pod ended successfully")
            return

        if len(self.started_pod_names) == 0:
            print("Pods did not fail but next generation was not started")
            print("Trying to finish current generation (should start next generation as well)")
            self.finish_generation()

        else:
            print("some pods got stuck and will be terminated forcefully")
            for pod_name in copy(self.started_pod_names):
                print("the following pod got stuck:", str(pod_name))
                self.failed_pods_in_generation += 1
                self.shutdown_pod(pod_name, repeat=False, successful=False)
            if generation_start == self.model_generation and self.pods_running > 0:
                print("somehow miscounted pods running, starting new Generation anyways")
                self.finish_generation()

    
    def create_mission_type_pods(self, deployment_yaml, mission_type, count=5):
        if mission_type == "zero-shot":
            destruction_factor = 2.0 # collecting a block becomes -1
        else:
            destruction_factor = 0.0 # only a reward for collecting a block

        for i in range(count):
            pod_name = self.run_name + str(mission_type) + "mc" + str(self.pods_started)
            deployment_yaml["metadata"]["name"] = pod_name

            # [server, environment, nodejs]
            deployment_yaml["spec"]["containers"][1]["env"] = [
                {"value": pod_name,
                 "name": "pod_name"
                 },

                {
                    "value": str(self.current_base_port),
                    "name": "trainer_tcp_port"
                },
                {
                    "value": self.trainer_ip_addr,
                    "name": "trainer_ip_addr"
                },
                {
                    "value": str(int(self.seeds_to_use[self.pods_started])),
                    "name": "seed"
                },
                {"value": str(destruction_factor),
                 "name": "destruction_factor"
                 },
                {"value":str(mission_type),
                 "name":"mission_type"}

            ]
            destruction_factor += 0.1

            # Change the seed of the yaml for more consistency when comparing runs
            # usefull seeds were pre-computed (forest biomes)
            # first casting seed to int to make sure that it is an Integer instead of a float
            deployment_yaml["spec"]["containers"][0]["env"][0]["value"] = str(int(self.seeds_to_use[self.pods_started]))
            self.current_base_port += 2  # 2 tcp connections per pod
            try:
                utils.create_from_yaml(self.k8s_client, yaml_objects=[
                    deployment_yaml], namespace="mineflayer-stack")
            except Exception as e:
                print(e)
                time.sleep(5)
                continue

            self.pods_running += 1
            self.started_pod_names.append(pod_name)
            self.pods_started += 1
    
    def create_pods(self):
        self.model_generation += 1
        self.failed_received_msg_this_generation = 0
        with open("mineflayer-stack.yaml") as f:
            deployment_yaml = yaml.safe_load(f)

        self.create_mission_type_pods(deployment_yaml, "wood", 5)
        self.create_mission_type_pods(deployment_yaml, "dirt", 5)
        self.create_mission_type_pods(deployment_yaml, "zero-shot", 1)

        self.current_base_port = self.base_port
        self.emergency_pod_deletion_thread = threading.Thread(target=self.emergency_pod_deletion, args=[self.model_generation])
        self.emergency_pod_deletion_thread.start()

    def save_model(self):
        start_time = time.time_ns()
        path = str(self.model_path) + "/" + self.model_name + "_step_" + str(self.model_step) + ".pt"
        # in case no model step happened since the last save
        if len(self.last_saved_models) > 0:
            if self.last_saved_models[-1] == path:
                print("No new model step, not saving current model")
                return path

        model_dict = {"model_state_dict": self.model.state_dict(),
                      "model_optimizer_dict": self.optimizer.state_dict(),
                      "model_specific_parameter": self.model.get_relevant_parameters()}
        torch.save(model_dict, path)
        print("model saved at:", str(path))
        end_time = time.time_ns()
        time_needed = (end_time - start_time) / 1000 / 1000
        if self.debug_mode:
            wandb.log({"time needed to save model step": time_needed})

        self.last_saved_models.append(path)
        return path

    def listen_tcp_for_gradients(self, target_host, target_port):
        socket_instance = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_instance.bind((target_host, target_port))
        socket_instance.listen()

        while True:
            try:
                conn, addr = socket_instance.accept()
                while True:
                    my_message = socket_functions.recv_msg(conn)
                    if my_message is None:
                        conn.shutdown(socket.SHUT_RDWR)
                        conn.close()
                        break

                    self.process_gradient_package(my_message)
                    #new_thread = threading.Thread(target=self.process_gradient_package, args=[my_message])
                    #new_thread.start()


            except Exception as e:
                print(e)




if __name__ == "__main__":
    trainer = Trainer()
    app.include_router(trainer.router)
    trainer.train()



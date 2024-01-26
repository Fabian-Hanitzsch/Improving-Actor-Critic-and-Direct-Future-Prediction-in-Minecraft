import math

import torch
import torch.nn as nn
import numpy as np
from utils.StateStorage import StateStorage
from utils.model_functions import get_possible_actions, action_dict_to_key
import torch.nn.functional as F
import minecraft_data
import json

def get_mc_data_items_by_name(version):
    mc_data = minecraft_data(version)
    return mc_data.items_name

class Base(nn.Module):
    def __init__(self, config):
        super(Base, self).__init__()
        self.mission_path = config["dirt_mission_path"]
        self.image_width = config["image_width"]
        self.image_height = config["image_height"]
        self.image_channels = config["image_channels"]
        self.sequence_length = config["sequence_length"]
        self.gamma = config["gamma"]
        self.reward_multiplicator = config["reward_multiplicator"]
        self.state_buffer_count = 0

        self.clipped_reward = True

        self.minecraft_data_version = config["minecraft_data_version"]

        self.mc_data_items_by_name = get_mc_data_items_by_name(self.minecraft_data_version)
        self.train_mode = True

        with open(self.mission_path + "/items-by-name.json") as f:
            self.items_by_name = json.load(f)

        self.items_by_id = {}
        for key in self.items_by_name:
            self.items_by_id[self.items_by_name[key]] = key

        with open(self.mission_path + "/equipable-items-by-name.json") as f:
            self.equipable_items_by_name = json.load(f)
        self.equipable_items_by_id = {}
        for key in self.equipable_items_by_name:
            self.equipable_items_by_id[self.equipable_items_by_name[key]] = key

        with open(self.mission_path + "/recipes-by-name.json") as f:
            self.recipes_by_name = json.load(f)

        self.recipes_by_id = {}
        for key in self.recipes_by_name:
            self.recipes_by_id[self.recipes_by_name[key]] = key

        with open(self.mission_path + "/goal.json") as f:
            self.goal_items = json.load(f)

        # For this work: 0 hands + inventory
        self.inventory_types = 1
        self.inventory_slots = 36

        # For this work: 0 hand (+2 per hand), 3 measurements
        self.extra_information = 3

        # 6 binary, 2 camera, 0 crafting/equipping
        self.action_encoded_count = 6 + 2 + 0


        self.item_types_count = len(self.items_by_name)
        self.goal_size = len(self.goal_items) * 2 + 1

        if "from_checkpoint" in config:
            self.load_model(config["from_checkpoint"])
        else:
            self._create_model()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")

    def load_model(self, path):
        pass

    def step(self):
        return

    def set_train_mode(self, train_mode):
        self.train_mode = train_mode

    def _create_model(self):
        image_width = self.image_width
        image_height = self.image_height

        self.vision_1 = nn.Conv2d(self.image_channels, 32, (8, 8), stride=(4, 4), padding=(3,3), padding_mode='zeros')
        image_width = math.floor(image_width / 4)
        image_height = math.floor(image_height / 4)

        self.vision_2 = nn.Conv2d(32, 64, (4, 4), stride=(2, 2), padding=(1,1), padding_mode='zeros')
        image_width = math.floor(image_width/ 2)
        image_height = math.floor(image_height/ 2)

        self.vision_3 = nn.Conv2d(64, 64, (3, 3), padding=(1,1), padding_mode="zeros")

        linear_dimension = image_width * image_height * 64
        self.linear_vision_1 = nn.Linear(linear_dimension, 256)
        #self.linear_vision_2 = nn.Linear(256 * self.sequence_length, 256)


        game_state_dimension = self.item_types_count * self.inventory_types + self.extra_information + self.action_encoded_count
        self.game_state_linear_1 = nn.Linear(game_state_dimension, 128)
        self.goal_linear_1 = nn.Linear(self.goal_size, 128)

        self.combined_features_input_length = 256 + 128 + 128
        self.combined_features_length = 256

        self.combined_features_linear_1 = nn.Linear(self.combined_features_input_length, self.combined_features_length)

    def get_relevant_parameters(self):
        return {"clipped_reward": self.clipped_reward}

    def update_relevant_parameters(self, new_dict):
        self.clipped_reward = new_dict["clipped_reward"]
        return

    def _replace_item_index(self, item_string):
        item_string.replace("#", "")
        for i in range(10):
            item_string.replace(str(i), "")
        return item_string

    # inventory[key]["amount"]
    # armor_dict[key]["amount"]


    def _inventory_or_armor_to_tensor(self, dictionary, armor=False, normalize=True):
        count_list = [0] * self.item_types_count
        if armor:
            max_slots = 1
        else:
            max_slots = self.inventory_slots


        for key in dictionary.keys():
            # remove indices for items (e.g. iron_axe#0) since I do not know how to build the system otherwise
            if armor:
                item_name = self._replace_item_index(dictionary[key]["type"])
            else:
                item_name = self._replace_item_index(key)

            item_amount = dictionary[key]["amount"]
            # we use the max amount of items to normalize the result
            # The none slot becomes the "other" slot
            # We use a constant for "other" to keep it reversable (e.g. prediction model -> inventory representation)
            if item_name not in self.items_by_name:
                # self.mc_data_items_by_name[item_name]["stackSize"]
                if normalize: item_amount /= 64 #* max_slots
                count_list[0] += item_amount
            elif item_name == "air":
                if normalize: item_amount /= max_slots
                count_list[self.items_by_name[item_name]] = item_amount
            else:
                if normalize: item_amount /= self.mc_data_items_by_name[item_name]["stackSize"] #* max_slots
                count_list[self.items_by_name[item_name]] = item_amount

        return torch.tensor(count_list, device=self.device).float()


    def _hand_to_tensor(self, hand_dict):
        hand_list = [0] * self.item_types_count
        item_name = self._replace_item_index(hand_dict["type"])

        # We reduce the input size by using a full stack to get to the range 0-1
        if item_name not in self.items_by_name:
            hand_list[0] += hand_dict["amount"] / self.mc_data_items_by_name[item_name]["stackSize"] # The none slot becomes the "other" slot
        elif item_name == "air":
            hand_list[self.items_by_name[item_name]] = 1
        else:
            hand_list[self.items_by_name[item_name]] = hand_dict["amount"] / self.mc_data_items_by_name[item_name]["stackSize"]

        hand_list.append(hand_dict["damage"])
        hand_list.append(hand_dict["max_damage"])
        return torch.tensor(hand_list, device=self.device).float()


    def _get_game_state_tensor(self, game_state):

        inventory_tensor = self._inventory_or_armor_to_tensor(game_state["inventory"], armor=False)
        action_tensor = self._action_dict_to_tensor(game_state["current_action"])

        #armor_tensor = self._inventory_or_armor_to_tensor(game_state["armor"], armor=True)
        #window_tensor = self._inventory_or_armor_to_tensor(game_state["window"], armor=False)
        #main_hand_tensor = self._hand_to_tensor(game_state["main_hand"])
        #second_hand_tensor = self._hand_to_tensor(game_state["second_hand"])

        values_tensor = torch.tensor(
            [game_state["health"] / 20,
             #game_state["experience"], game_state["level"], game_state["hunger"] / 20,
             game_state["mining_progress"], game_state["is_in_water"]], device=self.device).float()

        game_state_tensor = torch.cat(
            [inventory_tensor, action_tensor,
             #armor_tensor, window_tensor, second_hand_tensor, main_hand_tensor,
             values_tensor]).float()

        return game_state_tensor

    def _add_one_hot_encode_item_action(self, action_list, action_name):
        for i in range(self.item_types_count):
            item_name = self.items_by_id[i]
            if action_name == item_name:
                action_list.append(1)
            else:
                action_list.append(0)

        return action_list

    def _action_dict_to_tensor(self, action_dict):
        action_list = []
        action_list.append(int(action_dict["forward"]))
        action_list.append(int(action_dict["back"]))
        action_list.append(int(action_dict["left"]))
        action_list.append(int(action_dict["right"]))
        #action_list.append(int(action_dict["sprint"]))
        #action_list.append(int(action_dict["sneak"]))
        #action_list.append(int(action_dict["use"]))
        action_list.append(int(action_dict["attack"]))
        action_list.append(int(action_dict["jump"]))

        # For this work not relevant, so they were removed
        #action_list = self._add_one_hot_encode_item_action(action_list, action_dict["insert"])
        #action_list = self._add_one_hot_encode_item_action(action_list, action_dict["take"])
        #action_list = self._add_one_hot_encode_item_action(action_list, action_dict["equip"])

        for i in range(len(self.recipes_by_name)):
            item_name = self.recipes_by_id[i]
            if action_dict["craft"] == item_name:
                action_list.append(1)
            else:
                action_list.append(0)

        action_list.append(action_dict["camera"][0] / 180.0)
        action_list.append(action_dict["camera"][1] / 180.0)

        return torch.tensor(action_list, device=self.device).float()


    def _get_action_tensor(self, action_sequence):
        action_tensors = []
        for action in action_sequence:
            action_tensors.append(self._action_dict_to_tensor(action))
        action_tensors = torch.stack(action_tensors, dim=0)
        return action_tensors

    def _partial_forward(self, states):
        goals = []
        game_states = []
        observations = []
        #action_sequence = []

        for state in states:
            observation = self._v_wrap(state["image"])
            # reducing range of images from 0 to 255 -> 0 to 1
            observation = observation / 255
            # pytorch expects [channel, height, width, count], we got [height, width, channel, count]
            observation = observation.transpose(0, 2)  # [channel, width, height, count]
            observation = observation.transpose(1, 2)  # [channel, height, width, count]
            observations.append(observation)

            goal = state["goal"]
            goal_tensor = torch.tensor(goal, device=self.device).float()
            goals.append(goal_tensor)


            game_state = state["game_state"]
            game_state_tensor = self._get_game_state_tensor(game_state)
            game_states.append(game_state_tensor)

        game_states = torch.stack(game_states, dim=0)
        observations = torch.stack(observations, dim=0)
        goals = torch.stack(goals, dim=0)

        vision = self.vision_1(observations)
        vision = nn.functional.relu(vision)
        vision = self.vision_2(vision)
        vision = nn.functional.relu(vision)
        vision = self.vision_3(vision)
        vision = nn.functional.relu(vision)

        # The Sequence is on the last layer so we keep that layer intact and swap it with the features to
        # get it into the second position and the features into the last
        #vision = torch.flatten(vision, start_dim=1, end_dim=-2).transpose(1,2)
        #vision = self.linear_vision_1(vision)
        vision = torch.flatten(vision, start_dim=1, end_dim=-1)
        vision = self.linear_vision_1(vision)
        vision = nn.functional.relu(vision)

        game_state = self.game_state_linear_1(game_states)
        game_state = nn.functional.relu(game_state)

        goal = self.goal_linear_1(goals)
        goal = nn.functional.relu(goal)


        # 512 + 128 + 128

        combined_features_tensor = torch.cat([vision, game_state, goal], dim=-1)

        combined_features = self.combined_features_linear_1(combined_features_tensor)
        combined_features = nn.functional.relu(combined_features)


        return combined_features

    def _set_init(self, layers, std=0.01):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=std)
            nn.init.normal_(layer.bias, mean=0., std=std)

    def _v_wrap(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)


    def _get_value_of_state(self, state):
        raise NotImplementedError

    def _discounted_reward(self, states, rewards, last_state, finished=False):
        if finished:
            value_next_state = 0
            #value_next_state = self._get_value_of_state([last_state])
        else:
            value_next_state = self._get_value_of_state([last_state])

        buffer_v_target = []
        #action_length = last_state["game_state"]["action_length"]
        #k = len(states) - 1

        for r in rewards[::-1]:  # reverse rewards
            #value_next_state = (self.gamma ** (action_length - 1)) * (r + self.gamma * value_next_state)
            value_next_state = r + self.gamma * value_next_state
            #action_length = states[k]["game_state"]["action_length"]
            #k-= 1
            buffer_v_target.append(value_next_state)
        buffer_v_target.reverse()
        return buffer_v_target


    def calculate_gradients(self, state_storage:StateStorage):
        raise NotImplementedError


if __name__ == "__main__":
    with open('../Configs/model-parameters.json') as f:
        model_parameter_config = json.load(f)
    testBase = Base(model_parameter_config)
    test = sum([param.nelement() for param in testBase.parameters()])
    print(test)


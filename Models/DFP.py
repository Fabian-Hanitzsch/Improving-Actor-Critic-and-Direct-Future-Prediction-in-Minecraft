from Models.Base import Base
import torch.nn.functional as F
import torch.nn as nn
import torch
import json
from copy import deepcopy
import numpy as np
import random
from utils.StateStorage import StateStorage
from utils.model_functions import action_dict_to_key, get_possible_actions, inventory_change_as_goal_representation
import math

import time

class DFP(Base):

    def __init__(self, config):
        super(DFP, self).__init__(config)
        if "model_specific_parameters" in config:
            self._define_model_specific_values(config["model_specific_parameters"])
        else:
            self.timesteps = [1,2,4,8,16,32]
            self.timesteps_importance = [0,0,0,0.5,0.5,0.5]
            self.start_epsilon = 1.0
            self.min_epsilon = 0.1
            self.exploration_batches = 500

        # self.epsilon_multiplier = (self.min_epsilon / self.start_epsilon) ** (1.0/self.exploration_batches)
        self.epsilon_change = (self.start_epsilon - self.min_epsilon) / self.exploration_batches
        self.possible_actions_id, self.possible_actions_name = get_possible_actions(self.recipes_by_name, self.items_by_name)

        self.epsilon = self.start_epsilon

        #self.timesteps_importance = self.timesteps_importance/np.sum(self.timesteps_importance) # normalizing

        self.expectation_stream_1 = nn.Linear(self.combined_features_length, 128)
        self.expectation_stream_2 = nn.Linear(128, self.goal_size * len(self.timesteps))

        self.action_streams_1 = nn.ModuleList()
        self.action_streams_2 = nn.ModuleList()
        for i in range(len(self.possible_actions_name)):
            self.action_streams_1.append(nn.Linear(self.combined_features_length, 128))
            self.action_streams_2.append(nn.Linear(128, self.goal_size * len(self.timesteps)))


    def _define_model_specific_values(self, model_specific):
        self.timesteps = np.array(model_specific["timesteps"])
        self.state_buffer_count = self.timesteps[-1] + 1

        self.timesteps_importance = np.array(model_specific["timesteps_importance"])
        self.start_epsilon = model_specific["start_epsilon"]
        self.min_epsilon = model_specific["min_epsilon"]

        self.exploration_batches = model_specific["exploration_batches"]


    def step(self):
        super(DFP, self).step()
        #self.epsilon *= self.epsilon_multiplier
        self.epsilon -= self.epsilon_change
        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

    def forward(self, state_and_pov):
        combined_featues = super(DFP, self)._partial_forward(state_and_pov)

        expectation_stream = self.expectation_stream_1(combined_featues)
        expectation_stream = nn.functional.relu(expectation_stream)
        expectation_stream = self.expectation_stream_2(expectation_stream)


        actions_list = []
        for action_stream_layer_1, action_stream_layer_2 in zip(self.action_streams_1, self.action_streams_2):
            action_stream = action_stream_layer_1(combined_featues)
            action_stream = nn.functional.relu(action_stream)
            action_stream = action_stream_layer_2(action_stream)
            actions_list.append(action_stream)

        action_stream_tensor = torch.stack(actions_list, dim=1)
        # Normalizing the action stream so that every measurement is predicted as 0 on average.
        # I am not sure if the repeat is necessary or if pytorch would handle the subtraction correctly internally.
        action_stream_tensor = action_stream_tensor - action_stream_tensor.mean(dim=1, keepdim=True).repeat(1, action_stream_tensor.size(1), 1)

        # duplicating the expectation stream to have the same dimensions as the action stream
        # For this we first create a view which has an extra dimension along the action dimension of the action stream
        expectation_stream_duplicated = expectation_stream[:, None, :].repeat((1,action_stream_tensor.size(1), 1))

        # prediction is the action stream added with the expectation stream
        predictions_tensor = action_stream_tensor + expectation_stream_duplicated

        return predictions_tensor

    def choose_action(self, x):
        time_start = time.time_ns()
        if np.random.rand() <= self.epsilon and self.train_mode:
            action_id = random.randint(0, len(self.possible_actions_id) - 1)

        else:
            predictions_tensor = self.forward([x])
            goal = x["goal"] # get goal vector
            inference_goal = []
            for importance in self.timesteps_importance:
                inference_goal.extend(goal * importance)

            inference_goal = torch.tensor(inference_goal, device=self.device)

            multiplied_tensor = torch.mul(predictions_tensor, inference_goal)
            sum_tensor = torch.sum(multiplied_tensor, dim=2) # along the measurements
            action_id = torch.argmax(sum_tensor, dim=1).item()

        action = self.possible_actions_id[str(action_id)]

        time_end = time.time_ns()
        time_needed_milliseconds = (time_end - time_start) / 1000 / 1000

        return action, time_needed_milliseconds


    def get_relevant_parameters(self):
        base_relevant_parameters = super(DFP, self).get_relevant_parameters()
        base_relevant_parameters["epsilon"] = self.epsilon
        return base_relevant_parameters

    def update_relevant_parameters(self, new_dict):
        super(DFP, self).update_relevant_parameters(new_dict)
        self.epsilon = new_dict["epsilon"]


    def get_item_change(self, state_storage, start_index, end_index):
        if not self.clipped_reward:
            current_state = state_storage.get_game_state(start_index)
            end_state = state_storage.get_game_state(end_index)
            _, unclipped_change = inventory_change_as_goal_representation(state_storage.goal_items, current_state, end_state)
            return unclipped_change
        else:
            total_clipped_change = [0] * len(state_storage.goal_items) * 2 + [0]
            for i in range(end_index - start_index):
                current_state = state_storage.get_game_state(start_index + i)
                end_state = state_storage.get_game_state(start_index + i + 1)
                clipped_change, _ = inventory_change_as_goal_representation(state_storage.goal_items, current_state, end_state)
                total_clipped_change += clipped_change
            return np.array(total_clipped_change)

    def calculate_gradients(self, state_storage: StateStorage, update_type="normal"):

        max_index = len(state_storage.states) - (self.timesteps[-1] + 1)
        if max_index <= 0:
            print("not enough states for loss calculation")
            return None
        self.zero_grad(set_to_none=True)
        f_action_target = np.zeros((max_index, (self.goal_size * len(self.timesteps))))
        state_input = []
        action = []
        goals = []
        clipped_rewards = []
        un_clipped_rewards = []


        for index in range(max_index):
            future_measurements = []
            for j in self.timesteps:
                future_measurements.append(torch.tensor(self.get_item_change(state_storage, index, index + j)))
            future_measurements = torch.cat(future_measurements, dim=0).cpu()

            f_action_target[index, :] = np.array(future_measurements)

            state = state_storage.states[index]
            action_dict = state_storage.actions[index]
            state_input.append(state)
            action.append(int(self.possible_actions_name[action_dict_to_key(action_dict)]))
            goals.append(state["goal"])
            clipped_rewards.append(state_storage.clipped_rewards[index])
            un_clipped_rewards.append(state_storage.un_clipped_rewards[index])

        predictions_tensor = self.forward(state_input)

        f_target = torch.clone(predictions_tensor)

        f_action_target = torch.tensor(f_action_target, device=self.device)
        action_advantages = []

        for i in range(max_index):
            f_target[i][action[i]] = f_action_target[i]
            inference_goal = []
            for importance in self.timesteps_importance:
                inference_goal.extend(goals[i] * importance)

            inference_goal = torch.tensor(inference_goal, device=self.device)
            multiplied_tensor = torch.mul(f_target[i], inference_goal)

            avg_action_value = multiplied_tensor.sum(dim=1).mean()
            chosen_action_value = multiplied_tensor[action[i]].sum()

            action_advantages.append((chosen_action_value - avg_action_value).item())

        # The target should not influence the loss directly
        f_target = f_target.detach()

        td = f_target - predictions_tensor
        loss = td.pow(2)

        # The sum over the predictions, actions are summed together because only one is of relevance and the mean of the batch as a whole
        loss = loss.sum(dim=-1).sum(dim=-1).mean(dim=-1)

        action_advantage = np.mean(action_advantages)
        #print("loss:", str(loss))

        loss.backward()
        losses = {"total_loss": loss,
                  "action_advantage": action_advantage,
                  "un_clipped_rewards": np.mean(un_clipped_rewards),
                  "clipped_rewards": np.mean(clipped_rewards)}

        if self.clipped_reward:
            rewards = clipped_rewards
        else:
            rewards = un_clipped_rewards
        losses["rewards"] = np.mean(rewards)



        losses["clipped_max_rewards"] = np.max(clipped_rewards)
        losses["un_clipped_max_rewards"] = np.max(un_clipped_rewards)
        losses["clipped_min_rewards"] = np.min(clipped_rewards)
        losses["un_clipped_min_rewards"] = np.min(un_clipped_rewards)


        return losses

if __name__ == "__main__":
    with open('../Configs/model-parameters.json') as f:
        model_parameter_config = json.load(f)
    DFP_1 = DFP(model_parameter_config)
    DFP_2 = DFP(model_parameter_config)
    test = sum([param.nelement() for param in DFP_1.parameters()])
    print(test)
    start_time = time.time_ns()

    for param, new_param in zip(DFP_1.parameters(), DFP_2.parameters()):
        param.data = new_param.to("cpu")

    DFP_1.update_relevant_parameters(DFP_2.get_relevant_parameters())
    end_time = time.time_ns()
    time_needed = (end_time - start_time) / 1000 / 1000
    print("time needed:", str(time_needed) + "ms")



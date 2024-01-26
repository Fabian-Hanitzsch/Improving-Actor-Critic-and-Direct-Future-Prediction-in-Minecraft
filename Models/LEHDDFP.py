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

class LEHDDFP(Base):

    def __init__(self, config):
        super(LEHDDFP, self).__init__(config)
        if "model_specific_parameters" in config:
            self._define_model_specific_values(config["model_specific_parameters"])
        else:
            self.start_epsilon = 1.0
            self.min_epsilon = 0.1
            self.exploration_batches = 500
            self.limited_event_horizon = 32

        #self.epsilon_multiplier = (self.min_epsilon / self.start_epsilon) ** (1.0/self.exploration_batches)
        self.epsilon_change = (self.start_epsilon - self.min_epsilon) /self.exploration_batches
        self.possible_actions_id, self.possible_actions_name = get_possible_actions(self.recipes_by_name, self.equipable_items_by_name)
        self.epsilon = self.start_epsilon



        self.expectation_stream_1 = nn.Linear(self.combined_features_length, 128)
        self.expectation_stream_2 = nn.Linear(128, self.goal_size)

        self.action_streams_1 = nn.ModuleList()
        self.action_streams_2 = nn.ModuleList()
        for i in range(len(self.possible_actions_name)):
            self.action_streams_1.append(nn.Linear(self.combined_features_length, 128))
            self.action_streams_2.append(nn.Linear(128, self.goal_size))


    def _define_model_specific_values(self, model_specific):
        self.start_epsilon = model_specific["start_epsilon"]
        self.min_epsilon = model_specific["min_epsilon"]
        self.exploration_batches = model_specific["exploration_batches"]
        self.limited_event_horizon = model_specific["timesteps"][-1]


    def step(self):
        super(LEHDDFP, self).step()
        #self.epsilon *= self.epsilon_multiplier
        self.epsilon -= self.epsilon_change
        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

    def forward(self, state_and_pov):
        combined_featues = super(LEHDDFP, self)._partial_forward(state_and_pov)

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

    def choose_best_action(self, predictions_tensor, goal):
        goal = torch.tensor(goal, device=self.device)[None, :]

        multiplied_tensor = torch.mul(predictions_tensor, goal)
        sum_tensor = torch.sum(multiplied_tensor, dim=2)  # along the measurements
        action_id = torch.argmax(sum_tensor, dim=1).item()
        return action_id

    def choose_action(self, x):
        time_start = time.time_ns()
        if np.random.rand() <= self.epsilon and self.train_mode:
            action_id = random.randint(0, len(self.possible_actions_id) - 1)

        else:
            predictions_tensor = self.forward([x])
            goal = x["goal"] # get goal vector
            action_id = self.choose_best_action(predictions_tensor, goal)

        action = self.possible_actions_id[str(action_id)]

        time_end = time.time_ns()
        time_needed_milliseconds = (time_end - time_start) / 1000 / 1000

        return action, time_needed_milliseconds


    def get_relevant_parameters(self):
        base_relevant_parameters = super(LEHDDFP, self).get_relevant_parameters()
        base_relevant_parameters["epsilon"] = self.epsilon
        base_relevant_parameters["clipped_reward"] = self.clipped_reward
        return base_relevant_parameters

    def update_relevant_parameters(self, new_dict):
        super(LEHDDFP, self).update_relevant_parameters(new_dict)
        self.epsilon = new_dict["epsilon"]
        self.clipped_reward = new_dict["clipped_reward"]


    def _discounted_measurements(self, states, goal_items):
        max_index = len(states) - (self.limited_event_horizon + 1)
        states_values = []

        for state_index in range(max_index):
            discounted_measurement = [0.] * self.goal_size
            for step_index in range(self.limited_event_horizon):
                clipped_changes, unclipped_changes = inventory_change_as_goal_representation(goal_items, states[state_index + step_index]["game_state"],
                                                                                             states[state_index + step_index + 1]["game_state"])
                if self.clipped_reward:
                    measurement_changes = clipped_changes
                else:
                    measurement_changes = unclipped_changes
                discounted_measurement += measurement_changes * (self.gamma ** step_index)

            states_values.append(discounted_measurement)
        return states_values


    def calculate_gradients(self, state_storage: StateStorage):
        all_states_action_rewards = state_storage.get_all_state_action_reward_tuples()
        action_dicts = all_states_action_rewards["actions"]
        states = all_states_action_rewards["states"]
        last_state = state_storage.get_last_game_state_without_action()
        clipped_rewards = np.array(all_states_action_rewards["clipped_rewards"])
        un_clipped_rewards = np.array(all_states_action_rewards["un_clipped_rewards"])

        self.zero_grad()
        max_index = len(states) - (self.limited_event_horizon + 1)
        if max_index <= 0:
            print("not enough states for loss calculation")
            return None

        f_action_target = np.zeros((max_index, self.goal_size))

        actions = []
        goals = []

        for state, action_dict in zip(states, action_dicts):
            goals.append(state["goal"])
            actions.append(int(self.possible_actions_name[action_dict_to_key(action_dict)]))

        discounted_rewards = self._discounted_measurements(states, state_storage.goal_items)

        for index in range(max_index):
            f_action_target[index, :] = np.array(discounted_rewards[index])

        predictions_tensor = self.forward(states)
        f_target = torch.clone(predictions_tensor)

        f_action_target = torch.tensor(f_action_target, device=self.device)


        for i in range(max_index):
            f_target[i][actions[i]] = f_action_target[i]

        # The target should not influence the loss directly
        f_target = f_target.detach()

        td = f_target - predictions_tensor
        loss = td.pow(2)

        # The sum over the predictions, actions are summed together because only one is of relevance and the mean of the batch as a whole
        loss = loss.sum(dim=-1).sum(dim=-1).mean(dim=-1)

        loss.backward()
        losses = {"total_loss": loss,
                  "clipped_rewards": np.mean(clipped_rewards),
                  "un_clipped_rewards": np.mean(un_clipped_rewards)}

        if self.clipped_reward:
            rewards = clipped_rewards
        else:
            rewards = un_clipped_rewards
        losses["rewards"] = np.mean(rewards)
        losses["clipped_max_rewards"] = np.max(clipped_rewards)
        losses["un_clipped_max_rewards"] = np.max(un_clipped_rewards)
        losses["clipped_min_rewards"] = np.min(clipped_rewards)
        losses["un_clipped_min_rewards"] = np.min(un_clipped_rewards)


        returns = torch.tensor(np.array(discounted_rewards), device=self.device)
        for i in range(returns.size()[1]):
            actual_returns_key = "actual return " + str(i)
            losses[actual_returns_key] = returns[:, i].mean()

            advantage_key = "advantage " + str(i)
            losses[advantage_key] = td[:,:,i].sum(dim=-1).mean(dim=-1)

        for i in range(predictions_tensor.size()[1]):
            for j in range(predictions_tensor.size()[2]):
                action_logging_key = "action " + str(i) +  " " + str(j) + " loggits"
                losses[action_logging_key] = predictions_tensor[:,i,j].mean()

        return losses

if __name__ == "__main__":
    with open('../Configs/model-parameters.json') as f:
        model_parameter_config = json.load(f)
    LEHDDFP_1 = LEHDDFP(model_parameter_config)
    LEHDDFP_2 = LEHDDFP(model_parameter_config)
    test = sum([param.nelement() for param in LEHDDFP_1.parameters()])
    print(test)
    start_time = time.time_ns()

    for param, new_param in zip(LEHDDFP_1.parameters(), LEHDDFP_2.parameters()):
        param.data = new_param.to("cpu")

    LEHDDFP_2.update_relevant_parameters(LEHDDFP_2.get_relevant_parameters())
    end_time = time.time_ns()
    time_needed = (end_time - start_time) / 1000 / 1000
    print("time needed:", str(time_needed) + "ms")



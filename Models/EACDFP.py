from Models.Base import Base
import torch
import torch.nn as nn
import numpy as np
from utils.StateStorage import StateStorage
from utils.model_functions import get_possible_actions, action_dict_to_key, inventory_change_as_goal_representation
import time
import json
import random
import math
import torch.nn.functional as F
from copy import deepcopy

class EACDFP(Base):
    def __init__(self, config):
        super(EACDFP, self).__init__(config)
        self.possible_actions_id, self.possible_actions_name = get_possible_actions(self.recipes_by_name, self.equipable_items_by_name)

        self.action_hidden_1 = nn.Linear(self.combined_features_length, 128)
        self.action_heads = nn.ModuleList()
        for i in range(len(self.possible_actions_id)):
            self.action_heads.append(nn.Linear(128, self.goal_size))

        self.value_hidden_1 = nn.Linear(self.combined_features_length, 128)
        self.value_head = nn.Linear(128, self.goal_size)

        self.categorical_distribution = torch.distributions.Categorical



    def choose_action(self, state):
        time_start = time.time_ns()
        action_logits, _ = self.forward([state])

        goal_vector = deepcopy(state["goal"])
        #probabilities = F.softmax(action_logits, dim=1).data

        probabilities = self.custom_softmax_layer(action_logits, goal_vector)
        m = torch.distributions.Categorical(probabilities)
        action_id = m.sample().item()

        action = self.possible_actions_id[str(action_id)]
        time_end = time.time_ns()
        time_needed_milliseconds = (time_end - time_start) / 1000 / 1000
        if time_needed_milliseconds > 100:
            print("time needed to choose action:", str(time_needed_milliseconds) + "ms")

        return action, time_needed_milliseconds

    def forward(self, states):
        combined_featues = super(EACDFP, self)._partial_forward(states)
        action = self.action_hidden_1(combined_featues)
        action = nn.functional.relu(action)

        combined_action_logits = []
        for action_head in self.action_heads:
            action_logit = action_head(action)
            combined_action_logits.append(action_logit)

        action_logits = torch.stack(combined_action_logits, dim=1)

        state_value = self.value_hidden_1(combined_featues)
        state_value = nn.functional.relu(state_value)
        state_value = self.value_head(state_value)

        return action_logits, state_value

    def custom_softmax_layer(self, input_logits, goal):
        negate_logits = []
        for objective in goal:
            if objective >= 0:
                negate_logits.append(1)
            else:
                negate_logits.append(-1)

        goal_vector = torch.tensor(goal, device=self.device).abs()
        #goal_vector = goal_vector/goal_vector.sum(dim=-1, keepdims=True)
        goal_vector = goal_vector.expand(len(self.possible_actions_id), -1)[None, :]
        negate_logits = torch.tensor(negate_logits, device=self.device).expand(len(self.possible_actions_id), -1)[None, :]
        negate_logits = negate_logits.expand(input_logits.size()[0], -1, -1)
        goal_vector = goal_vector.expand(input_logits.size()[0], -1, -1)
        corrected_logits = input_logits * negate_logits
        calculated_logits = torch.exp(corrected_logits) * goal_vector
        probabilities = calculated_logits.sum(dim=-1)/calculated_logits.sum(dim=-1).sum(dim=-1, keepdims=True)

        return probabilities


    def loss_func(self, states, discounted_measurements, actions, update_type="normal"):
        action_ids = []
        action_repetitions = []
        goal = deepcopy(states[0]["goal"]) # we assume only one goal per mini-episode

        for action in actions:
            action_repetitions.append(1 - int(action["repeated"]))
            action_key = action_dict_to_key(action)
            if action_key not in self.possible_actions_name:
                print("Missing action dict inside possible action ids")
                print(action)
                raise Exception
            action_ids.append(int(self.possible_actions_name[action_key]))

        saved_actions_tensor = torch.tensor(action_ids, device=self.device)
        main_goal_tensor = torch.tensor(goal, device=self.device)

        if update_type == "staggered":
            with torch.no_grad():
                combined_featues = super(EACDFP, self)._partial_forward(states)
            action_logits, state_values = self._get_probabilities(combined_featues)
        else:
            action_logits, state_values = self.forward(states)
        returns = torch.tensor(np.array(discounted_measurements), device=self.device)
        advantage = (returns - state_values)
        adjusted_advantage = advantage * main_goal_tensor

        probabilities = self.custom_softmax_layer(action_logits, goal)
        m = torch.distributions.Categorical(probabilities)
        entropy_loss = m.entropy()
        critic_loss = advantage.pow(2) * main_goal_tensor.abs()
        critic_loss = critic_loss.sum(dim=-1)
        losses = {"critic_loss": critic_loss.mean(),
                  "entropy_loss": entropy_loss.mean(),
                  "entropy_max": entropy_loss.max(),
                  "entropy_min": entropy_loss.min()
                  }



        a_loss = (-m.log_prob(saved_actions_tensor) * adjusted_advantage.detach().sum(dim=1)) - 0.001 * entropy_loss
        a_loss = a_loss * torch.tensor(action_repetitions, dtype=torch.float64, device=self.device)

        losses["actor_loss"] = a_loss.mean()

        total_loss = (critic_loss + a_loss).mean()
        losses["total_loss"] = total_loss

        for i in range(returns.size()[1]):
            actual_returns_key = "actual return " + str(i)
            losses[actual_returns_key] = returns[:, i].mean()

            advantage_key = "advantage " + str(i)
            losses[advantage_key] = advantage[:, i].mean()

            expected_returns_key = "expected returns " + str(i)
            losses[expected_returns_key] = state_values[:, i].mean()

        for i in range(probabilities.size()[1]):
            action_logging_key = "action " + str(i) + " probability"
            losses[action_logging_key] = probabilities[:, i].mean()

            for j in range(action_logits.size()[2]):
                action_logging_key = "action " + str(i) + " " + str(j) + " loggits"
                losses[action_logging_key] = action_logits[:, i, j].mean()

        return losses

    def _discounted_measurements(self, states, last_state, goal_items, finished):
        if finished:
            discounted_measurements_next_state = [0.] * self.goal_size
        else:
            _, discounted_measurements_next_state = self.forward([last_state])
            discounted_measurements_next_state = discounted_measurements_next_state[0].detach().cpu().numpy()


        next_state = last_state
        states_values = []

        for state in states[::-1]:
            current_state = state
            clipped_changes, unclipped_changes = inventory_change_as_goal_representation(goal_items, current_state["game_state"], next_state["game_state"])
            if self.clipped_reward:
                measurement_changes = clipped_changes
            else:
                measurement_changes = unclipped_changes

            discounted_measurements_next_state = self.reward_multiplicator * measurement_changes + discounted_measurements_next_state * self.gamma
            next_state = current_state
            states_values.append(discounted_measurements_next_state)

        states_values.reverse()
        return states_values


    def calculate_gradients(self, state_storage: StateStorage, update_type="normal"):
        all_states_action_rewards = state_storage.get_all_state_action_reward_tuples()
        actions = all_states_action_rewards["actions"]
        states = all_states_action_rewards["states"]
        clipped_rewards = np.array(all_states_action_rewards["clipped_rewards"])
        un_clipped_rewards = np.array(all_states_action_rewards["un_clipped_rewards"])

        if self.clipped_reward:
            rewards = clipped_rewards
        else:
            rewards = un_clipped_rewards
        rewards = np.array(rewards)

        last_state = state_storage.get_last_game_state_without_action()
        discounted_rewards = self._discounted_measurements(states, last_state, state_storage.goal_items, state_storage.finished)
        self.zero_grad(set_to_none=True)

        losses = self.loss_func(states, discounted_rewards, actions, update_type=update_type)
        losses["rewards"] = np.sum(rewards) / len(states)
        losses["clipped_rewards"] = np.sum(clipped_rewards) / len(states)
        losses["un_clipped_rewards"] = np.sum(un_clipped_rewards) / len(states)


        losses["clipped_max_rewards"] = np.max(clipped_rewards)
        losses["un_clipped_max_rewards"] = np.max(un_clipped_rewards)

        losses["clipped_min_rewards"] = np.min(clipped_rewards)
        losses["un_clipped_min_rewards"] = np.min(un_clipped_rewards)

        total_loss = losses["total_loss"]

        total_loss.backward()
        return losses

if __name__ == "__main__":
    with open('../Configs/model-parameters.json') as f:
        model_parameter_config = json.load(f)
    testBase = EACDFP(model_parameter_config)
    test = sum([param.nelement() for param in testBase.parameters()])
    print(test)

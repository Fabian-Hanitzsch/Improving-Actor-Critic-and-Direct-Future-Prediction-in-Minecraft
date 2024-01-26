from Models.Base import Base
import torch
import torch.nn as nn
import numpy as np
from utils.StateStorage import StateStorage
from utils.model_functions import get_possible_actions, action_dict_to_key
import time
import json
import random
import math
import torch.nn.functional as F

class A3C(Base):
    def __init__(self, config):
        super(A3C, self).__init__(config)
        self.possible_actions_id, self.possible_actions_name = get_possible_actions(self.recipes_by_name, self.equipable_items_by_name)
        self.action_hidden_1 = nn.Linear(self.combined_features_length, 128)
        self.action_head = nn.Linear(128, len(self.possible_actions_id))

        self.value_hidden_1 = nn.Linear(self.combined_features_length, 128)
        self.value_head = nn.Linear(128, 1)

        self.categorical_distribution = torch.distributions.Categorical




    def choose_action(self, x):
        # self.train(mode=False)
        time_start = time.time_ns()
        logits, state_value = self.forward([x])
        probabilities = F.softmax(logits, dim=1).data

        m = torch.distributions.Categorical(probabilities)
        action_id = m.sample().item()

        action = self.possible_actions_id[str(action_id)]
        time_end = time.time_ns()
        time_needed_milliseconds = (time_end - time_start) / 1000 / 1000
        if time_needed_milliseconds > 100:
            print("time needed to choose action:", str(time_needed_milliseconds) + "ms")

        return action, time_needed_milliseconds

    def forward(self, states):
        combined_featues = super(A3C, self)._partial_forward(states)
        action_logits, state_value = self._get_probabilities(combined_featues)
        return action_logits, state_value

    def _get_probabilities(self, combined_featues):
        action = self.action_hidden_1(combined_featues)
        action = nn.functional.relu(action)
        action = self.action_head(action)
        state_value = self.value_hidden_1(combined_featues)
        state_value = nn.functional.relu(state_value)
        state_value = self.value_head(state_value)

        return action, state_value

    def _get_value_of_state(self, state):
        _, state_value = self.forward(state)
        return state_value[0].item()

    def loss_func(self, states, actions, actual_values, update_type="normal"):
        action_ids = []
        action_repetitions = []
        # t
        # t-i i=128

        # s_t+1 = derzeitiger step -> nächster step: (s,a,r)
        # r_t =
        # s_t = vorheriger step
        # V(t-1) = r + yV(t)
        # s(t-1) ~~ s(t) -> V(t-1) ~~ V(t) -> (V(t-1) - yV(t))  = r

        for action in actions:
            action_repetitions.append(1 - int(action["repeated"]))
            action_key = action_dict_to_key(action)
            if action_key not in self.possible_actions_name:
                print("Missing action dict inside possible action ids")
                print(action)
                raise Exception
            action_ids.append(int(self.possible_actions_name[action_key]))

        if update_type=="staggered":
            with torch.no_grad():
                combined_featues = super(A3C, self)._partial_forward(states)
            action_logits, state_values = self._get_probabilities(combined_featues)
        else:
            action_logits, state_values = self.forward(states)


        action_probabilities = F.softmax(action_logits, dim=1)
        returns = torch.tensor(actual_values, device=self.device)[:, None]


        # avg_absolute_reward = returns.abs().mean(keepdims=true)
        # state_importance = returns.abs()/avg_absolute_reward

        advantage = returns - state_values

        losses = {"advantage": advantage.mean(),
                  "actual_returns": returns.mean(),
                  "expected_returns": state_values.mean()}


        critic_loss_available = False


        m = self.categorical_distribution(probs=action_probabilities)
        log_prob_actions = m.log_prob(torch.tensor(action_ids, dtype=torch.float64, device=self.device))[:, None]
        entropy_loss = m.entropy() * 0.001
        actor_loss = (-log_prob_actions * advantage.detach()) - entropy_loss[:, None]
        actor_loss = actor_loss * torch.tensor(action_repetitions, dtype=torch.float64, device=self.device)[:, None]
        losses ["entropy_loss"] =  entropy_loss.mean()
        losses["entropy_max"] = entropy_loss.max()
        losses ["entropy_min"] =  entropy_loss.min()
        losses ["actor_loss"] =  actor_loss.mean()


        critic_loss = advantage.pow(2)
        losses["critic_loss"] = critic_loss.mean()

        total_loss = critic_loss + actor_loss

        #total_loss = total_loss * torch.tensor(batch_size, dtype=torch.float64, device=self.device)[:, None]

        #total_loss = total_loss.sum() / torch.tensor(expected_batch_size, dtype=torch.float64, device=self.device)
        total_loss = total_loss.mean()
        losses["total_loss"] = total_loss

        for i in range(action_probabilities.size()[1]):
            action_logging_key = "action " + str(i) + " probability"
            losses[action_logging_key] = action_probabilities[:,i].mean()
            action_logging_key = "action " + str(i) + " loggits"
            losses[action_logging_key] = action_logits[:,i].mean()

        return losses

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

        last_state = state_storage.get_last_game_state_without_action()
        discounted_rewards = self._discounted_reward(states, rewards, last_state, state_storage.finished)

        losses = self.loss_func(states, actions, discounted_rewards, update_type=update_type)
        losses["rewards"] = np.sum(rewards) / len(states)
        losses["clipped_rewards"] = np.sum(clipped_rewards) / len(states)
        losses["un_clipped_rewards"] = np.sum(un_clipped_rewards) / len(states)


        losses["clipped_max_rewards"] = np.max(clipped_rewards)
        losses["un_clipped_max_rewards"] = np.max(un_clipped_rewards)

        losses["clipped_min_rewards"] = np.min(clipped_rewards)
        losses["un_clipped_min_rewards"] = np.min(un_clipped_rewards)

        total_loss = losses["total_loss"]
        self.zero_grad(set_to_none=True)
        total_loss.backward()
        return losses


if __name__ == "__main__":
    with open('../Configs/model-parameters.json') as f:
        model_parameter_config = json.load(f)
    testBase = A3C(model_parameter_config)
    test = sum([param.nelement() for param in testBase.parameters()])
    print(test)

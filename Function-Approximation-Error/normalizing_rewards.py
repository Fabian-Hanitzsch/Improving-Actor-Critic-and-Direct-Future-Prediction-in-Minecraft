import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


gamma = 0.9
steps_to_reward = 30
input_size = steps_to_reward * 2 + 1

neutral_state = [0.0] * input_size
neutral_state[steps_to_reward] = 1.0
game_states = [torch.tensor(neutral_state)]



visits_per_positive_reward_2 = 4
positive_reward = 1.0
negative_reward = -1.0

start_expected_value_base_state = ((positive_reward + negative_reward) / 2) * (gamma ** steps_to_reward)

for i in range(1, steps_to_reward+1):
    positive_state = [0.0] * input_size
    negative_state = [0.0] * input_size
    negative_state[steps_to_reward - i] = 1.0
    positive_state[steps_to_reward + i] = 1.0

    game_states.insert(0, torch.tensor(negative_state))
    game_states.append(torch.tensor(positive_state))


class Base(nn.Module):
    def __init__(self, normalization_type="normalized"):
        super(Base, self).__init__()
        self.normalization_type = normalization_type
        critic_bias = 0.05
        self.actor_output = nn.Linear(input_size, 2)
        nn.init.constant_(self.actor_output.bias, 0.5)
        nn.init.constant_(self.actor_output.weight, 0.5)

        self.critic_output = nn.Linear(3, 1)
        # initializing the critic with the true value of the states, avoiding 0 so that it can still learn something
        nn.init.constant_(self.critic_output.bias, critic_bias)
        self.gamma = gamma

        true_values = [-critic_bias] * input_size
        for i in range(1, steps_to_reward+1):
            true_values[steps_to_reward + i] += positive_reward * (self.gamma ** (steps_to_reward - i))
            true_values[i-1] += negative_reward * (self.gamma ** (i-1))

        true_values[steps_to_reward] += start_expected_value_base_state
        self.critic_output.weight.data = torch.tensor([true_values])

        self.softmax_layer = torch.nn.functional.softmax


    def forward(self, input_tensor):
        logits = self.actor_output(input_tensor)

        value = self.critic_output(input_tensor)
        return self.softmax_layer(logits), value

    def choose_action(self, state):
        action_probability, _ = self.forward(state)
        m = torch.distributions.Categorical(action_probability)
        return m.sample().item()

    def discounted_reward(self, rewards, last_state, finished=False):

        if not finished:
            _, value_next_state = self.forward(last_state)
            value_next_state = value_next_state.detach().cpu().numpy()[0]
        else:
            value_next_state = 0.0

        buffer_v_target = []  # last state only the approximation will be used
        for r in rewards[::-1]:  # reverse rewards
            value_next_state = r + self.gamma * value_next_state
            buffer_v_target.append(value_next_state)
        buffer_v_target.reverse()
        return buffer_v_target

    def calculate_loss(self, states, rewards, actions):
        action_probabilities, expected_values = self.forward(torch.stack(states))
        advantage = (torch.tensor(rewards)[:,None] - expected_values)
        c_loss = advantage.pow(2)
        m = torch.distributions.Categorical(probs=action_probabilities)
        a_loss = -m.log_prob(torch.tensor(actions))[:, None] * advantage.detach()
        total_loss = (c_loss + a_loss).mean()
        return total_loss

def train_model(normalization_type="normalized", max_episode=2000, color="green"):
    model = Base(normalization_type=normalization_type)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-03, alpha=0.99, eps=1e-08)
    total_rewards = []
    total_rewards_running_avg = []
    max_running_avg = 20

    if normalization_type=="normalized":
        label = "normalized rewards"
    else:
        label = "unnormalized rewards"

    for i in range(max_episode):
        if i%100 == 0:
            print("Started episode:", str(i))

        rewards = []
        actions = []
        states = []
        current_state_pointer = steps_to_reward
        state_direction = 0
        finished = False

        while not finished:
            state_tensor = game_states[current_state_pointer]
            action = model.choose_action(state_tensor)
            actions.append(action)
            states.append(state_tensor)
            reward = 0
            if current_state_pointer == steps_to_reward:
                if action == 0:
                    state_direction = -1
                else:
                    state_direction = 1
            current_state_pointer += state_direction
            if current_state_pointer < 0:
                finished = True
                reward = negative_reward
            elif current_state_pointer >= len(game_states):
                finished = True
                reward = positive_reward
            rewards.append(reward)

        discounted_reward = model.discounted_reward(rewards, None, finished)
        discounted_reward = np.array(discounted_reward)
        eps = 10e-6
        if normalization_type == "normalized":
            discounted_reward = (discounted_reward - discounted_reward.mean()) / (discounted_reward.std() + eps)
        else:
            discounted_reward = discounted_reward


        action_probabilities, _ = model.forward(game_states[steps_to_reward])
        action_1_probability = action_probabilities[1].detach().cpu().numpy()
        action_0_probability = action_probabilities[0].detach().cpu().numpy()

        total_expected_reward = action_0_probability * negative_reward + action_1_probability * positive_reward

        total_loss = model.calculate_loss(states, discounted_reward, actions)
        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_rewards_running_avg.append(total_expected_reward)
        if len(total_rewards_running_avg) > max_running_avg:
            total_rewards_running_avg.pop(0)

        total_rewards.append(sum(total_rewards_running_avg) / len(total_rewards_running_avg))

    plt.plot(total_rewards, color=color, label=label)

print("start normalized advantages training")
train_model(normalization_type="normalized", color="red")
print("start unnormalized advantages training")
train_model(normalization_type="nothing", color="green")

plt.grid()
plt.xlim(0)
plt.xlabel("Episode")
plt.ylabel("Average reward of episode")
plt.legend()
#plt.title("Estimation Error from normalizing rewards")
plt.savefig("created-images/normalizing_rewards.png")
plt.show()
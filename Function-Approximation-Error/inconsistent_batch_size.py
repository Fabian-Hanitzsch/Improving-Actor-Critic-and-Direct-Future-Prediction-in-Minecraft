import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

class Base(nn.Module):
    def __init__(self, update_method="mean"):
        super(Base, self).__init__()
        self.actor_output = nn.Linear(2, 2)
        nn.init.constant_(self.actor_output.bias, 0.5)
        nn.init.constant_(self.actor_output.weight, 0.5)

        self.critic_output = nn.Linear(2, 1)
        nn.init.constant_(self.critic_output.bias, 0.25)
        self.critic_output.weight.data = torch.tensor([[0.25, -0.25]])
        self.softmax_layer = torch.nn.functional.softmax
        self.gamma = 0.0
        self.update_method = update_method

    def forward(self, input_tensor):
        logits = self.actor_output(input_tensor)
        value = self.critic_output(input_tensor)
        return self.softmax_layer(logits), value

    def choose_action(self, state):
        action_probability, _ = self.forward(state)
        m = torch.distributions.Categorical(action_probability)
        return m.sample().item()

    def discounted_reward(self, rewards, last_state, finished):
        if finished:
            value_next_state = 0
        else:
            _, value_next_state = self.forward(torch.tensor(last_state))
            value_next_state = value_next_state.detach().cpu().numpy()[0]

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
        if self.update_method == "mean":
            total_loss = c_loss.mean()
        else:
            total_loss = c_loss.sum()
        return total_loss

mean_model = Base(update_method="mean")
mean_optimizer = torch.optim.RMSprop(mean_model.parameters(), lr=1e-03, alpha=0.99, eps=1e-03)

sum_model = Base(update_method="sum")
sum_optimizer = torch.optim.RMSprop(sum_model.parameters(), lr=1e-03, alpha=0.99, eps=1e-03)

current_state = 0


#for i in range(200):
game_states = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]


mean_model_expected_values = []
running_mean_model_expected_values = []

sum_model_expected_values = []
running_sum_model_expected_values = []

max_running_length = 1

action_1_probabilities = []

logging_rewards = []
max_episode = 1000
batch_size = 10
current_batch = 0

for i in range(max_episode):
    if i%1000 == 0:
        print("Started episode:", str(i))
    current_state = 0
    rewards = []
    actions = []
    states = []
    batch_rewards = []
    total_reward = 0
    current_batch = 0
    finished = False
    truncated = False
    episode_ended = False
    while not (finished or truncated):
        current_batch += 1
        next_state = current_state
        if i%2 == 0:
            action = 0
        else:
            action = 1

        reward = 0
        if action == 0 and current_state == 0:
            reward = 1
            finished = True
        elif action == 1 and current_state == 0:
            reward = 0
            next_state = 1


        if current_batch >= batch_size:
            truncated = True
        total_reward += reward
        rewards.append(reward)
        actions.append(action)
        states.append(game_states[current_state])
        current_state = next_state

    discounted_reward = mean_model.discounted_reward(rewards, game_states[current_state], finished)
    mean_total_loss = mean_model.calculate_loss(states, discounted_reward, actions)
    mean_total_loss.backward()
    mean_optimizer.step()
    mean_optimizer.zero_grad()

    sum_total_loss = sum_model.calculate_loss(states, discounted_reward, actions)
    sum_total_loss.backward()
    sum_optimizer.step()
    sum_optimizer.zero_grad()

    _, expected_value_1 = mean_model.forward(game_states[0])
    expected_value_1 = expected_value_1.detach().cpu().numpy()[0]
    running_mean_model_expected_values.append(expected_value_1)

    _, expected_value_2 = sum_model.forward(game_states[0])
    expected_value_2 = expected_value_2.detach().cpu().numpy()[0]
    running_sum_model_expected_values.append(expected_value_2)

    if len(running_mean_model_expected_values) > max_running_length:
        running_mean_model_expected_values.pop(0)
        running_sum_model_expected_values.pop(0)

    sum_model_expected_values.append(sum(running_sum_model_expected_values) / len(running_mean_model_expected_values))
    mean_model_expected_values.append(sum(running_mean_model_expected_values) / len(running_mean_model_expected_values))



print(mean_model_expected_values[-1])
plt.plot(mean_model_expected_values, color="green", label="mean of loss")
plt.plot(sum_model_expected_values, color="red", label="sum of loss")
plt.plot([0.5] * len(mean_model_expected_values), color="blue", label="true value")

plt.grid()
plt.xlim(0)
plt.xlabel("Episode")
plt.ylabel("Expected reward")
plt.legend()
#plt.title("Estimation Error by inconsistent Batch size")

plt.savefig("created-images/inconsitent_batch_size.png")
plt.show()

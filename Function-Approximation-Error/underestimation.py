import torch
from torch import nn
from utils.custom_rmsprop import customRMSProp
import matplotlib.pyplot as plt
import numpy as np

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.actor_output = nn.Linear(1, 2)
        nn.init.constant_(self.actor_output.bias, 0.05)
        self.actor_output.bias.data = torch.tensor([2.0, -1.0])

        nn.init.constant_(self.actor_output.weight, 0.5)

        self.critic_output = nn.Linear(1, 1)
        nn.init.constant_(self.critic_output.weight, 0.05)
        nn.init.constant_(self.critic_output.bias, 0.047)
        self.softmax_layer = torch.nn.functional.softmax
        self.gamma = 0.0

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
        for i in range(critic_update_steps):
            action_probabilities, expected_values = self.forward(torch.stack(states))
            advantage = (torch.tensor(rewards)[:, None] - expected_values)
            c_loss = advantage.pow(2)
            c_loss = c_loss.mean()
            c_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        action_probabilities, expected_values = self.forward(torch.stack(states))
        advantage = (torch.tensor(rewards)[:, None] - expected_values)
        m = torch.distributions.Categorical(probs=action_probabilities)
        a_loss = -m.log_prob(torch.tensor(actions))[:, None] * advantage.detach()
        a_loss = a_loss.mean()
        a_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model = Base()
#optimizer = customRMSProp(model.parameters(), lr=1e-03, alpha=0.0, beta=0.0, eps=1e-08)
optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-03, alpha=0.99, eps=10e-8)
#actor_optimizer = torch.optim.RMSprop(model.parameters(), lr=3e-03, alpha=0.99, eps=10e-8)
#optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.9),lr=1e-03,  eps=1e-08)

#for i in range(200):
game_state = torch.tensor([0.0])


total_rewards = []

differences = []
running_differences = []

differences2 = []
running_differences2 = []
previous_probability = None

action_1_probabilities = []
running_action_1_probabilities = []

running_avg = []
max_running_avg_length = 50



logging_rewards = []
max_episode = 600
epsiode_length = 1000
counter = 0
first_display = 2

critic_update_steps = 1

for i in range(max_episode):
    if i % 100 == 0:
        print("Started episode:", str(i))
    states = []
    rewards = []
    actions = []

    for j in range(epsiode_length):
        action = model.choose_action(game_state)
        actions.append(action)
        counter += 1
        if action == 0:
            reward = 0
        else:
            reward = 1
        rewards.append(reward)
        states.append(game_state)

    discounted_reward = model.discounted_reward(rewards, states, True)
    if i% 100 == 0:
        print(sum(discounted_reward) / len(discounted_reward))

    action_probabilities, expected_value_1 = model.forward(game_state)
    expected_value_1 = expected_value_1.detach().cpu().numpy()[0]
    action_1_probability = action_probabilities.detach().cpu().numpy()[1]
    action_1_probabilities.append(action_1_probability)
    difference = expected_value_1 - action_1_probability
    running_differences.append(difference)
    if len(running_differences) > max_running_avg_length:
        running_differences.pop(0)

    if i>=first_display:
        differences.append(sum(running_differences) / len(running_differences))
    model.calculate_loss(states, discounted_reward, actions)
    _, expected_value_1 = model.forward(game_state)
    expected_value_1 = expected_value_1.detach().cpu().numpy()[0]
    difference2 = expected_value_1 - action_1_probability
    running_differences2.append(difference2)
    if len(running_differences2) > max_running_avg_length:
        running_differences2.pop(0)
    if i >= first_display:
        differences2.append(sum(running_differences2) / len(running_differences2))



#plt.plot(total_rewards)
#
plt.plot(differences, color="red", label="pre update")
plt.plot(differences2, color="green", label="post update")
#plt.plot(action_1_probabilities, color="blue")
print(action_1_probabilities[-1])

plt.grid(visible=True)
plt.legend()
plt.xlim(0)
plt.xlabel("Episode")
plt.ylabel("Estimation error")
#plt.title("Underestimation of the critic")
plt.savefig("created-images/underestimation_error.png")

plt.show()

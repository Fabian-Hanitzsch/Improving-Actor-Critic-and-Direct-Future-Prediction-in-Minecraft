import torch
from torch import nn
from utils.custom_rmsprop import customRMSProp
import matplotlib.pyplot as plt
import numpy as np

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.actor_output = nn.Linear(1, 2)
        nn.init.constant_(self.actor_output.bias, 0.5)
        nn.init.constant_(self.actor_output.weight, 0.5)

        self.critic_output = nn.Linear(1, 1)
        nn.init.constant_(self.critic_output.weight, -0.005)
        nn.init.constant_(self.critic_output.bias, 0.005)
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
        action_probabilities, expected_values = self.forward(torch.stack(states))
        advantage = (torch.tensor(rewards)[:,None] - expected_values)
        c_loss = advantage.pow(2)
        m = torch.distributions.Categorical(probs=action_probabilities)
        a_loss = -m.log_prob(torch.tensor(actions))[:, None] * advantage.detach()
        #total_loss = (c_loss + a_loss).mean()
        total_loss = c_loss.mean()
        return total_loss

clipped_model = Base()
unclipped_model = Base()

#optimizer = customRMSProp(model.parameters(), lr=1e-03, alpha=0.0, beta=0.0, eps=1e-08)
clipped_optimizer = torch.optim.RMSprop(clipped_model.parameters(), lr=5e-03, alpha=0.99, eps=1e-08)
unclipped_optimizer = torch.optim.RMSprop(unclipped_model.parameters(), lr=5e-03, alpha=0.99, eps=1e-08)
#optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.9),lr=1e-03,  eps=1e-08)

#for i in range(200):
game_state = torch.tensor([0.0])


total_rewards = []

non_clipped_expected_values = []
un_clipped_running_avg = []
clipped_expected_values = []
clipped_running_avg = []
max_running_avg_length = 10

logging_rewards = []
max_episode = 2000
epsiode_length = 1
counter = 0

for i in range(max_episode):
    if i % 1000 == 0:
        print("Started episode:", str(i))
    states = []
    rewards = []
    actions = []

    for j in range(epsiode_length):
        actions.append(0)
        counter += 1
        if counter % 10 == 0:
            reward = -9
        else:
            reward = 1
        rewards.append(reward)
        states.append(game_state)

    discounted_reward = clipped_model.discounted_reward(rewards, states, True)
    total_loss = clipped_model.calculate_loss(states, discounted_reward, actions)
    total_loss.backward()
    _ = nn.utils.clip_grad_norm_(clipped_model.parameters(), max_norm=10, norm_type=2)
    clipped_optimizer.step()
    clipped_optimizer.zero_grad()

    total_loss = unclipped_model.calculate_loss(states, discounted_reward, actions)
    total_loss.backward()
    #_ = nn.utils.clip_grad_norm_(unclipped_model.parameters(), max_norm=10e9, norm_type=2)
    unclipped_optimizer.step()
    unclipped_optimizer.zero_grad()


    _, clipped_expected_value = clipped_model.forward(game_state)
    clipped_expected_value = clipped_expected_value.detach().cpu().numpy()[0]
    clipped_running_avg.append(clipped_expected_value)

    _, unclipped_expected_value = unclipped_model.forward(game_state)
    unclipped_expected_value = unclipped_expected_value.detach().cpu().numpy()[0]
    un_clipped_running_avg.append(unclipped_expected_value)

    if len(clipped_running_avg) > max_running_avg_length:
        clipped_running_avg.pop(0)
        un_clipped_running_avg.pop(0)

    clipped_expected_values.append(sum(clipped_running_avg) / len(clipped_running_avg))
    non_clipped_expected_values.append(sum(un_clipped_running_avg) / len(un_clipped_running_avg))


#plt.plot(total_rewards)

plt.plot(non_clipped_expected_values, color="green", label="unclipped gradients")
plt.plot(clipped_expected_values, color="red", label="clipped gradients")
plt.grid()
plt.legend()
plt.xlim(0)
#plt.ylim(0)
plt.xlabel("Episode")
plt.ylabel("Expected reward")
#plt.title("Estimation Error from clipping gradients")

plt.savefig("created-images/clipping_gradients.png")

plt.show()

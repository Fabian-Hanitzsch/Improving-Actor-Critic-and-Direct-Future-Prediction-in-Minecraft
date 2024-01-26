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
        nn.init.constant_(self.critic_output.weight, 0.005)
        nn.init.constant_(self.critic_output.bias, 0.05)
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

low_alpha_model = Base()
high_alpha_model = Base()

low_betas_adam_model = Base()
high_beta_1_adam_model = Base()
high_beta_2_adam_model = Base()


#optimizer = customRMSProp(model.parameters(), lr=1e-03, alpha=0.0, beta=0.0, eps=1e-08)
low_alpha_optimizer = torch.optim.RMSprop(low_alpha_model.parameters(), lr=5e-03, alpha=0.9, eps=1e-08)
high_alpha_optimizer = torch.optim.RMSprop(high_alpha_model.parameters(), lr=5e-03, alpha=0.99, eps=1e-08)

low_betas_adam_optimizer = torch.optim.Adam(low_betas_adam_model.parameters(), lr=5e-03, betas=(0.9, 0.9), eps=1e-08)
high_beta_1_adam_optimizer = torch.optim.Adam(high_beta_1_adam_model.parameters(), lr=5e-03, betas=(0.99, 0.9), eps=1e-08)
high_beta_2_adam_optimizer = torch.optim.Adam(high_beta_2_adam_model.parameters(), lr=5e-03, betas=(0.9, 0.99), eps=1e-08)



#for i in range(200):
game_state = torch.tensor([0.0])


total_rewards = []

rmsprop_099_expected_values = []
rmsprop_099_expected_values_running_avg = []

rmsprop_09_expected_values = []
rmsprop_09_expected_values_running_avg = []

adam_09_expected_values = []
adam_09_expected_values_running_vag = []

adam_high_beta_1_expected_values = []
adam_high_beta_1_expected_values_running_avg = []

adam_high_beta_2_expected_values = []
adam_high_beta_2_expected_values_running_avg = []


max_running_avg_length = 10

logging_rewards = []
max_episode = 2500
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

    discounted_reward = low_alpha_model.discounted_reward(rewards, states, True)
    total_loss = low_alpha_model.calculate_loss(states, discounted_reward, actions)
    total_loss.backward()
    low_alpha_optimizer.step()
    low_alpha_optimizer.zero_grad()

    total_loss = high_alpha_model.calculate_loss(states, discounted_reward, actions)
    total_loss.backward()
    high_alpha_optimizer.step()
    high_alpha_optimizer.zero_grad()

    total_loss = low_betas_adam_model.calculate_loss(states, discounted_reward, actions)
    total_loss.backward()
    low_betas_adam_optimizer.step()
    low_betas_adam_optimizer.zero_grad()

    total_loss = high_beta_1_adam_model.calculate_loss(states, discounted_reward, actions)
    total_loss.backward()
    high_beta_1_adam_optimizer.step()
    high_beta_1_adam_optimizer.zero_grad()

    total_loss = high_beta_2_adam_model.calculate_loss(states, discounted_reward, actions)
    total_loss.backward()
    high_beta_2_adam_optimizer.step()
    high_beta_2_adam_optimizer.zero_grad()


    _, clipped_expected_value = low_alpha_model.forward(game_state)
    clipped_expected_value = clipped_expected_value.detach().cpu().numpy()[0]
    rmsprop_09_expected_values_running_avg.append(clipped_expected_value)

    _, unclipped_expected_value = high_alpha_model.forward(game_state)
    unclipped_expected_value = unclipped_expected_value.detach().cpu().numpy()[0]
    rmsprop_099_expected_values_running_avg.append(unclipped_expected_value)


    _, adam_09_expected_value = low_betas_adam_model.forward(game_state)
    adam_09_expected_value = adam_09_expected_value.detach().cpu().numpy()[0]
    adam_09_expected_values_running_vag.append(adam_09_expected_value)

    _, adam_high_beta_1_expected_value = high_beta_1_adam_model.forward(game_state)
    adam_high_beta_1_expected_value = adam_high_beta_1_expected_value.detach().cpu().numpy()[0]
    adam_high_beta_1_expected_values_running_avg.append(adam_high_beta_1_expected_value)

    _, adam_high_beta_2_expected_value = high_beta_2_adam_model.forward(game_state)
    adam_high_beta_2_expected_value = adam_high_beta_2_expected_value.detach().cpu().numpy()[0]
    adam_high_beta_2_expected_values_running_avg.append(adam_high_beta_2_expected_value)

    if len(rmsprop_09_expected_values_running_avg) > max_running_avg_length:
        rmsprop_09_expected_values_running_avg.pop(0)
        rmsprop_099_expected_values_running_avg.pop(0)
        adam_09_expected_values_running_vag.pop(0)

        adam_high_beta_1_expected_values_running_avg.pop(0)
        adam_high_beta_2_expected_values_running_avg.pop(0)

    rmsprop_09_expected_values.append(sum(rmsprop_09_expected_values_running_avg) / len(rmsprop_09_expected_values_running_avg))
    rmsprop_099_expected_values.append(sum(rmsprop_099_expected_values_running_avg) / len(rmsprop_099_expected_values_running_avg))
    adam_09_expected_values.append(sum(adam_09_expected_values_running_vag) / len(adam_09_expected_values_running_vag))
    adam_high_beta_1_expected_values.append(sum(adam_high_beta_1_expected_values_running_avg) / len(adam_high_beta_1_expected_values_running_avg))
    adam_high_beta_2_expected_values.append(sum(adam_high_beta_2_expected_values_running_avg) / len(adam_high_beta_2_expected_values_running_avg))


#plt.plot(total_rewards)

plt.plot(rmsprop_099_expected_values, color="green", label=r"RMSProp $𝛼=0.99$", linestyle="--")
plt.plot(rmsprop_09_expected_values, color="green", label=r"RMSProp $𝛼=0.9$", linestyle="-")
plt.plot(adam_09_expected_values, color="red", label=r"Adam $𝛼=0.9, 𝛽=0.9$", linestyle="-")
plt.plot(adam_high_beta_1_expected_values, color="red", label=r"Adam $𝛼=0.99, 𝛽=0.9$", linestyle="--")
plt.plot(adam_high_beta_2_expected_values, color="blue", label=r"Adam $𝛼=0.9, 𝛽=0.9$", linestyle="-.")

plt.grid()
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Expected reward")
plt.xlim(0)
plt.ylim(0)
#plt.title("Estimation Error from optimizer")
plt.savefig("created-images/optimizer_overestimation.png")

plt.show()

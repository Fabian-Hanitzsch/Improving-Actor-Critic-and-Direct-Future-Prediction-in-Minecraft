import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.actor_output = nn.Linear(1, 2)
        nn.init.constant_(self.actor_output.bias, 0.5)
        nn.init.constant_(self.actor_output.weight, 0.5)

        self.critic_output = nn.Linear(1, 1)
        self.softmax_layer = torch.nn.functional.softmax
        self.gamma = 0.99

    def forward(self, input_tensor):
        logits = self.actor_output(input_tensor)

        value = self.critic_output(input_tensor)
        return self.softmax_layer(logits), value

    def choose_action(self, state):
        action_probability, _ = self.forward(state)
        m = torch.distributions.Categorical(action_probability)
        return m.sample().item()

    def discounted_reward(self, rewards, last_state):
        _, value_next_state = self.forward(torch.tensor(last_state))

        buffer_v_target = []  # last state only the approximation will be used
        for r in rewards[::-1]:  # reverse rewards
            value_next_state = r + self.gamma * value_next_state
            buffer_v_target.append(value_next_state)
        buffer_v_target.reverse()
        return buffer_v_target

    def calculate_loss(self, states, rewards, actions):
        action_probabilities, expected_values = self.forward(torch.tensor(states)[:,None])
        advantage = (torch.tensor(rewards)[:,None] - expected_values)
        c_loss = advantage.pow(2)
        m = torch.distributions.Categorical(probs=action_probabilities)
        a_loss = -m.log_prob(torch.tensor(actions))[:, None] * advantage.detach()
        total_loss = (c_loss + a_loss).mean()
        return total_loss

state_tensor = torch.tensor([1.0])

reset_action_0_probabilities = []
no_reset_action_0_probabilities = []

def train_model(type="reset", plot="probabilities", max_episode=1000, batch_size=10, episode_length=100, color="green"):
    model = Base()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-03, alpha=0.99, eps=1e-03)
    total_rewards = []
    total_rewards_running_avg = []
    max_running_avg = 200

    if type=="reset":
        label = "environment reset"
    else:
        label = "no environment reset"

    current_batch = 0

    for i in range(max_episode):
        if i%100 == 0:
            print("Started episode:", str(i))
        destroyed_blocks = 0
        theoretical_destroyed_blocks = 0
        rewards = []
        actions = []
        states = []
        theoretical_total_reward = 0

        for j in range(episode_length):
            current_batch += 1
            action = model.choose_action(state_tensor)
            reward = 0
            theoretical_reward = 0

            if action == 0:
                destroyed_blocks += 1
                theoretical_destroyed_blocks += 1
            elif action == 1 and destroyed_blocks > 0:
                destroyed_blocks -= 1
                reward = 1
            if action ==1 and theoretical_destroyed_blocks > 0:
                theoretical_destroyed_blocks -= 1
                theoretical_reward = 1

            theoretical_total_reward += theoretical_reward
            rewards.append(reward)
            actions.append(action)
            states.append(state_tensor)
            if current_batch % batch_size == 0:
                discounted_reward = model.discounted_reward(rewards, state_tensor)
                total_loss = model.calculate_loss(states, discounted_reward, actions)
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if type=="reset":
                    destroyed_blocks = 0
                current_batch = 0
                states = []
                rewards = []
                actions = []
                action_probabilities, _ = model.forward(state_tensor)

                if type=="reset":
                    reset_action_0_probabilities.append(action_probabilities[0].detach().cpu().numpy())
                else:
                    no_reset_action_0_probabilities.append(action_probabilities[0].detach().cpu().numpy())

        total_rewards_running_avg.append(theoretical_total_reward)
        if len(total_rewards) > max_running_avg:
            total_rewards_running_avg.pop(0)

        total_rewards.append(sum(total_rewards_running_avg) / len(total_rewards_running_avg))

    plt.plot(total_rewards, color=color, label=label)

print("start reset training")
train_model(type="reset", plot="rewards", color="green")
print("start no reset training")
train_model(type="nothing", plot="rewards", color="red")

plt.grid()
plt.xlim(0)
plt.xlabel("Episode")
plt.ylabel("Reward per episode")
plt.legend()
#plt.title("Unobservable supportive action environment")
plt.savefig("created-images/partial_observable_state_reward")
plt.show()

plt.clf()


plt.plot(reset_action_0_probabilities, color="green", label="environment reset")
plt.plot(no_reset_action_0_probabilities, color="red", label="no environment reset")
plt.grid()
plt.xlim(0)
plt.xlabel("Episode")
plt.ylabel("Probability to destroy a block")
plt.savefig("created-images/partial_observable_state_probability")



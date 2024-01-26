import numpy as np
import cv2

class StateStorage:
    def __init__(self, img_size, goals, initial_data=None, train_mode=True):
        self.train_mode = train_mode
        self.img_width, self.img_height, self.img_colors  = img_size
        self.goal_size = len(goals)
        self.goal_items = goals
        self.actions = []
        self.states = []
        self.un_clipped_rewards = []
        self.clipped_rewards = []
        self.forced_continued_digging = False

        self.current_goal = [0] * self.goal_size
        self.newest_image = np.zeros((self.img_height, self.img_width, self.img_colors), np.uint8)
        self.got_a_new_image = False

        self.current_state_counter = 0
        self.finished = False
        self.offset = 0

        if initial_data is not None:
            self.states.extend(initial_data["states"])
            self.actions.extend(initial_data["actions"])
            self.un_clipped_rewards.extend(initial_data["un_clipped_rewards"])
            self.clipped_rewards.extend(initial_data["clipped_rewards"])
            self.offset = len(initial_data)


    def get_last_n_states(self, n):
        first_index_states = len(self.states) - (n + 1)
        if first_index_states < 0:
            first_index_states = 0

        first_index_actions = len(self.states) - n
        if first_index_actions < 0:
            first_index_actions = 0

        data = {"states": self.states[first_index_states:],
                "actions": self.actions[first_index_actions:],
                "un_clipped_rewards": self.un_clipped_rewards[first_index_actions:],
                "clipped_rewards": self.clipped_rewards[first_index_actions:],
                "newest_image": self.newest_image}
        return data

    def remove_last_action(self):
        if len(self.actions) >= len(self.states) and len(self.actions) > 0:
            self.actions.pop()

    def add_img(self, img):
        self.got_a_new_image = True
        self.newest_image = img

    def reset(self, initial_data=None):
        self.__init__((self.img_width, self.img_height, self.img_colors), self.goal_items,
                      initial_data=initial_data, train_mode=self.train_mode)

    def get_game_state(self, idx):
        if idx >= len(self.states) or idx < 0:
            raise KeyError("measurement idx outside of range")
        return self.states[idx]["game_state"]



    def add_state(self, new_game_state):
        reusing_old_img = False
        newest_image = cv2.resize(self.newest_image, dsize=(self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)
        if not self.got_a_new_image:
            reusing_old_img = True
        new_state = {
                     "game_state": new_game_state,
                     "goal":self.current_goal,
                     "image": newest_image}

        self.got_a_new_image = False
        if self.train_mode:
            self.states.append(new_state)
        else:
            self.states = [new_state]

        return reusing_old_img

    def add_action(self, action):
        if self.train_mode:
            self.actions.append(action)

    def get_last_game_state_without_action(self):
        if len(self.states) == 0:
            return None

        return self.states[(len(self.states) - 1)]


    def get_all_state_action_reward_tuples(self):
        return {
            "states": self.states[:-1], # last state has no action and reward
            "actions": self.actions,
            "un_clipped_rewards": self.un_clipped_rewards,
            "clipped_rewards": self.clipped_rewards,
        }

    def add_rewards(self, un_clipped_reward, clipped_reward):
        self.un_clipped_rewards.append(un_clipped_reward)
        self.clipped_rewards.append(clipped_reward)

    def add_goal(self, goal):
        self.current_goal = goal

import numpy as np
import json


class Reward:
    def __init__(self):
        self.goal_items = {}
        self.item_types_count = 0
        self.recipes_by_id = {}
        self.recipes_by_name = {}
        self.items_by_name = {}
        self.items_by_id = {}
        self.most_valuable_item_gotten = 0
        self.destruction_weight = 0.1
        self.collection_weight = 1 - self.destruction_weight

        self.blocks_collected = {}
        self.blocks_collected_clipped = {}
        self.blocks_destroyed = {}


    def get_blocks_collected(self):
        return self.blocks_collected, self.blocks_collected_clipped

    def get_blocks_destroyed(self):
        return self.blocks_destroyed

    def is_goal_block(self, block_name):
        return block_name in self.goal_items.keys()


    def extract_reward(self, previous_state:dict, new_state:dict):
        if len(previous_state.keys()) == 0:
            return None, None

        if new_state["just_died"]:
            return -1, -1

        clipped_reward = 0
        un_clipped_reward = 0
        for goal_item in self.goal_items.keys():
            previous_item_count = 0
            if goal_item in previous_state["inventory"].keys():
                previous_item_count = previous_state["inventory"][goal_item]["amount"]

            previous_blocks_destroyed_count = 0
            if goal_item in previous_state["blocks_destroyed"].keys():
                previous_blocks_destroyed_count = previous_state["blocks_destroyed"][goal_item]

            current_item_count = 0
            if goal_item in new_state["inventory"].keys():
                current_item_count = new_state["inventory"][goal_item]["amount"]

            current_blocks_destroyed_count = 0
            if goal_item in new_state["blocks_destroyed"].keys():
                current_blocks_destroyed_count = new_state["blocks_destroyed"][goal_item]

            # we consider only a maximum of one item change in the clipped reward (multiple items can still change in the same step)
            item_count_change = current_item_count - previous_item_count
            if item_count_change < 0:
                item_count_change = 0

            self.blocks_collected[goal_item] += item_count_change

            item_count_change_clipped = max(0, min(1, item_count_change))
            self.blocks_collected_clipped[goal_item] += item_count_change_clipped

            blocks_destroyed_count = current_blocks_destroyed_count - previous_blocks_destroyed_count
            blocks_destroyed_count_clipped = max(0, min(1, blocks_destroyed_count))
            self.blocks_destroyed[goal_item] += blocks_destroyed_count

            clipped_reward += (item_count_change_clipped * self.collection_weight + self.destruction_weight *
                               blocks_destroyed_count_clipped) * self.goal_items[goal_item]
            un_clipped_reward += (item_count_change * self.collection_weight + self.destruction_weight * blocks_destroyed_count)\
                                 * self.goal_items[goal_item]
            if item_count_change > 0 and self.goal_items[goal_item] > self.most_valuable_item_gotten:
                self.most_valuable_item_gotten = self.goal_items[goal_item]

        return clipped_reward, un_clipped_reward

    def get_goal_size(self):
        return len(self.goal_items) * 2 + 1

    def get_goal_as_numerical_list(self):
        # death is currently hardcoded as -1 as reward
        count_list = [0] * len(self.goal_items) * 2 + [-1]

        # goal is structured as goal1_collection, goal1_destruction, goal2_collection, goal2_destruction, ...
        for index, goal_item in enumerate(self.goal_items.keys()):
            count_list[index * 2] = self.goal_items[goal_item] * self.collection_weight
            count_list[index * 2 + 1] = self.goal_items[goal_item] * self.destruction_weight

        goal = np.array(count_list)
        return goal



    def define_mission(self, mission_path, destruction_weight=0.1):
        self.destruction_weight = destruction_weight
        self.collection_weight = 1 - self.destruction_weight

        # clipping the max and min value to 1 and -1 respectively. allows for the zero-shot environment to be at 0 with the standard policy
        # destroying blocks reward, collecting blocks punishment
        self.destruction_weight = max(-1.0, min(1.0, self.destruction_weight))
        self.collection_weight = max(-1.0, min(1.0, self.collection_weight))


        with open(str(mission_path) + '/items-by-name.json') as f:
            self.items_by_name = json.load(f)

        self.items_by_id = {}
        for key in self.items_by_name:
            self.items_by_id[self.items_by_name[key]] = key

        with open(str(mission_path) + '/recipes-by-name.json') as f:
            self.recipes_by_name = json.load(f)

        self.recipes_by_id = {}
        for key in self.recipes_by_name:
            self.recipes_by_id[self.recipes_by_name[key]] = key

        self.item_types_count = len(self.items_by_name)
        with open(str(mission_path) + '/goal.json') as f:
            self.goal_items = json.load(f)

        for goal_item in self.goal_items.keys():
            self.blocks_collected[goal_item] = 0
            self.blocks_destroyed[goal_item] = 0
            self.blocks_collected_clipped[goal_item] = 0

if __name__ == "__main__":
    test_reward = Reward()
    test_reward.define_mission("../Missions/dirt")
    reward = test_reward.get_goal_as_numerical_list()
    print(reward)
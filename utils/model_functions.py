from copy import deepcopy
import json
import numpy as np


def action_dict_to_key(action_dict: dict) -> str:
    action_key = ""
    keys_of_relevance = ["forward", "left", "right", "sneak", "sprint", "back", "jump", "attack",
        "use", "insert", "take", "craft", "equip", "camera"]

    for key in keys_of_relevance:
        if key not in action_dict.keys():
            raise Exception("Missing Key Value in action_dict: " + str(key))

        action_key += str(action_dict[key])
    return action_key

def get_possible_actions(recipes_by_name, equipable_items_by_name) -> (dict, dict):
    with open('../Configs/base-movement-actions.json') as f:
        possible_actions_id = json.load(f)

    base_dict = {"forward": False, "left": False,
                 "right": False, "sneak": False,
                 "sprint": False, "back": False,
                 "jump": False, "attack": False, "use": False,

                 "insert": "none",
                 "take": "none",
                 "craft": "none",
                 "equip": "none", "camera": (0, 0)}

    for recipe in recipes_by_name:
        if not recipe == "none":
            current_dict = deepcopy(base_dict)
            current_dict["craft"] = recipe
            possible_actions_id[str(len(possible_actions_id))] = current_dict


    #action_names = ["insert", "take", "equip"]
    action_names = ["equip"]

    for item in equipable_items_by_name:
        if item == "none": continue
        for action_name in action_names:
            current_dict = deepcopy(base_dict)
            current_dict[action_name] = item
            possible_actions_id[str(len(possible_actions_id))] = current_dict

    possible_actions_name = {}
    for index in possible_actions_id:
        key = action_dict_to_key(possible_actions_id[index])
        possible_actions_name[key] = index

    return possible_actions_id, possible_actions_name

def inventory_change_as_goal_representation(goal_items:dict, previous_state: dict, new_state: dict):
    if new_state["just_died"]:
        count_list = [0] * len(goal_items) * 2 + [1]
        return np.array(count_list), np.array(count_list)

    count_list_clipped = []
    count_list_un_clipped = []


    for goal_item in goal_items.keys():
        previous_item_count = 0
        if goal_item in previous_state["inventory"].keys():
            previous_item_count = previous_state["inventory"][goal_item]["amount"]

        current_item_count = 0
        if goal_item in new_state["inventory"].keys():
            current_item_count = new_state["inventory"][goal_item]["amount"]

        # we consider only a maximum of one item change in the clipped reward (multiple items can still change in the same step)
        item_count_change = current_item_count - previous_item_count
        if item_count_change < 0:
            item_count_change = 0

        count_list_un_clipped.append(item_count_change)
        item_count_change_clipped = max(0, min(1, item_count_change))
        count_list_clipped.append(item_count_change_clipped)

        previous_blocks_destroyed_count = 0
        if goal_item in previous_state["blocks_destroyed"].keys():
            previous_blocks_destroyed_count = previous_state["blocks_destroyed"][goal_item]

        current_blocks_destroyed_count = 0
        if goal_item in new_state["blocks_destroyed"].keys():
            current_blocks_destroyed_count = new_state["blocks_destroyed"][goal_item]

        blocks_destroyed_count = current_blocks_destroyed_count - previous_blocks_destroyed_count
        count_list_clipped.append(blocks_destroyed_count)
        count_list_un_clipped.append(blocks_destroyed_count)

    count_list_clipped.append(0)
    count_list_un_clipped.append(0)

    return np.array(count_list_clipped), np.array(count_list_un_clipped)
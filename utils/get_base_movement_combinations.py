import json
from copy import deepcopy, copy


def is_valid(loop_dict):
    if loop_dict["back"] and loop_dict["forward"]:
        return False
    if loop_dict["right"] and loop_dict["left"]:
        return False
    if loop_dict["sprint"] and not loop_dict["forward"]:
        return False
    if loop_dict["sneak"] and (loop_dict["sprint"] or loop_dict["jump"]):
        return False
    if loop_dict["sneak"] and not (loop_dict["forward"] or loop_dict["back"] or loop_dict["right"] or loop_dict["left"]):
        return False

    return True

def loop_step(loop_dict, loop_names):
    global combinations
    global combination_count
    if len(loop_names) == 0:
        loop_dict["camera"] = (0,0)
        if not is_valid(loop_dict):
            return
        combinations[combination_count] = deepcopy(loop_dict)
        combination_count += 1
        return
    loop_name = loop_names.pop(0)

    for state in [True, False]:
        loop_dict[loop_name] = state
        loop_step(loop_dict, copy(loop_names))


def lower_count():
    base_dict = {"forward": False, "left": False,
                 "right": False, "sneak": False,
                 "sprint": False, "back": False,
                 "jump": False, "attack": False, "use": False,

                 "insert": "none",
                 "take": "none",
                 "craft": "none",
                 "equip": "none", "camera": (0, 0)}

    combination_count = 0
    combinations = {}

    single_state_names = ["attack"]
    movement_state_names = ["forward", "left", "right"]

    for state_name in single_state_names:
        new_dict = deepcopy(base_dict)
        new_dict[state_name] = True
        combinations[combination_count] = deepcopy(new_dict)
        combination_count += 1

    for state_name in movement_state_names:
        new_dict = deepcopy(base_dict)
        new_dict[state_name] = True
        combinations[combination_count] = deepcopy(new_dict)
        combination_count += 1

    new_dict = deepcopy(base_dict)
    new_dict["jump"] = True
    new_dict["forward"] = True
    combinations[combination_count] = deepcopy(new_dict)
    combination_count += 1

    for vertical_degree in [5, 25, -5, -25]:
        new_dict = deepcopy(base_dict)
        new_dict["camera"] = (vertical_degree, 0)
        combinations[combination_count] = deepcopy(new_dict)
        combination_count += 1
    for horizontal_degree in [5, 25, -5, -25]:
        new_dict = deepcopy(base_dict)
        new_dict["camera"] = (0, horizontal_degree)
        combinations[combination_count] = deepcopy(new_dict)
        combination_count += 1

    print(combination_count)
    print("finished")
    json_object = json.dumps(combinations, indent=4)

    with open('../Configs/base-movement-actions.json', "w") as f:
        f.write(json_object)

def main():
    base_dict = {"forward": False, "left": False,
                 "right": False, "sneak": False,
                 "sprint": False, "back": False,
                 "jump": False, "attack": False, "use": False,

                 "insert": "none",
                 "take": "none",
                 "craft": "none",
                 "equip": "none", "camera": (0, 0)}


    combination_count = 0
    combinations = {}

    state_names = ["forward", "back", "left", "right", "sprint", "sneak", "use", "attack", "jump"]
    forward_dict = {"insert": "none",
                 "take": "none",
                 "craft": "none",
                 "equip": "none"}
    loop_step(forward_dict, state_names)

    camera_count = 0
    for vertical_degree in [0, 1, 5, 25, -1, -5, -25]:
        for horizontal_degree in [0, 1, 5, 25, -1, -5, -25]:
            new_dict = deepcopy(base_dict)
            new_dict["camera"] = (vertical_degree, horizontal_degree)
            combinations[combination_count] = deepcopy(new_dict)
            combination_count += 1
            camera_count += 1
    print("camera_count:", str(camera_count))


    print(combination_count)
    print("finished")
    json_object = json.dumps(combinations, indent=4)

    with open('../Configs/base-movement-actions.json', "w") as f:
        f.write(json_object)

if __name__ == "__main__":
    #main()
    lower_count()

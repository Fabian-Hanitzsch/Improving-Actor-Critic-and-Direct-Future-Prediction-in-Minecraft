import json
import os
import shutil


path_to_entity_textures = "public/"
path_to_block_textures = "public/textures/1.19.1/items_textures.json"
path_to_json = "viewer/lib/entity/entities.json"
path_to_splash_potion = "public/textures/1.19.1/items/potion_bottle_splash.png"
path_to_empty_image = "empty_image.png"
path_to_basic_cat = "public/textures/1.19.1/entity/cat/british_shorthair.png"
path_to_firework = "firework.png"

def search_other_files(missing_path, texture):
    file_found = False
    file_path = ""
    for directory in os.listdir(path_to_entity_textures + "/textures"):
        texture_path = path_to_entity_textures + texture.replace("textures", "textures/" + directory) + ".png"
        if os.path.isfile(texture_path):
            file_found = True
            file_path = texture_path
            break
    if file_found:
        print("found missing texture: " + texture)
        print("path: ", file_path)
        shutil.copyfile(file_path, missing_path)


def main():
    total = 0
    all_entities = json.load(open(path_to_json))

    for entity in all_entities:
        texture_keys = all_entities[entity]["textures"]
        for texture in texture_keys.values():
            texture_path = path_to_entity_textures + texture.replace("textures", "textures/1.19.1") + ".png"
            if not os.path.isfile(texture_path):
                if "potion_bottle_splash" in texture_path:
                    shutil.copyfile(path_to_splash_potion, texture_path)
                    print("missing splash potion: ", texture_path)
                elif "none" in texture_path:
                    print("missing empty image: ", texture_path)
                    shutil.copyfile(path_to_empty_image, texture_path)
                elif "cat" in texture_path:
                    print("missing cat: ", texture_path)
                    shutil.copyfile(path_to_basic_cat, texture_path)
                elif "firework" in texture_path:
                    print("missing firework: ", texture_path)
                    shutil.copyfile(path_to_firework, texture_path)


                #search_other_files(texture_path, texture)
                total += 1

    all_blocks = json.load(open(path_to_block_textures))
    for entity in all_blocks:
        texture_path = entity["texture"]
        if not texture_path:
            continue
        texture_path = texture_path.replace("minecraft:", "")
        block_type = texture_path[0:5]
        if (block_type == "block"):
            texture_path = "textures/blocks/" + texture_path[6:]
        else:
            texture_path = "textures/items/" + texture_path[6:]

        texture_path = path_to_entity_textures + texture_path.replace("textures", "textures/1.19.1") + ".png"

        if not os.path.isfile(texture_path):
            print("missing: ", texture_path)
            #search_other_files(texture)
            total += 1

    print("total missing: ", total)


if __name__ == "__main__":
    main()
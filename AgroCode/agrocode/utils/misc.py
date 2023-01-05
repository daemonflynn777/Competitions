import yaml
from typing import Dict
import os
from tqdm import tqdm
import shutil
import agrocode.config as cfg

def load_yaml_safe(yaml_path: str) -> Dict:
    with open(yaml_path, "r") as stream:
        try:
            yaml_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_config

def sort_images(id_to_class: Dict[int, int], train_val_step: int) -> None:
    classes = list(set(list(id_to_class.values())))
    classes_counters = {c: 1 for c in classes}
    if os.path.exists(cfg.TRAIN_SORTED_IMAGES_PATH):
        shutil.rmtree(cfg.TRAIN_SORTED_IMAGES_PATH)
    if os.path.exists(cfg.VAL_IMAGES_PATH):
        shutil.rmtree(cfg.VAL_IMAGES_PATH)
    for c in classes:
        # os.mkdir(cfg.TRAIN_SORTED_IMAGES_PATH)
        if not os.path.exists(os.path.join(cfg.TRAIN_SORTED_IMAGES_PATH, str(c))):
            os.makedirs(os.path.join(cfg.TRAIN_SORTED_IMAGES_PATH, str(c)))
        # os.mkdir(cfg.TRAIN_SORTED_IMAGES_PATH)
        if not os.path.exists(os.path.join(cfg.VAL_IMAGES_PATH, str(c))):
            os.makedirs(os.path.join(cfg.VAL_IMAGES_PATH, str(c)))
    
    for id, c in tqdm(id_to_class.items()):
        src_path = os.path.join(cfg.TRAIN_IMAGES_PATH, f"{id}.png")
        if classes_counters[c] % train_val_step == 0:
            dest_path = os.path.join(cfg.VAL_IMAGES_PATH, str(c), f"{id}.png")
        else:
            dest_path = os.path.join(cfg.TRAIN_SORTED_IMAGES_PATH, str(c), f"{id}.png")
        shutil.copy(src_path, dest_path)
        classes_counters[c] += 1
    pass


def get_hidden_state(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
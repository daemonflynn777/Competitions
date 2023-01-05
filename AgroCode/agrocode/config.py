import os
from torchvision import transforms

PATTERNS = [r'[0-9]', '&quot;', r'[^\w\s]', r'\b\w{1,3}\b', '[a-zA-Z]']
PATTERNS_REPLACEMENTS = ['', '"', ' ', '', '']

ITEM_COL = "item_nm"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT_PATH = os.getcwd()
TITLES_PATH = os.path.join(ROOT_PATH, "data", "train.csv")
TEST_TITLES_PATH = os.path.join(ROOT_PATH, "data", "test.csv")
QUERIES_TITLES_PATH = os.path.join(ROOT_PATH, "data", "queries.csv")
SUBMISSION_PATH = os.path.join(ROOT_PATH, "data", "submission.csv")
TRAIN_IMAGES_PATH = os.path.join(ROOT_PATH, "data", "train")
TRAIN_SORTED_IMAGES_PATH = os.path.join(ROOT_PATH, "data", "train_sorted")
TEST_IMAGES_PATH = os.path.join(ROOT_PATH, "data", "test")
VAL_IMAGES_PATH = os.path.join(ROOT_PATH, "data", "val")
QUERIES_IMAGES_PATH = os.path.join(ROOT_PATH, "data", "queries")
WEIGHTS_PATH = os.path.join(ROOT_PATH, "weights", "swin.pth")
HF_TOKENIZER_PATH = os.path.join(ROOT_PATH, "weights", "hf_tokenizer")
HF_MODEL_PATH = os.path.join(ROOT_PATH, "weights", "hf_model")

BASE_TRANSFORMS = [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
TRANSFORMS_DICT = {
    "RandomRotation": transforms.RandomRotation(degrees = 135),
    "RandomResizedCrop": transforms.RandomResizedCrop(224, scale=(0.65, 0.9), ratio=(0.9, 1.1)),
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=1.0),
    "RandomVerticalFlip": transforms.RandomVerticalFlip(p=1.0),
}
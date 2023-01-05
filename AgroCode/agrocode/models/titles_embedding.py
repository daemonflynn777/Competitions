from transformers import AutoTokenizer, AutoModel
from typing import List
import torch
import agrocode.config as cfg


class TitlesEmbeddings():
    def __init__(self, titles: List[str]):
        # self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
        # self.tokenizer.save_pretrained(cfg.HF_TOKENIZER_PATH)
        # self.model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
        # self.model.save_pretrained(cfg.HF_MODEL_PATH)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.HF_TOKENIZER_PATH)
        self.model = AutoModel.from_pretrained(cfg.HF_MODEL_PATH)
        self.titles = titles

    def get_encoded_input(self):
        return self.tokenizer(self.titles, padding=True, truncation=True, max_length=16, return_tensors="pt")

    def run(self) -> torch.Tensor:
        encoded_input = self.get_encoded_input()
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)

        self.tokenizer = None
        self.model = None
        
        return embeddings
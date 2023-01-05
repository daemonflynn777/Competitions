import pandas as pd
import re
from pymorphy2 import MorphAnalyzer
import agrocode.config as cfg


class TitlesProcessing():
    def __init__(self, titles_path: str):
        self.titles_path = titles_path

    def get_titles(self) -> pd.DataFrame:
        return pd.read_csv(self.titles_path)

    @staticmethod
    def normalize_text(txt: str, morph: MorphAnalyzer) -> str:
        tokens = txt.split()
        tokens_norm = []
        for tk in tokens:
            tk = tk.strip().lower()
            tk_norm = morph.normal_forms(tk)[0]
            tokens_norm.append(tk_norm)
        return " ".join(tokens_norm)

    def clean_titles(self, titles_df: pd.DataFrame) -> pd.DataFrame:
        for pt, pt_replace in zip(cfg.PATTERNS, cfg.PATTERNS_REPLACEMENTS):
            titles_df[cfg.ITEM_COL] = titles_df[cfg.ITEM_COL].apply(lambda x: re.sub(pt, pt_replace, x))
        return titles_df

    def normalize_titles(self, titles_df: pd.DataFrame) -> pd.DataFrame:
        morph = MorphAnalyzer()
        titles_df[cfg.ITEM_COL] = titles_df[cfg.ITEM_COL].apply(lambda x: TitlesProcessing.normalize_text(x, morph))
        return titles_df

    def run(self) -> pd.DataFrame:
        titles_df = self.get_titles()
        titles_df = self.clean_titles(titles_df)
        titles_df = self.normalize_titles(titles_df)
        return titles_df

# solver/dictionary.py
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Set

import pandas as pd

from .logging_utils import get_logger

logger = get_logger()


@lru_cache(maxsize=1)
def load_jukugo_and_index(
    jukugo_path: str | Path,
) -> Tuple[pd.DataFrame, Dict[int, Set[int]], Dict[Tuple[int, int, str], Set[int]]]:
    """
    熟語辞書 CSV を読み込み、以下を返す:
      - df_jukugo: DataFrame (letters: List[str], count: 長さ, letter_1..)
      - index_len: {L -> 行インデックス集合}
      - index_len_pos_char: {(L, pos, ch) -> 行インデックス集合}
    """
    path = Path(jukugo_path)
    if not path.exists():
        raise FileNotFoundError(f"jukugo CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    if "letters" not in df.columns:
        raise ValueError("jukugo.csv には 'letters' 列が必要です。")

    # "['登','山']" → ["登","山"]
    df["letters"] = df["letters"].apply(ast.literal_eval)
    df["count"] = df["letters"].apply(len)
    maxlen = int(df["count"].max())

    # letter_1 .. letter_n を作成
    for i in range(maxlen):
        col = f"letter_{i+1}"
        df[col] = df["letters"].apply(
            lambda L, pos=i: L[pos] if pos < len(L) else ""
        )

    index_len: Dict[int, Set[int]] = {}
    index_len_pos_char: Dict[Tuple[int, int, str], Set[int]] = {}

    # index をすべて int に統一する（ここが重要）
    for idx_obj, row in df.iterrows():
        try:
            idx = int(str(idx_obj))
        except Exception:
            continue  # 想定外型の index は無視

        L = int(row["count"])
        index_len.setdefault(L, set()).add(idx)

        for pos, ch in enumerate(row["letters"]):
            index_len_pos_char.setdefault((L, pos, ch), set()).add(idx)

    logger.info(
        "Loaded jukugo dictionary from %s (rows=%d, maxlen=%d)",
        path,
        len(df),
        maxlen,
    )

    return df, index_len, index_len_pos_char

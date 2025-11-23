# solver/grid_model.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from .logging_utils import get_logger

logger = get_logger()


KANJI_RE = re.compile(r"[一-龯]")


def is_kanji(ch: str) -> bool:
    """漢字判定（CJK Unified Ideographs のざっくり判定）"""
    return isinstance(ch, str) and KANJI_RE.match(ch) is not None


def is_variable(ch: str) -> bool:
    """シンボル判定: '#1', '#2', ... の形式"""
    return isinstance(ch, str) and ch.startswith("#") and ch[1:].isdigit()


def convert_numbers_to_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame 内の「数字だけのセル」を '#<数字>' 形式のシンボルに変換する。
    それ以外（漢字・黒マス '■'・空白など）はそのまま。
    """

    def conv(x: Any) -> Any:
        s = str(x).strip()
        if s.isdigit():
            return f"#{int(s)}"
        return x

    return df.apply(lambda col: col.map(conv))


def parse_grid(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    グリッドから横・縦の「熟語候補（連続マス）」を抽出する。

    戻り値: [{'letters': [...], 'positions': [(i,j), ...]}, ...]
    """
    words: List[Dict[str, Any]] = []
    rows, cols = df.shape

    # 横方向
    for i in range(rows):
        j = 0
        while j < cols:
            if df.iat[i, j] != "■":
                letters: List[Any] = []
                pos: List[Tuple[int, int]] = []
                while j < cols and df.iat[i, j] != "■":
                    letters.append(df.iat[i, j])
                    pos.append((i, j))
                    j += 1
                if len(letters) > 1:
                    words.append({"letters": letters, "positions": pos})
            else:
                j += 1

    # 縦方向
    for j in range(cols):
        i = 0
        while i < rows:
            if df.iat[i, j] != "■":
                letters: List[Any] = []
                pos: List[Tuple[int, int]] = []
                while i < rows and df.iat[i, j] != "■":
                    letters.append(df.iat[i, j])
                    pos.append((i, j))
                    i += 1
                if len(letters) > 1:
                    words.append({"letters": letters, "positions": pos})
            else:
                i += 1

    return words

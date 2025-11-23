# solver/domains.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any, Set
from collections import Counter

import pandas as pd

from .grid_model import is_kanji, is_variable
from .logging_utils import get_logger

logger = get_logger()


def build_occurrences_and_domains(
    words: List[Dict[str, Any]],
    df_jukugo: pd.DataFrame,
    index_len: Dict[int, Set[int]],
    index_len_pos_char: Dict[Tuple[int, int, str], Set[int]],
):
    """
    - occurrences: { "#2": [(w_idx, pos), ...] }
    - domains    : { "#2": ["風", "海", "場", ...] }
    - solvable_word_indices: 辞書的に候補が1つ以上存在する単語のインデックス集合
    """
    occurrences: Dict[str, List[Tuple[int, int]]] = {}
    solvable_word_indices: Set[int] = set()

    # シンボル出現位置
    for w_idx, w in enumerate(words):
        for p_idx, ch in enumerate(w["letters"]):
            if is_variable(ch):
                occurrences.setdefault(ch, []).append((w_idx, p_idx))

    # 各単語が辞書に存在しうるか？
    for w_idx, w in enumerate(words):
        L = len(w["letters"])
        cand_ids = index_len.get(L, set()).copy()
        if not cand_ids:
            continue

        for pos, ch in enumerate(w["letters"]):
            if is_kanji(ch):
                cand_ids &= index_len_pos_char.get((L, pos, ch), set())

        if cand_ids:
            solvable_word_indices.add(w_idx)
        else:
            logger.info(
                "[domains] word_idx=%d '%s' has no dictionary candidates. treated as unsolvable.",
                w_idx,
                "".join(map(str, w["letters"])),
            )

    # ドメイン構築
    domains: Dict[str, List[str]] = {}

    for sym, occs in occurrences.items():
        common_chars: Set[str] | None = None
        singles: List[str] = []

        for w_idx, p_idx in occs:
            w = words[w_idx]
            L = len(w["letters"])
            cand_ids = index_len.get(L, set()).copy()
            if not cand_ids:
                continue

            # 既知漢字でフィルタ
            for pos, ch2 in enumerate(w["letters"]):
                if is_kanji(ch2):
                    cand_ids &= index_len_pos_char.get((L, pos, ch2), set())

            # 辞書マッチなしの単語は無視
            if not cand_ids:
                continue

            # 候補漢字集合
            raw_chars = {
                df_jukugo.at[i, f"letter_{p_idx+1}"]
                for i in cand_ids
                if f"letter_{p_idx+1}" in df_jukugo.columns
            }

            # ★ 空文字の除去（重要）および型をstrに統一（mypy対策）
            chars = {str(c) for c in raw_chars if c and not pd.isna(c)}

            if not chars:
                continue

            if len(chars) == 1:
                singles.append(next(iter(chars)))

            common_chars = chars if common_chars is None else (common_chars & chars)

        # singles による確定
        if singles:
            uniq = set(singles)
            if len(uniq) == 1:
                domains[sym] = [uniq.pop()]
                logger.debug("[domains] symbol=%s fixed by singles -> %s", sym, domains[sym][0])
                continue

        # 共通文字
        if common_chars:
            domains[sym] = sorted(common_chars)
        else:
            domains[sym] = []

        logger.debug(
            "[domains] symbol=%s domain_size=%d domain=%s",
            sym,
            len(domains[sym]),
            domains[sym],
        )

    # 追加フィルタ：共有漢字の軽減
    all_cands = [c for dom in domains.values() for c in dom]
    counts = Counter(all_cands)

    for sym, dom in domains.items():
        if len(dom) <= 1:
            continue
        filtered = [c for c in dom if counts[c] == 1]
        if filtered:
            domains[sym] = filtered

    return occurrences, domains, solvable_word_indices

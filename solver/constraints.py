# solver/constraints.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any, Set

import pandas as pd

from .grid_model import is_kanji, is_variable
from .logging_utils import get_logger

logger = get_logger()


def forward_check(
    assign: Dict[str, str],
    domains: Dict[str, List[str]],
    words: List[Dict[str, Any]],
    df_jukugo: pd.DataFrame,
    index_len,
    index_len_pos_char,
    solvable_indices: Set[int] | None = None,
):
    """
    forward checking:
    割り当て assign を仮に適用したときに、
    ・辞書にマッチする熟語候補が1つも残らない単語がないか確認
    ・各シンボルのドメインを「辞書的にありうる漢字」に絞り込む

    返り値: (ok, new_domains)
    """
    new_dom = {k: v[:] for k, v in domains.items()}

    for w_idx, w in enumerate(words):
        letters = w["letters"]
        L = len(letters)

        cand_ids = index_len.get(L, set()).copy()
        if not cand_ids:
            # そもそも辞書に L 文字熟語がない → 無視
            continue

        for pos, ch in enumerate(letters):
            if is_kanji(ch):
                cand_ids &= index_len_pos_char.get((L, pos, ch), set())
            elif is_variable(ch):
                if ch in assign:
                    val = assign[ch]
                    cand_ids &= index_len_pos_char.get((L, pos, val), set())

        if not cand_ids:
            # solvable フラグが指定されている場合のみ、厳格に落とす
            if solvable_indices is None or w_idx in solvable_indices:
                logger.debug(
                    "[forward_check] word_idx=%d became impossible under assign=%s",
                    w_idx,
                    assign,
                )
                return False, domains
            else:
                # もともと辞書にないと判定した単語 → 制約から除外
                continue

        # この単語の制約を使って、シンボルのドメインを少し絞る
        for pos, ch in enumerate(letters):
            if is_variable(ch) and ch not in assign:
                possible = {
                    df_jukugo.at[i, f"letter_{pos+1}"] for i in cand_ids
                }
                filtered = [v for v in new_dom[ch] if v in possible]
                if not filtered:
                    if solvable_indices is None or w_idx in solvable_indices:
                        logger.debug(
                            "[forward_check] symbol=%s in word_idx=%d "
                            "filtered to empty. assign=%s",
                            ch,
                            w_idx,
                            assign,
                        )
                        return False, domains
                    else:
                        # 辞書外単語に由来するなら無視
                        continue
                new_dom[ch] = filtered

    return True, new_dom

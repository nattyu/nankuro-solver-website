# solver/result_format.py
from __future__ import annotations

from typing import Any, Dict, List

from .grid_model import is_kanji, is_variable


def build_word_infos(
    words: List[Dict[str, Any]],
    assign: Dict[str, str],
    domains: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    UI 表示用に、単語ごとのテキストを作る:
      - 確定シンボル: その漢字
      - 未確定シンボルだが候補あり: [候補A/候補B]
      - 候補なし: □
    """
    out: List[Dict[str, Any]] = []

    for w in words:
        disp = []
        unresolved = False
        has_symbol = False

        for ch in w["letters"]:
            if is_kanji(ch):
                disp.append(ch)
            elif is_variable(ch):
                has_symbol = True
                if ch in assign and assign[ch]:
                    disp.append(assign[ch])
                elif ch in domains and domains[ch]:
                    cands = "/".join(domains[ch])
                    disp.append(f"[{cands}]")
                    unresolved = True
                else:
                    disp.append("□")
                    unresolved = True
            else:
                disp.append(str(ch))

        out.append(
            {
                "text": "".join(disp),
                "positions": w["positions"],
                "has_symbol": has_symbol,
                "unresolved": unresolved,
            }
        )

    return out


def build_mapping_list(
    original_domains: Dict[str, List[str]],
    assign: Dict[str, str],
    final_domains: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    シンボル → {kanji, candidates, num} の一覧を作る。
    """
    mapping: List[Dict[str, Any]] = []

    for sym in sorted(original_domains.keys(), key=lambda s: int(s[1:])):
        kanji = assign.get(sym, "")
        cands = final_domains.get(sym, [])

        mapping.append(
            {
                "symbol": sym,
                "kanji": kanji,
                "candidates": cands,
                "num": int(sym[1:]),
            }
        )

    return mapping

# solver/search.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any, Set

from .logging_utils import get_logger
from .constraints import forward_check

logger = get_logger()


def evaluate(domains: Dict[str, List[str]]) -> int:
    """
    評価関数:
    ・ドメインの総サイズ（小さいほど良い）
    """
    return sum(len(v) for v in domains.values())


def beam_search(
    occurrences: Dict[str, List[Tuple[int, int]]],
    domains: Dict[str, List[str]],
    words: List[Dict[str, Any]],
    df_jukugo,
    index_len,
    index_len_pos_char,
    beam_width: int = 10,
    use_tqdm: bool = False,
    solvable_word_indices: Set[int] | None = None,
):
    """
    ビームサーチでシンボル→漢字の割当てを探索する。
    戻り値: (assign, final_domains)
    """
    # 空ドメインを除いた初期状態
    doms = {k: v[:] for k, v in domains.items() if v}
    if not doms:
        logger.warning("All domains are empty or size 0. No assignment will be made.")
        return {}, domains

    vars_all = list(doms.keys())

    # 状態: (assign_dict, domains_dict, score)
    beam = [({}, doms, evaluate(doms))]

    depth_iter = range(len(vars_all))
    if use_tqdm:
        from tqdm import tqdm
        depth_iter = tqdm(depth_iter, desc="Depth")

    for depth in depth_iter:
        new_beam = []

        for assign, cur_dom, score in beam:
            if len(assign) == len(vars_all):
                return assign, cur_dom

            # まだ未割当のうち、最小ドメインの変数を選ぶ
            unassigned = [v for v in vars_all if v not in assign]
            var = min(unassigned, key=lambda v: len(cur_dom[v]))

            for val in cur_dom[var]:
                a2 = dict(assign)
                a2[var] = val

                d2 = {k: v[:] for k, v in cur_dom.items()}
                d2[var] = [val]

                ok, d3 = forward_check(
                    assign=a2,
                    domains=d2,
                    words=words,
                    df_jukugo=df_jukugo,
                    index_len=index_len,
                    index_len_pos_char=index_len_pos_char,
                    solvable_indices=solvable_word_indices,
                )
                if ok:
                    new_score = evaluate(d3)
                    new_beam.append((a2, d3, new_score))

        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_width]

        if not beam:
            logger.warning("[beam_search] beam became empty at depth=%d", depth)
            break

    if not beam:
        return {}, domains

    best_assign, best_dom, best_score = min(beam, key=lambda x: x[2])
    logger.info("[beam_search] finished. best_score=%d", best_score)
    return best_assign, best_dom

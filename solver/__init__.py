# solver/__init__.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_JUKUGO_PATH,
    USE_TQDM_DEFAULT,
)
from .logging_utils import (
    setup_solver_logger,
    teardown_solver_logger,
    get_logger,
)
from .dictionary import load_jukugo_and_index
from .grid_model import convert_numbers_to_variables, parse_grid
from .domains import build_occurrences_and_domains
from .search import beam_search
from .result_format import build_word_infos, build_mapping_list

logger = get_logger()


def solve_grid(
    df: pd.DataFrame,
    jukugo_path: str | Path = DEFAULT_JUKUGO_PATH,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    use_tqdm: bool = USE_TQDM_DEFAULT,
) -> Dict[str, Any]:
    """
    ナンクロ盤面 df を受け取り、自動解答の情報を返すメイン関数。

    戻り値の例:
    {
        "mapping": [
            {"symbol": "#1", "kanji": "天", "candidates": ["天"], "num": 1},
            ...
        ],
        "words": [
            {
              "text": "登[山/場]",
              "positions": [...],
              "has_symbol": True,
              "unresolved": True
            },
            ...
        ]
    }
    """
    logger, handler, log_file = setup_solver_logger()
    logger.info("===== solve_grid() START =====")
    logger.info("Grid shape: %s", df.shape)

    # ログファイルを solve 実行ごとに作成
    logger.info("Log file: %s", log_file)

    try:
        # 1) 熟語辞書とインデックスの読み込み
        df_jukugo, index_len, index_len_pos_char = load_jukugo_and_index(
            jukugo_path
        )

        # 2) 盤面の前処理（数字 → シンボル "#n"）
        df_sym = convert_numbers_to_variables(df)
        logger.debug("Converted grid (numbers -> symbols):\n%s", df_sym)

        # 3) グリッドから横・縦の熟語候補を抽出
        words = parse_grid(df_sym)
        logger.info("Parsed %d words from grid.", len(words))
        if words:
            sample = ["".join(map(str, w["letters"])) for w in words[:5]]
            logger.debug("Example words (up to 5): %s", sample)

        # 4) シンボル出現箇所・候補ドメイン・solvable フラグの構築
        occurrences, domains, solvable_word_indices = build_occurrences_and_domains(
            words, df_jukugo, index_len, index_len_pos_char
        )
        logger.info(
            "Domains: %d symbols, sizes=%s",
            len(domains),
            {k: len(v) for k, v in domains.items()},
        )
        logger.info(
            "Occurrences: %d symbols, total_occurrences=%d",
            len(occurrences),
            sum(len(v) for v in occurrences.values()),
        )
        logger.info(
            "Solvable words: %d / %d",
            len(solvable_word_indices),
            len(words),
        )

        empty_syms = [s for s, d in domains.items() if not d]
        if empty_syms:
            logger.warning(
                "No candidates for symbols %s. They will remain blank.",
                empty_syms,
            )

        # 5) ビームサーチで割り当て探索
        assign, final_domains = beam_search(
            occurrences=occurrences,
            domains=domains,
            words=words,
            df_jukugo=df_jukugo,
            index_len=index_len,
            index_len_pos_char=index_len_pos_char,
            beam_width=beam_width,
            use_tqdm=use_tqdm,
            solvable_word_indices=solvable_word_indices,
        )
        logger.info("Final assignment: %s", assign)

        # 5.5) BERT で未確定シンボルの候補を再ランキング（任意）
        from .config import BERT_ENABLED
        from .nlp.bert_scorer import refine_assign_with_bert

        if BERT_ENABLED:
            assign, final_domains = refine_assign_with_bert(
                assign=assign,
                domains=final_domains,
                words=words,
                occurrences=occurrences,
            )
            logger.info("Final assignment after BERT refine: %s", assign)

        # 6) 単語ごとの表示情報
        word_infos = build_word_infos(words, assign, final_domains)

        # 7) シンボル→漢字マッピング一覧
        mapping = build_mapping_list(domains, assign, final_domains)

        logger.info("Mapping list: %s", mapping)
        logger.info("===== solve_grid() END (log file: %s) =====", log_file)

        return {
            "mapping": mapping,
            "words": word_infos,
        }

    finally:
        teardown_solver_logger(handler)


# 互換用: 既存コードが from solver import solve を呼んでいても動くようにする
def solve(*args, **kwargs):
    return solve_grid(*args, **kwargs)

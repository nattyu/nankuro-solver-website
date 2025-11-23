# solver/logging_utils.py
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

from .config import LOG_DIR, LOG_LEVEL

LOGGER_NAME = "solver"


def get_logger() -> logging.Logger:
    """
    solver パッケージ共通のロガーを取得。
    （ハンドラの追加・削除は solve_grid() 実行単位で行う）
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)  # 詳細ログは取りつつ、出力はハンドラ側で絞る
    return logger


def setup_solver_logger() -> Tuple[logging.Logger, logging.Handler, Path]:
    """
    solve_grid() 実行ごとに専用のログファイルを作成し、
    ロガーに FileHandler をアタッチして返す。
    """
    LOG_DIR.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = LOG_DIR / f"solver_{ts}.log"

    logger = get_logger()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )
    fh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.info("=== New solve_grid() run. Log file: %s ===", log_file)

    return logger, fh, log_file


def teardown_solver_logger(handler: logging.Handler) -> None:
    """
    solve_grid() 終了時に FileHandler を外してクローズする。
    """
    logger = get_logger()
    logger.removeHandler(handler)
    handler.close()

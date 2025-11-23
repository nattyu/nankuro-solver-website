# solver/config.py
from __future__ import annotations

import logging
from pathlib import Path

# ==== 辞書・探索まわり ======================================================

# 熟語辞書 CSV のパス（プロジェクト直下に jukugo.csv がある前提）
DEFAULT_JUKUGO_PATH: Path = Path("jukugo.csv")

# ビームサーチの幅
DEFAULT_BEAM_WIDTH: int = 10

# tqdm を使うかどうか（基本 False）
USE_TQDM_DEFAULT: bool = False

# ログディレクトリ
LOG_DIR: Path = Path("solver_log")

# ログの基本レベル（コンソール / ファイル共通の最低レベル）
LOG_LEVEL: int = logging.INFO

# ==== BERT / 言語モデルまわり ==============================================

# 使用する日本語 BERT のモデル名
# 例: 東北大 BERT（事前学習済みの日本語 MaskedLM）
BERT_MODEL_NAME: str = "cl-tohoku/bert-base-japanese-v2"

# デバイス ("cpu" 推奨。GPU あるなら "cuda" でも可)
BERT_DEVICE: str = "cpu"

# 1つの記号(#n)について、何個まで熟語パターンを BERT に投げるか
# 大きくすると精度 ↑ / 時間 ↑
BERT_MAX_PATTERNS_PER_SYMBOL: int = 10

# BERT を使うかどうか（とりあえず ON にしておく）
BERT_ENABLED: bool = True


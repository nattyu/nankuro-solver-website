import ast
import logging
import re
from functools import lru_cache
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------
# ロガー設定（モジュール共通）
# ---------------------------------------
logger = logging.getLogger(__name__)

# solve() ごとに個別ファイルへログを書き出すためのディレクトリ
LOG_DIR = Path("solver_log")
LOG_DIR.mkdir(exist_ok=True)


# --- ユーティリティ関数 ---
def is_kanji(ch: str) -> bool:
    """漢字判定"""
    return isinstance(ch, str) and re.match(r"[一-龯]", ch) is not None


def is_variable(ch: str) -> bool:
    """#番号 のシンボル判定"""
    return isinstance(ch, str) and ch.startswith("#") and ch[1:].isdigit()


def convert_numbers_to_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    マスの中にある「数字」を '#数字' 形式のシンボルに変換する。
    それ以外（漢字・黒マス・空白など）はそのまま。
    """
    def conv(x):
        s = str(x).strip()
        if s.isdigit():
            return f"#{int(s)}"
        return x

    # pandas 1.x 互換のため applymap を使用
    return df.applymap(conv) # type: ignore


# --- 熟語辞書とインデックス構築（キャッシュ付き） ---
@lru_cache(maxsize=1)
def _load_jukugo_and_index(jukugo_path: str):
    """
    jukugo.csv を読み込み、以下を返す:
      df_jukugo : DataFrame
      index_len : {長さL: set(行idx)}
      index_len_pos_char : {(L, pos, ch): set(行idx)}
    """
    path = Path(jukugo_path)
    if not path.exists():
        raise FileNotFoundError(f"Jukugo dictionary not found: {jukugo_path}")

    logger.info("Loading jukugo dictionary from %s ...", jukugo_path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    # 'letters' 列が文字列のリストを表す文字列になっている前提で ast.literal_eval
    df["letters"] = df["letters"].apply(ast.literal_eval)
    df["count"] = df["letters"].apply(len)

    maxlen = df["count"].max()
    for i in range(maxlen):
        df[f"letter_{i+1}"] = df["letters"].apply(
            lambda L, k=i: L[k] if k < len(L) else ""
        )

    index_len = {}
    index_len_pos_char = {}

    for idx, row in df.iterrows():
        L = row["count"]
        letters = row["letters"]

        index_len.setdefault(L, set()).add(idx)
        for pos, ch in enumerate(letters):
            index_len_pos_char.setdefault((L, pos, ch), set()).add(idx)

    logger.info(
        "Loaded jukugo dictionary from %s (rows=%d)",
        jukugo_path,
        len(df),
    )
    return df, index_len, index_len_pos_char


# --- グリッド解析 ---
def parse_grid(df: pd.DataFrame) -> list:
    """
    グリッドから単語(横・縦)を抽出する。
    戻り値: [{"letters": [...], "positions": [(r,c), ...]}, ...]
    """
    words = []
    rows, cols = df.shape

    # 横方向
    for i in range(rows):
        j = 0
        while j < cols:
            if df.iat[i, j] != "■":
                letters, pos = [], []
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
                letters, pos = [], []
                while i < rows and df.iat[i, j] != "■":
                    letters.append(df.iat[i, j])
                    pos.append((i, j))
                    i += 1
                if len(letters) > 1:
                    words.append({"letters": letters, "positions": pos})
            else:
                i += 1

    return words


# --- ドメイン構築 ---
def build_occurrences_and_domains(words, df_jukugo, index_len, index_len_pos_char):
    """
    各シンボルについて:
      - どの単語のどの位置に現れるか (occurrences)
      - そのシンボルに入る可能性のある漢字の集合 (domains)
    を構築する。
    """
    occurrences = {}  # { "#3": [(word_idx, pos_idx), ...], ... }

    # まずシンボルがどの単語に何回出るかを集計
    for w_idx, w in enumerate(words):
        letters = w["letters"]
        for p_idx, ch in enumerate(letters):
            if is_variable(ch):
                occurrences.setdefault(ch, []).append((w_idx, p_idx))

    domains = {}  # { "#3": ["漢", "字", ...], ... }

    for sym, occs in occurrences.items():
        logger.debug("[build_domains] symbol=%s occurrences=%s", sym, occs)

        common = None
        singles = []

        for w_idx, p_idx in occs:
            w = words[w_idx]
            L = len(w["letters"])
            letters = w["letters"]

            # まずは長さ L の候補行を全て取る
            cand_ids = index_len.get(L, set()).copy()
            if not cand_ids:
                continue

            # 既知の漢字でフィルタ（シンボルはここでは制約に使わない）
            for pos, ch2 in enumerate(letters):
                if is_kanji(ch2):
                    cand_ids &= index_len_pos_char.get((L, pos, ch2), set())

            if not cand_ids:
                continue

            # cand_ids から、このシンボル位置に入り得る文字の集合を取る
            chars = {df_jukugo.at[i, "letters"][p_idx] for i in cand_ids}

            if not chars:
                continue

            if len(chars) == 1:
                singles.append(next(iter(chars)))

            common = chars if common is None else (common & chars)

        # singles が全て同じなら即決
        if singles:
            uniq = set(singles)
            if len(uniq) == 1:
                val = uniq.pop()
                domains[sym] = [val]
                logger.debug(
                    "[build_domains] symbol=%s fixed by singles -> %s",
                    sym, val
                )
                continue

        # それ以外は common を使う（無ければ空）
        if common:
            dom = sorted(common)
            domains[sym] = dom
            logger.debug(
                "[build_domains] symbol=%s domain_size=%d domain=%s",
                sym, len(dom), dom
            )
        else:
            domains[sym] = []
            logger.debug(
                "[build_domains] symbol=%s domain_size=0 domain=[]",
                sym
            )

    # ドメインの重複候補を除外して、ユニークな漢字候補だけ残す処理（弱い制約）
    all_cand = [k for dom in domains.values() for k in dom]
    counts = {}
    for k in all_cand:
        counts[k] = counts.get(k, 0) + 1

    for sym, dom in domains.items():
        filtered = [k for k in dom if counts.get(k, 0) == 1]
        if filtered:
            logger.debug(
                "[build_domains] symbol=%s domain filtered by uniqueness: %s -> %s",
                sym, dom, filtered
            )
            domains[sym] = filtered

    logger.info(
        "Domains built: %d symbols, sizes=%s",
        len(domains),
        {k: len(v) for k, v in domains.items()}
    )

    return occurrences, domains


# --- 単語ごとの「辞書に載りうるか」の前計算 ---
def compute_word_solvable(words, index_len, index_len_pos_char):
    """
    各単語について「初期状態で辞書候補があるかどうか」を判定する。
    （シンボルは自由、既知の漢字だけで絞り込み）

    戻り値:
      word_solvable: list[bool] (len = len(words))
    """
    solvable = []

    for w_idx, w in enumerate(words):
        letters = w["letters"]
        L = len(letters)
        cand_ids = index_len.get(L, set()).copy()

        if not cand_ids:
            solvable.append(False)
            continue

        for pos, ch in enumerate(letters):
            if is_kanji(ch):
                cand_ids &= index_len_pos_char.get((L, pos, ch), set())
                if not cand_ids:
                    break

        ok = bool(cand_ids)
        solvable.append(ok)
        logger.debug(
            "[compute_word_solvable] word_idx=%d '%s' solvable=%s",
            w_idx, "".join(map(str, letters)), ok
        )

    logger.info(
        "Word solvable stats: total=%d, solvable=%d, unsolvable=%d",
        len(words),
        sum(solvable),
        len(words) - sum(solvable)
    )
    return solvable


# --- 制約伝播 ---
def forward_check(
    assign,
    domains,
    words,
    df_jukugo,
    index_len,
    index_len_pos_char,
    word_solvable,
    symbol_to_words,
    symbol_solvable_counts,
    threshold: float,
):
    """
    現在の割り当て assign とドメイン domains に対して前向きチェックを行う。

    ロジック（ユーザー意図）:
      - 各シンボル (#番号) について、
        そのシンボルを含む「辞書に載る可能性のある単語」のうち、
        何割が「現在の割り当てで辞書候補を持てているか」を計算する。
      - その割合が threshold（例: 0.8）以上なら OK、
        未満ならこの assign は失敗として枝刈りする。

    ここでは
      word_solvable[w_idx] == True の単語のみを「辞書に載りうる単語」とみなす。
    """
    # domains はこのビームでの局所ドメインなのでディープコピー
    new_dom = {k: v[:] for k, v in domains.items()}

    # 各シンボルごとに「辞書的に整合している単語数」をカウント
    # 分母は symbol_solvable_counts[sym] で事前計算済み。
    matched_counts = {
        sym: 0
        for sym in assign.keys()
        if symbol_solvable_counts.get(sym, 0) > 0
    }

    # 単語ごとに cand_ids を計算し、
    #   - 辞書候補が存在するか（word_solvable & cand_ids非空）
    #   - ドメイン縮小（未割り当てシンボル）を行う
    for w_idx, w in enumerate(words):
        letters = w["letters"]
        L = len(letters)

        # まず長さでざっくり候補を取る
        cand_ids = index_len.get(L, set()).copy()
        if not cand_ids:
            # そもそもこの長さの熟語が辞書に無い → この単語は辞書的制約に使わない
            continue

        # 既知の漢字 & 割り当て済みシンボルでフィルタ
        for pos, ch in enumerate(letters):
            if is_kanji(ch):
                cand_ids &= index_len_pos_char.get((L, pos, ch), set())
            elif ch in assign:
                cand_ids &= index_len_pos_char.get((L, pos, assign[ch]), set())

            if not cand_ids:
                break

        if not word_solvable[w_idx]:
            # 初期状態でそもそも辞書候補ゼロの単語は、
            # 「辞書に載らない固有名詞など」とみなし、スコアの分母にも分子にも入れない
            continue

        # cand_ids が非空なら、現在の assign でも「辞書候補がまだ残っている」
        matched = bool(cand_ids)

        # この単語に含まれるシンボルのうち、
        # 「既に割り当てられている」かつ「辞書的に意味を持つシンボル」についてカウント
        for ch in letters:
            if not is_variable(ch):
                continue
            if ch not in matched_counts:
                continue
            # ch がこの単語の中に現れていて、かつこの単語が matched なら 1 加算
            if matched:
                matched_counts[ch] += 1

        # ドメイン縮小:
        #   この単語に対して cand_ids が非空であれば、
        #   まだ割り当てていないシンボルのドメインを「この熟語候補で取りうる漢字」に絞る。
        if cand_ids:
            for pos, ch in enumerate(letters):
                if is_variable(ch) and ch not in assign:
                    if ch not in new_dom or not new_dom[ch]:
                        continue
                    possible = {df_jukugo.at[i, "letters"][pos] for i in cand_ids}
                    filt = [x for x in new_dom[ch] if x in possible]
                    if not filt:
                        # このシンボルにとって辞書的候補が一切なくなった → この枝は成立しない
                        logger.debug(
                            "[forward_check] domain -> empty: symbol=%s in word_idx=%d",
                            ch, w_idx
                        )
                        return False, domains
                    if len(filt) < len(new_dom[ch]):
                        logger.debug(
                            "[forward_check] domain shrink: symbol=%s %d -> %d",
                            ch, len(new_dom[ch]), len(filt)
                        )
                        new_dom[ch] = filt

    # シンボルごとの「辞書一致率」を計算し、threshold を満たさないものがあれば失敗
    for sym, matched in matched_counts.items():
        denom = symbol_solvable_counts.get(sym, 0)
        if denom <= 0:
            continue  # このシンボルにとって辞書的な単語が一つもない場合はスキップ

        ratio = matched / denom
        logger.debug(
            "[forward_check] symbol=%s matched=%d denom=%d ratio=%.3f",
            sym, matched, denom, ratio
        )

        if ratio < threshold:
            logger.debug(
                "[forward_check] FAIL: symbol=%s ratio=%.3f < threshold=%.3f (assign=%s)",
                sym, ratio, threshold, assign
            )
            return False, domains

    return True, new_dom


# --- 評価関数 ---
def evaluate(domains):
    """
    単純に「ドメインサイズの総和」（小さいほど制約が強い）を評価値とする。
    必要に応じて、今後「完成している単語数」などをスコアに組み込む余地あり。
    """
    return sum(len(v) for v in domains.values() if v)


# --- ビームサーチ ---
def beam_search(
    occurrences,  # 現状は使用していないがインタフェース維持
    domains,
    words,
    df_jukugo,
    index_len,
    index_len_pos_char,
    word_solvable,
    symbol_to_words,
    symbol_solvable_counts,
    threshold: float,
    beam_width: int = 10,
    use_tqdm: bool = False,
):
    """
    シンボル → 漢字の割り当てをビームサーチで探索する。

    ・「ドメインが非空のシンボル」だけを探索対象とする
    ・forward_check を、漢字を1つ割り当てるたびに呼び出し、
      各シンボルごとの「辞書一致率 >= threshold」を満たしているかを毎回チェックする
    """

    # ドメインが非空のシンボルだけ対象
    doms = {k: v[:] for k, v in domains.items() if v}
    if not doms:
        logger.warning(
            "[beam_search] All domains are empty or size 0. No assignment will be made."
        )
        return {}, domains

    vars_all = list(doms.keys())
    logger.info("[beam_search] start: vars_all=%s", vars_all)

    # ビームの要素: (assign, domains, score)
    beam = [({}, doms, evaluate(doms))]

    depth_iter = range(len(vars_all))
    if use_tqdm:
        depth_iter = tqdm(depth_iter, desc="Depth")

    for depth in depth_iter:
        new_beam = []
        logger.debug(
            "[beam_search] depth=%d beam_size=%d",
            depth, len(beam)
        )

        for assign, cur_dom, score in beam:
            # まだ割り当て可能なシンボル（ドメイン非空 & 未割り当て）を列挙
            candidate_vars = [
                v for v in cur_dom.keys() if cur_dom[v] and v not in assign
            ]

            # これ以上このビームからは進めない → 現状の部分解をそのまま保持
            if not candidate_vars:
                logger.debug(
                    "[beam_search] depth=%d no more assignable vars, keep assign=%s",
                    depth, assign
                )
                new_beam.append((assign, cur_dom, score))
                continue

            # 最小ドメイン（MRV）を持つシンボルを優先的に割り当てる
            var = min(candidate_vars, key=lambda v: len(cur_dom[v]))
            dom_v = cur_dom[var]

            logger.debug(
                "[beam_search] depth=%d choose var=%s domain_size=%d domain=%s",
                depth, var, len(dom_v), dom_v
            )

            # ドメインの各候補を試す
            for c in dom_v:
                a2 = assign.copy()
                a2[var] = c

                d2 = {k: v[:] for k, v in cur_dom.items()}
                d2[var] = [c]

                logger.debug(
                    "[beam_search] depth=%d try var=%s -> %s (current_assign=%s)",
                    depth, var, c, a2
                )

                ok, d3 = forward_check(
                    a2,
                    d2,
                    words,
                    df_jukugo,
                    index_len,
                    index_len_pos_char,
                    word_solvable,
                    symbol_to_words,
                    symbol_solvable_counts,
                    threshold,
                )
                if ok:
                    sc = evaluate(d3)
                    logger.debug(
                        "[beam_search] depth=%d success assign=%s score=%d",
                        depth, a2, sc
                    )
                    new_beam.append((a2, d3, sc))
                else:
                    logger.debug(
                        "[beam_search] depth=%d prune assign=%s by forward_check",
                        depth, a2
                    )

        if not new_beam:
            logger.warning(
                "[beam_search] beam has become empty at depth=%d",
                depth
            )
            break

        # スコアでソートし、上位 beam_width 個だけ残す
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_width]

    if not beam:
        logger.warning("[beam_search] no assignment found at all.")
        return {}, domains

    # 最後に残っているビームのうち「スコア最小」の部分解を使う
    best_assign, best_dom, best_score = min(beam, key=lambda x: x[2])

    logger.info(
        "[beam_search] finished. best_score=%d best_assign=%s",
        best_score, best_assign
    )

    # best_dom はこのコンポーネント内の「最終ドメイン」
    return best_assign, best_dom


# --- シンボルグラフ（連結成分）の構築 ---
def build_symbol_components(words, domains):
    """
    ドメイン非空のシンボルだけをノードとし、
    同じ単語内に現れるシンボル同士にエッジを張ったグラフから
    連結成分（コンポーネント）のリストを返す。

    戻り値:
      [
        ['#2', '#15', '#49', ...],   # コンポーネント1
        ['#30', '#37', ...],         # コンポーネント2
        ...
      ]
    """
    active_syms = {s for s, d in domains.items() if d}
    logger.info(
        "[build_symbol_components] active_syms=%s", active_syms
    )
    if not active_syms:
        return []

    # 隣接リストを構築
    adj = {s: set() for s in active_syms}

    for w in words:
        letters = w["letters"]
        syms_in_word = [ch for ch in letters if is_variable(ch) and ch in active_syms]

        # その単語の中で一緒に現れたシンボル同士を完全グラフでつなぐ
        for i in range(len(syms_in_word)):
            for j in range(i + 1, len(syms_in_word)):
                a, b = syms_in_word[i], syms_in_word[j]
                adj[a].add(b)
                adj[b].add(a)

    # DFS/BFS で連結成分を抽出
    visited = set()
    components = []

    for s in active_syms:
        if s in visited:
            continue
        stack = [s]
        visited.add(s)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        components.append(sorted(comp, key=lambda x: int(x[1:])))

    logger.info(
        "[build_symbol_components] components=%s",
        components
    )
    return components


# --- 単語ごとの表示用情報構築 ---
def build_word_infos(words, assign, domains):
    """
    各単語について、表示用情報を構築する。

    assign : 確定したシンボル→漢字マッピング
    domains: 最終ドメイン（未確定シンボルに対する候補リスト）

    戻り値: [
      {
        "text": "四[漢/館]熟語",
        "positions": [...],
        "has_symbol": True/False,
        "unresolved": True/False,  # 未確定（候補複数 or 候補なし）を含むか？
      },
      ...
    ]
    """
    word_infos = []

    for w in words:
        disp_letters = []
        has_symbol = False
        unresolved = False

        for ch in w["letters"]:
            if is_kanji(ch):
                disp_letters.append(ch)
            elif is_variable(ch):
                has_symbol = True

                # 1. 確定している場合
                if ch in assign and assign[ch]:
                    disp_letters.append(assign[ch])

                # 2. 未確定だが、候補（ドメイン）が残っている場合
                elif ch in domains and domains[ch]:
                    # 表記例: "[漢/館]" のように候補を列挙して表示
                    cands = "/".join(domains[ch])
                    disp_letters.append(f"[{cands}]")
                    unresolved = True

                # 3. 候補もない（解けない）場合
                else:
                    disp_letters.append("□")
                    unresolved = True
            else:
                disp_letters.append(str(ch))

        text = "".join(disp_letters)
        word_infos.append(
            {
                "text": text,
                "positions": w["positions"],
                "has_symbol": has_symbol,
                "unresolved": unresolved,
            }
        )

    return word_infos


# --- メイン解答器 ---
def solve(
    df: pd.DataFrame,
    jukugo_path: str = "jukugo.csv",
    beam_width: int = 10,
    use_tqdm: bool = False,
    threshold: float = 0.8,  # ★ ここで「8割以上ならOK」を指定（0.0〜1.0で調整可能）
) -> dict:
    """
    グリッド DataFrame と熟語辞書パスを受け取り、
    ・シンボル → 漢字割り当て（mapping）
    ・各単語の仮置き熟語情報（words）
    を返す。

    戻り値:
    {
        "mapping": [
            {
                "symbol": "#1",
                "kanji": "漢" または "",                 # 確定文字（空なら未確定）
                "candidates": ["漢", "館", ...] or [],  # 最終的な候補リスト
                "num": 1
            },
            ...
        ],
        "words": [
            {
              "text": "四[漢/館]熟語",
              "positions": [...],
              "has_symbol": True,
              "unresolved": True
            },
            ...
        ]
    }

    ・各シンボルごとに「辞書に載る可能性のある単語」のうち
      threshold（例: 0.8=8割）以上が辞書と整合していれば OK
    ・forward_check をビームサーチの各ステップで呼び出して、
      この条件を毎回チェックする
    """
    # --- 実行ごとのファイルロガーをセットアップ ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = LOG_DIR / f"solver_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    logger.info("===== solve() START =====")
    logger.info(
        "Grid shape: %s, jukugo_path=%s, beam_width=%d, use_tqdm=%s, threshold=%.3f",
        df.shape, jukugo_path, beam_width, use_tqdm, threshold
    )

    try:
        # 熟語辞書とインデックスはキャッシュ付きでロード
        df_jukugo, idx_len, idx_pos = _load_jukugo_and_index(jukugo_path)

        # 数字を '#n' 形式のシンボルに変換
        df2 = convert_numbers_to_variables(df)
        logger.debug("Converted grid (numbers -> symbols):\n%s", df2)

        # グリッドから単語（横・縦）を抽出
        words = parse_grid(df2)
        logger.info("Parsed %d words from grid.", len(words))
        if words:
            logger.debug(
                "Example words (up to 5): %s",
                ["".join(map(str, w["letters"])) for w in words[:5]]
            )

        # 各単語が「辞書に載りうるか」を事前計算
        word_solvable = compute_word_solvable(words, idx_len, idx_pos)

        # シンボルの出現位置とドメインを構築
        occs, doms = build_occurrences_and_domains(words, df_jukugo, idx_len, idx_pos)

        logger.info(
            "Occurrences built: %d symbols, total_occurrences=%d",
            len(occs),
            sum(len(v) for v in occs.values()),
        )

        # 空ドメイン（候補ゼロ）のシンボルがあれば情報だけ出す（失敗ではない）
        empty_syms = [s for s, d in doms.items() if not d]
        if empty_syms:
            logger.warning(
                "No candidates for symbols %s. They will remain blank in the result unless inferred indirectly.",
                empty_syms,
            )

        # シンボルごとに「辞書に載りうる単語の個数（分母）」を計算
        symbol_to_words = {
            sym: sorted({w_idx for (w_idx, _) in occs.get(sym, [])})
            for sym in occs.keys()
        }
        symbol_solvable_counts = {
            sym: sum(1 for w_idx in word_list if word_solvable[w_idx])
            for sym, word_list in symbol_to_words.items()
        }
        logger.info(
            "Symbol solvable counts: %s",
            symbol_solvable_counts
        )

        # シンボルグラフの連結成分ごとにビームサーチを実行
        components = build_symbol_components(words, doms)
        logger.info("Components: %s", components)

        assign_global = {}
        # 最終ドメイン（候補リスト）はまず初期ドメインでコピーしておき、
        # コンポーネントごとに更新していく
        final_domains = {k: v[:] for k, v in doms.items()}

        if components:
            for idx, comp_syms in enumerate(components):
                logger.info(
                    "[component %d] symbols=%s", idx, comp_syms
                )
                # このコンポーネントに関係する単語だけ抽出（ログ用）
                comp_words = [
                    w for w in words
                    if any(is_variable(ch) and ch in comp_syms for ch in w["letters"])
                ]
                logger.info(
                    "[component %d] related words=%d",
                    idx, len(comp_words)
                )

                # このコンポーネントのドメインだけを切り出し
                comp_domains = {s: doms[s] for s in comp_syms}

                # ビームサーチ
                comp_assign, comp_dom = beam_search(
                    occs,          # 現状未使用だが渡しておく
                    comp_domains,
                    words,         # 盤面全体の単語リストを渡す
                    df_jukugo,
                    idx_len,
                    idx_pos,
                    word_solvable,
                    symbol_to_words,
                    symbol_solvable_counts,
                    threshold=threshold,
                    beam_width=beam_width,
                    use_tqdm=use_tqdm,
                )

                logger.info(
                    "[component %d] assign=%s", idx, comp_assign
                )
                assign_global.update(comp_assign)

                # final_domains にもこのコンポーネントの最終ドメインを反映
                for s, dom_s in comp_dom.items():
                    final_domains[s] = dom_s

        logger.info("Final assignment (partial allowed): %s", assign_global)

        # 仮置き熟語リスト（候補付き）
        word_infos = build_word_infos(words, assign_global, final_domains)
        logger.info(
            "Built %d word_infos. Example (up to 5): %s",
            len(word_infos),
            [w["text"] for w in word_infos[:5]]
        )

        # シンボル→漢字マッピングの整形
        mapping = []
        # doms (初期ドメイン) のキー全てについて結果を作成
        for sym in sorted(doms.keys(), key=lambda x: int(x[1:])):
            # 確定値があればそれ、なければ空文字
            fixed_val = assign_global.get(sym, "")

            # 最終候補リスト
            candidates = final_domains.get(sym, [])

            mapping.append(
                {
                    "symbol": sym,
                    "kanji": fixed_val,
                    "candidates": candidates,
                    "num": int(sym[1:]),
                }
            )

        logger.info("Mapping list: %s", mapping)
        logger.info("===== solve() END (log file: %s) =====", log_file)

        return {
            "mapping": mapping,
            "words": word_infos,
        }

    finally:
        # この solve() 呼び出し専用のファイルハンドラを外す
        logger.removeHandler(file_handler)
        file_handler.close()

#!/usr/bin/env python3
"""
solver_core.py

ナンクロ自動解答プログラム単体検証スクリプト（分岐多数解マジョリティ採用版）

候補が一意に定まっていない変数について、全ての候補パターンで
ビームサーチを行い、最終的に各変数でもっとも頻出する漢字を採用します。
重複禁止＋同率タイは不採用のマジョリティ選出を実装。
詳細なドメインサイズもデバッグレベルでログに記録します。
"""

import argparse
import ast
import copy
import itertools
import logging
import re
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- ロギング設定（DEBUGレベルも記録） ---
logging.basicConfig(
    filename='solver.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --- ユーティリティ関数 ---
def is_kanji(ch: str) -> bool:
    return re.match(r'[一-龯]', ch) is not None

def is_variable(ch: str) -> bool:
    return isinstance(ch, str) and ch.startswith('#') and ch[1:].isdigit()

def convert_numbers_to_variables(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Converting numbers to variables")
    return df.applymap(lambda x: f"#{int(x)}" if str(x).isdigit() else x)

# --- 熟語データ処理 ---
def load_jukugo(path: str) -> pd.DataFrame:
    logger.debug(f"Loading jukugo from {path}")
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    df['letters'] = df['letters'].apply(ast.literal_eval)
    df['count'] = df['letters'].apply(len)
    maxlen = df['count'].max()
    for i in range(maxlen):
        df[f'letter_{i+1}'] = df['letters'].apply(lambda L: L[i] if i < len(L) else '')
    logger.debug(f"Loaded {len(df)} entries, max length {maxlen}")
    return df

def build_reverse_index(df_jukugo: pd.DataFrame):
    logger.debug("Building reverse index")
    index_len = {}
    index_len_pos_char = {}
    for idx, row in df_jukugo.iterrows():
        L = row['count']
        index_len.setdefault(L, set()).add(idx)
        for pos, ch in enumerate(row['letters']):
            index_len_pos_char.setdefault((L, pos, ch), set()).add(idx)
    logger.debug(f"Index lengths: {sorted(index_len.keys())}")
    return index_len, index_len_pos_char

# --- 1Dクラスタリング ---
def simple_1d_cluster_np(coords, thresh):
    arr = np.asarray(coords, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(arr)
    sorted_arr = arr[order]
    diffs = np.diff(sorted_arr)
    new_group = diffs > thresh
    labels_sorted = np.concatenate(([0], np.cumsum(new_group)))
    labels = np.empty_like(labels_sorted)
    labels[order] = labels_sorted
    logger.debug(f"Clustered coords {coords} into labels {labels.tolist()}")
    return labels

# --- グリッド解析 ---
def parse_grid(df: pd.DataFrame) -> list:
    logger.debug("Parsing grid into words")
    words = []
    rows, cols = df.shape
    # 横方向
    for i in range(rows):
        j = 0
        while j < cols:
            if df.iat[i, j] != '■':
                letters, pos = [], []
                while j < cols and df.iat[i, j] != '■':
                    letters.append(df.iat[i, j])
                    pos.append((i, j))
                    j += 1
                if len(letters) > 1:
                    words.append({'letters': letters, 'positions': pos})
                    logger.debug(f"Found horizontal word {letters} at row {i}")
            else:
                j += 1
    # 縦方向
    for j in range(cols):
        i = 0
        while i < rows:
            if df.iat[i, j] != '■':
                letters, pos = [], []
                while i < rows and df.iat[i, j] != '■':
                    letters.append(df.iat[i, j])
                    pos.append((i, j))
                    i += 1
                if len(letters) > 1:
                    words.append({'letters': letters, 'positions': pos})
                    logger.debug(f"Found vertical word {letters} at col {j}")
            else:
                i += 1
    logger.debug(f"Total parsed words: {len(words)}")
    return words

# --- ドメイン構築 ---
def build_occurrences_and_domains(words, df_jukugo, index_len, index_len_pos_char):
    logger.debug("Building occurrences and domains")
    occurrences = {}
    for w_idx, w in enumerate(words):
        for p_idx, ch in enumerate(w['letters']):
            if is_variable(ch):
                occurrences.setdefault(ch, []).append((w_idx, p_idx))

    domains = {}
    for sym, occs in occurrences.items():
        common = None
        singles = []
        for w_idx, p_idx in occs:
            w = words[w_idx]
            L = len(w['letters'])
            cand = index_len.get(L, set()).copy()
            for pos, ch2 in enumerate(w['letters']):
                if is_kanji(ch2):
                    cand &= index_len_pos_char.get((L, pos, ch2), set())

            chars = {df_jukugo.at[i, 'letters'][p_idx] for i in cand}
            logger.debug(f"Variable {sym} in word {w['letters']} at pos {p_idx} has candidates: {chars}")
            if not chars:
                logger.debug(f"No candidates for {sym} at position {p_idx}, skipping")
                continue
            if len(chars) == 1:
                singles.append(next(iter(chars)))
            if common is None:
                common = set(chars)
            else:
                common &= chars

        # 一意候補検証
        if singles:
            unique = set(singles)
            if len(unique) == 1:
                domains[sym] = [unique.pop()]
                logger.debug(f"All singles agree for {sym}, domain = {domains[sym]}")
                continue
            else:
                logger.debug(f"Conflicting singles for {sym}: {singles}, falling back to intersection")

        domains[sym] = sorted(common) if common else []
        logger.debug(f"Domain for {sym}: {domains[sym]}")
    '''
    # 重複候補除外
    all_cand = [k for dom in domains.values() for k in dom]
    counts = Counter(all_cand)
    for sym, dom in domains.items():
        filtered = [k for k in dom if counts[k] == 1]
        if len(filtered) != len(dom):
            logger.debug(f"Excluding overlapping candidates for {sym}: removed {set(dom) - set(filtered)}")
        domains[sym] = filtered
    '''
    logger.debug(f"Built domains for variables: {{ {', '.join(f'{s}:{len(d)}' for s,d in domains.items())} }}")
    return occurrences, domains

# --- 制約伝播 ---
def forward_check(assign, domains, words, df_jukugo):
    new_domains = copy.deepcopy(domains)
    for w in words:
        letters = w['letters']
        cand = df_jukugo[df_jukugo['count'] == len(letters)].copy()
        for idx, ch in enumerate(letters):
            if is_kanji(ch):
                cand = cand[cand[f'letter_{idx+1}'] == ch]
            elif ch in assign:
                cand = cand[cand[f'letter_{idx+1}'] == assign[ch]]
        if cand.empty:
            return False, domains
        for idx, ch in enumerate(letters):
            if is_variable(ch) and ch not in assign:
                possible = set(cand[f'letter_{idx+1}'])
                new = [x for x in new_domains.get(ch, []) if x in possible]
                if not new:
                    return False, domains
                new_domains[ch] = new
    return True, new_domains

# --- 評価関数 ---
def evaluate(domains):
    score = sum(len(v) for v in domains.values())
    logger.debug(f"Evaluating domains {{ {', '.join(f'{s}:{len(d)}' for s,d in domains.items())} }} => score {score}")
    return score

# --- ビームサーチ ---
def beam_search(occs, domains, words, df_jukugo, beam_width):
    vars_ = list(domains.keys())
    logger.debug(f"Starting beam_search for vars {vars_} width={beam_width}")
    doms = {k: v for k, v in domains.items() if v}
    beam = [({}, doms.copy(), evaluate(doms))]
    for depth in range(len(vars_)):
        new_beam = []
        for assign, doms_cur, score in beam:
            if len(assign) == len(vars_):
                logger.debug("All vars assigned; breaking beam_search")
                return assign
            unassigned = [v for v in vars_ if v not in assign]
            var = min(unassigned, key=lambda v: len(doms_cur[v]))
            for c in doms_cur[var]:
                a2, d2 = assign.copy(), doms_cur.copy()
                a2[var] = c; d2[var] = [c]
                ok, d3 = forward_check(a2, d2, words, df_jukugo)
                if not ok:
                    continue
                new_beam.append((a2, d3, evaluate(d3)))
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_width]
        logger.debug(f"Depth {depth}: beam size={len(beam)}")
    result = beam[0][0] if beam else {}
    logger.debug("Beam search complete")
    return result

# --- メイン処理: 分岐とマジョリティ選出 ---
def solve_with_branching(df, jukugo_path="jukugo.csv", beam_width=10):
    logger.info("Starting solve_with_branching")
    df_jukugo = load_jukugo(jukugo_path)
    index_len, index_len_pos_char = build_reverse_index(df_jukugo)
    df2 = convert_numbers_to_variables(df)
    words = parse_grid(df2)
    occs, doms = build_occurrences_and_domains(words, df_jukugo, index_len, index_len_pos_char)

    logger.info(f"Domain sizes: {{ {', '.join(f'{s}:{len(d)}' for s,d in doms.items())} }}")

    fixed = {s: d[0] for s, d in doms.items() if len(d) == 1}
    branch_domains = {s: d for s, d in doms.items() if len(d) > 1}
    logger.info(f"Branch domains: {list(branch_domains.keys())}")

    assignments = []
    if not branch_domains:
        logger.info("No branch domains; single beam search")
        beam = beam_search(occs, {}, words, df_jukugo, beam_width)
        assignments.append({**fixed, **beam})
    else:
        keys = list(branch_domains.keys())
        for picks in itertools.product(*(branch_domains[k] for k in keys)):
            pick_map = dict(zip(keys, picks))
            logger.debug(f"Branch picks: {pick_map}")
            branch_fixed = fixed.copy()
            branch_fixed.update(pick_map)
            beam = beam_search(occs, {}, words, df_jukugo, beam_width)
            assignments.append({**branch_fixed, **beam})

    # マジョリティ選出（重複禁止＋同率タイは不採用）
    final = {}
    used = set()
    for sym in sorted(doms.keys(), key=lambda x: int(x[1:])):
        # tally を作成
        tally = {}
        for a in assignments:
            val = a.get(sym, "")
            if not val:
                continue
            tally[val] = tally.get(val, 0) + 1

        candidates = sorted(tally.items(), key=lambda x: -x[1])
        if not candidates:
            final[sym] = ""
            continue

        freqs = [f for _, f in candidates]
        max_freq = freqs[0]
        # 同率タイなら不採用
        if len(candidates) > 1 and all(f == max_freq for f in freqs):
            final[sym] = ""
            continue

        # 最頻かつ未使用の文字を選択
        chosen = ""
        for kanji, freq in candidates:
            if freq == max_freq and kanji not in used:
                chosen = kanji
                used.add(kanji)
                break

        final[sym] = chosen
        logger.debug(f"Final choice for {sym}: chosen={chosen}, used so far={used}")

    result = []
    for sym in sorted(doms.keys(), key=lambda x: int(x[1:])):
        result.append({"symbol": sym, "kanji": final[sym], "num": int(sym[1:])})
    logger.info("solve_with_branching complete")
    return result

# --- CLI 部分 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_csv", help="盤面CSVファイルパス")
    parser.add_argument("--jukugo", default="jukugo.csv", help="熟語CSVパス")
    parser.add_argument("--beam_width", type=int, default=10, help="ビーム幅")
    parser.add_argument("--output", default="solution.csv", help="出力先CSV")
    args = parser.parse_args()

    df_grid = pd.read_csv(args.grid_csv, header=None, dtype=str).fillna("■")
    result = solve_with_branching(df_grid, jukugo_path=args.jukugo, beam_width=args.beam_width)
    pd.DataFrame(result).to_csv(args.output, index=False, encoding='utf-8-sig')
    logger.info(f"Solution written to {args.output}")
    print(f"Solution written to {args.output}")

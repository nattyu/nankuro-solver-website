import ast
import copy
import itertools
import logging
import re
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# --- ユーティリティ関数 ---
def is_kanji(ch: str) -> bool:
    return re.match(r'[一-龯]', ch) is not None

def is_variable(ch: str) -> bool:
    return isinstance(ch, str) and ch.startswith('#') and ch[1:].isdigit()

def convert_numbers_to_variables(df: pd.DataFrame) -> pd.DataFrame:
    return df.applymap(lambda x: f"#{int(x)}" if str(x).isdigit() else x)

# --- 熟語辞書とインデックス ---
def load_jukugo(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    df['letters'] = df['letters'].apply(ast.literal_eval)
    df['count']   = df['letters'].apply(len)
    maxlen = df['count'].max()
    for i in range(maxlen):
        df[f'letter_{i+1}'] = df['letters'].apply(lambda L: L[i] if i < len(L) else '')
    return df

def build_reverse_index(df_jukugo: pd.DataFrame):
    index_len          = {}
    index_len_pos_char = {}
    for idx, row in df_jukugo.iterrows():
        L = row['count']
        index_len.setdefault(L, set()).add(idx)
        for pos, ch in enumerate(row['letters']):
            index_len_pos_char.setdefault((L, pos, ch), set()).add(idx)
    return index_len, index_len_pos_char

# --- グリッド解析 ---
def parse_grid(df: pd.DataFrame) -> list:
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
            else:
                i += 1
    return words

# --- ドメイン構築 ---
def build_occurrences_and_domains(words, df_jukugo, index_len, index_len_pos_char):
    occurrences = {}
    for w_idx, w in enumerate(words):
        for p_idx, ch in enumerate(w['letters']):
            if is_variable(ch):
                occurrences.setdefault(ch, []).append((w_idx, p_idx))

    domains = {}
    for sym, occs in occurrences.items():
        common  = None
        singles = []
        for w_idx, p_idx in occs:
            w = words[w_idx]
            L = len(w['letters'])
            cand = index_len.get(L, set()).copy()
            for pos, ch2 in enumerate(w['letters']):
                if is_kanji(ch2):
                    cand &= index_len_pos_char.get((L, pos, ch2), set())
            chars = {df_jukugo.at[i, 'letters'][p_idx] for i in cand}
            if not chars:
                continue
            if len(chars) == 1:
                singles.append(next(iter(chars)))
            common = chars if common is None else (common & chars)

        # 全ての singles が同じなら即決
        if singles:
            uniq = set(singles)
            if len(uniq) == 1:
                domains[sym] = [uniq.pop()]
                continue
        domains[sym] = sorted(common) if common else []

    # 重複候補を除外
    all_cand = [k for dom in domains.values() for k in dom]
    counts   = Counter(all_cand)
    for sym, dom in domains.items():
        domains[sym] = [k for k in dom if counts[k] == 1]

    return occurrences, domains

# --- 制約伝播 ---
def forward_check(assign, domains, words, df_jukugo):
    new_dom = copy.deepcopy(domains)
    for w in words:
        letters = w['letters']
        cand    = df_jukugo[df_jukugo['count'] == len(letters)].copy()
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
                filt     = [x for x in new_dom[ch] if x in possible]
                if not filt:
                    return False, domains
                new_dom[ch] = filt
    return True, new_dom

# --- 評価関数 ---
def evaluate(domains):
    return sum(len(v) for v in domains.values())

# --- ビームサーチ ---
def beam_search(occurrences, domains, words, df_jukugo, beam_width=10):
    doms = {k: v for k, v in domains.items() if v}
    vars_ = list(doms.keys())
    beam  = [({}, doms.copy(), evaluate(doms))]
    for _ in tqdm(range(len(vars_)), desc="Depth"):
        new_beam = []
        for assign, cur_dom, score in beam:
            if len(assign) == len(vars_):
                return assign
            unassigned = [v for v in vars_ if v not in assign]
            var = min(unassigned, key=lambda v: len(cur_dom[v]))
            for c in cur_dom[var]:
                a2 = assign.copy(); a2[var] = c
                d2 = cur_dom.copy(); d2[var]   = [c]
                ok, d3 = forward_check(a2, d2, words, df_jukugo)
                if ok:
                    new_beam.append((a2, d3, evaluate(d3)))
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_width]
        if not beam:
            break
    return beam[0][0] if beam else {}

# --- 解答器本体 ---
def solve(df: pd.DataFrame, jukugo_path: str = "jukugo.csv", beam_width: int = 10) -> list:
    df_jukugo = load_jukugo(jukugo_path)
    idx_len, idx_pos = build_reverse_index(df_jukugo)
    df2      = convert_numbers_to_variables(df)
    words    = parse_grid(df2)
    occs, doms = build_occurrences_and_domains(words, df_jukugo, idx_len, idx_pos)

    # 空ドメインチェック（中止せずに警告のみ）
    empty_syms = [s for s, d in doms.items() if not d]
    if empty_syms:
        print(f"Warning: no candidates for symbols {empty_syms}. Continuing with blanks.")

    # 固定／分岐対象を設定
    fixed  = {s: d[0] for s, d in doms.items() if len(d) == 1}
    branch = {s: d   for s, d in doms.items() if len(d) > 1}

    # 全ての分岐パターンまたは単一ビームで割り当て取得
    assignments = []
    if not branch:
        assign = beam_search(occs, {}, words, df_jukugo, beam_width)
        assignments.append({**fixed, **assign})
    else:
        keys = list(branch.keys())
        for picks in itertools.product(*(branch[k] for k in keys)):
            base   = fixed.copy()
            base.update(dict(zip(keys, picks)))
            assign = beam_search(occs, {}, words, df_jukugo, beam_width)
            assignments.append({**base, **assign})

    # マジョリティ選出
    final = {}
    used  = set()
    for sym in sorted(doms.keys(), key=lambda x: int(x[1:])):
        tally = {}
        for a in assignments:
            v = a.get(sym, "")
            if v:
                tally[v] = tally.get(v, 0) + 1
        if not tally:
            final[sym] = ""
            continue
        candidates = sorted(tally.items(), key=lambda x: -x[1])
        freqs = [f for _, f in candidates]
        m     = freqs[0]
        # 同率トップのみなら採用、同値多数なら空文字
        if len(candidates) > 1 and all(f == m for f in freqs):
            final[sym] = ""
        else:
            # 他と重複しないものから順に選択
            chosen = ""
            for k, f in candidates:
                if f == m and k not in used:
                    chosen = k
                    used.add(k)
                    break
            final[sym] = chosen

    # 結果リスト化
    result = []
    for sym in sorted(doms.keys(), key=lambda x: int(x[1:])):
        result.append({
            "symbol": sym,
            "kanji":  final[sym],
            "num":    int(sym[1:])
        })
    return result

# solver/nlp/bert_scorer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForMaskedLM  # type: ignore

from solver.grid_model import is_kanji, is_variable
from solver.logging_utils import get_logger
from solver.config import (
    BERT_MODEL_NAME,
    BERT_DEVICE,
    BERT_MAX_PATTERNS_PER_SYMBOL,
)

logger = get_logger()


# ===== BERT 本体 ============================================================


@dataclass
class BertCandidateScorer:
    """
    BERT（日本語 BERT）を使って、[MASK] を含む熟語パターンに
    候補漢字を当てはめたときのスコアを計算するクラス。

    - pattern 例: "登[MASK]場", "[MASK]口店"
    - candidates 例: ["山", "場", "校"]
    """

    model_name: str = BERT_MODEL_NAME
    device: str = BERT_DEVICE

    def __post_init__(self) -> None:
        logger.info("[BERT] Loading model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

        if self.mask_token is None or self.mask_token_id is None:
            logger.warning(
                "[BERT] tokenizer.mask_token / mask_token_id が取得できません。"
                "パターンスコアリングは 0 扱いになります。"
            )

    @torch.no_grad()
    def score_pattern(
        self,
        pattern: str,
        candidates: List[str],
    ) -> Dict[str, float]:
        """
        1つのパターンに対して、各候補漢字のスコアを返す。

        戻り値: { 漢字: スコア }  （スコアが高いほど自然）
        """
        if self.mask_token is None or self.mask_token_id is None:
            return {c: 0.0 for c in candidates}

        # ユーザー側の "[MASK]" を BERT のマスクトークンに置き換え
        text = pattern.replace("[MASK]", self.mask_token)

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # マスク位置（複数ある場合、とりあえず1個目だけ使用）
        mask_positions = (inputs["input_ids"][0] == self.mask_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(mask_positions) == 0:
            return {c: 0.0 for c in candidates}

        mask_idx = mask_positions[0].item()

        scores: Dict[str, float] = {}
        for c in candidates:
            # 1文字の漢字が 1 トークンになる前提
            tokens = self.tokenizer.tokenize(c)
            if len(tokens) != 1:
                # うまくトークン単位にならない場合はいったん 0
                scores[c] = 0.0
                continue
            tid = self.tokenizer.convert_tokens_to_ids(tokens[0])
            scores[c] = float(logits[0, mask_idx, tid].item())

        return scores

    @torch.no_grad()
    def score_patterns(
        self,
        patterns: List[str],
        candidates: List[str],
    ) -> Dict[str, float]:
        """
        複数パターン（同じ記号に関わる複数の熟語）を総合的に採点し、
        候補ごとの合計スコアを返す。
        """
        total: Dict[str, float] = {c: 0.0 for c in candidates}
        if not candidates or not patterns:
            return total

        # 必要ならここをバッチ化することも可能だが、とりあえず1つずつ
        for p in patterns:
            s = self.score_pattern(p, candidates)
            for c in candidates:
                total[c] += s.get(c, 0.0)

        return total

    @torch.no_grad()
    def choose_best(
        self,
        patterns: List[str],
        candidates: List[str],
    ) -> Tuple[str | None, Dict[str, float]]:
        """
        パターン群＋候補群から、最もスコアの高い 1 文字を返す。
        """
        if not candidates:
            return None, {}

        # 安全弁：パターン数を制限
        if len(patterns) > BERT_MAX_PATTERNS_PER_SYMBOL:
            patterns = patterns[:BERT_MAX_PATTERNS_PER_SYMBOL]

        scores = self.score_patterns(patterns, candidates)
        if not scores:
            return None, {}

        best = max(scores.items(), key=lambda x: x[1])[0]
        return best, scores


# ===== パターン生成 & solver との橋渡し ================================


def build_patterns_for_symbol(
    sym: str,
    words: List[Dict[str, Any]],
    occurrences: Dict[str, List[Tuple[int, int]]],
    assign: Dict[str, str],
) -> List[str]:
    """
    記号 `sym`（例: "#12"）に関わるすべての熟語をパターン文字列化する。

    ルール:
      - sym 自身 → "[MASK]"
      - 他のシンボル:
          - assign で確定している → その漢字
          - 未確定 → "□"
      - 漢字 → そのまま
    """
    if sym not in occurrences:
        return []

    pats: List[str] = []

    for (w_idx, _pos) in occurrences[sym]:
        w = words[w_idx]
        elems: List[str] = []
        for ch in w["letters"]:
            if is_kanji(ch):
                elems.append(ch)
            elif is_variable(ch):
                if ch == sym:
                    elems.append("[MASK]")
                else:
                    v = assign.get(ch, "")
                    elems.append(v if v else "□")
            else:
                elems.append(str(ch))
        pats.append("".join(elems))

    return pats


# scorer を毎回ロードしないように簡易シングルトンにする
_global_scorer: BertCandidateScorer | None = None


def get_global_scorer() -> BertCandidateScorer:
    global _global_scorer
    if _global_scorer is None:
        _global_scorer = BertCandidateScorer()
    return _global_scorer


def refine_assign_with_bert(
    assign: Dict[str, str],
    domains: Dict[str, List[str]],
    words: List[Dict[str, Any]],
    occurrences: Dict[str, List[Tuple[int, int]]],
):
    """
    既存の論理的な割当（assign, domains）に対して、
    BERT で「自然そうな候補」を選び、1文字に確定させていく。

    - 既に assign で確定しているシンボルは変更しない
    - domains の候補が 1 つ以下しかない場合も変更しない
    - それ以外について、BERT のスコアが最大の候補で上書きする
    """
    scorer = get_global_scorer()

    new_assign = dict(assign)
    new_domains = {k: v[:] for k, v in domains.items()}

    for sym, cand_list in domains.items():
        # すでに確定している or 候補 1つ以下 → 触らない
        if new_assign.get(sym) or len(cand_list) <= 1:
            continue

        patterns = build_patterns_for_symbol(sym, words, occurrences, new_assign)
        if not patterns:
            continue

        best, scores = scorer.choose_best(patterns, cand_list)

        if not best:
            continue

        # BERT の結果で上書き
        new_assign[sym] = best
        new_domains[sym] = [best]

        logger.info(
            "[BERT] symbol=%s patterns=%s candidates=%s -> best=%s",
            sym,
            patterns,
            cand_list,
            best,
        )
        logger.debug("[BERT] scores=%s", scores)

    return new_assign, new_domains

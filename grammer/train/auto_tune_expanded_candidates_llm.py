#!/usr/bin/env python3
"""基于真实数据 + LLM 评审的扩展候选阈值自动调参。

流程：
1. 读取真实数据（描述/结论）
2. 对每组离线阈值构建 expanded candidate lexicon
3. 仅评估“扩展候选命中”（不含显式混淆对命中）
4. 调用 LLM 对命中样本做真假判断，估计 precision
5. 输出 markdown 对比表并给出最佳参数
"""

import argparse
import json
import random
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_service import get_llm_validator
from train.phase2_mine_blacklist import (
    build_expanded_candidate_lexicon,
    load_medical_vocab,
    load_radiology_vocab,
    load_same_pinyin_dict,
)
from inference.detect_real_data_final import MedicalTypoDetectorFinal, find_error_context
from utils.config import MEDICAL_DICT_PATHS, RADIOLOGY_VOCAB


@dataclass
class GroupConfig:
    name: str
    max_variant_freq: int
    min_radio_freq: int
    max_candidates_per_wrong: int
    two_char_threshold: int


def parse_grid(raw: str) -> List[GroupConfig]:
    groups: List[GroupConfig] = []
    for i, seg in enumerate(raw.split(";"), 1):
        seg = seg.strip()
        if not seg:
            continue
        parts = [p.strip() for p in seg.split(",")]
        if len(parts) != 4:
            raise ValueError(f"无效参数组: {seg}")
        mv, mr, mc, tct = map(int, parts)
        groups.append(GroupConfig(f"G{i}", mv, mr, mc, tct))
    return groups


def is_chinese_token(token: str) -> bool:
    return bool(token) and all('\u4e00' <= c <= '\u9fff' for c in token)


def build_text_pool(df: pd.DataFrame) -> List[str]:
    pool: List[str] = []
    for col in ("描述", "结论"):
        if col not in df.columns:
            continue
        series = df[col].fillna("").astype(str)
        pool.extend([x for x in series.tolist() if x.strip()])
    return pool


def detect_expanded_only(
    detector: MedicalTypoDetectorFinal,
    text: str,
    candidate_lexicon: Dict[str, List[Dict]],
) -> List[Dict]:
    """仅检测扩展候选触发项（排除显式混淆对命中）。"""
    hits: List[Dict] = []

    direct_matches = detector.corrector.scan(text)
    matched_spans = {(m.position[0], m.position[1]) for m in direct_matches}

    detector.candidate_lexicon = candidate_lexicon

    for token, start, end in detector._tokenize_with_positions(text):
        if (start, end) in matched_spans:
            continue
        if len(token) < 2 or len(token) > 8:
            continue
        if not is_chinese_token(token):
            continue

        best = detector._select_best_expanded_candidate(text, token, start, end)
        if not best:
            continue

        hits.append(
            {
                "error": token,
                "suggestion": best,
                "context": find_error_context(text, start, end),
                "start": start,
                "end": end,
            }
        )
    return hits


def llm_judge_typo(validator, context: str, error: str, suggestion: str) -> Tuple[bool, float, str]:
    prompt = f"""你是医学影像报告文本质控专家。请判断下面这个纠错建议在该上下文中是否正确。

【上下文】
{context}

【错误词】{error}
【建议词】{suggestion}

请输出 JSON：
{{
  "is_valid": true/false,
  "confidence": 0.0-1.0,
  "reason": "简要理由"
}}
"""
    try:
        raw = validator._call_llm(prompt)
        parsed = validator._parse_json_response(raw)
        is_valid = bool(parsed.get("is_valid", False))
        conf = float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", ""))
        return is_valid, conf, reason
    except Exception as e:
        return False, 0.0, f"LLM调用失败: {e}"


def choose_best(results: List[Dict]) -> Dict:
    # 先按高置信 precision，再按触发率低，再按样本量高
    return sorted(
        results,
        key=lambda x: (
            x["precision_high_conf"],
            -x["trigger_rate"],
            x["llm_judged"],
        ),
        reverse=True,
    )[0]


def choose_topk_for_llm(results: List[Dict], top_k: int) -> List[Dict]:
    # 先保留兵变可覆盖；再优先低触发率、低平均命中、高词典质量
    ranked = sorted(
        results,
        key=lambda x: (
            1 if x["has_bingbian"] else 0,
            1 if not x["has_jixing"] else 0,
            -x["trigger_rate"],
            -x["avg_hits_per_text"],
            x["lexicon_words"],
        ),
        reverse=True,
    )
    return ranked[:top_k]


def render_table(rows: List[Dict]) -> str:
    headers = [
        "group",
        "max_variant",
        "min_radio",
        "max_cands",
        "two_char_th",
        "lexicon_words",
        "trigger_rate",
        "avg_hits_per_text",
        "llm_judged",
        "precision_high_conf",
        "precision_all",
        "mean_conf",
        "has_兵变",
        "has_积性",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append(
            "| " + " | ".join(
                [
                    r["group"],
                    str(r["max_variant"]),
                    str(r["min_radio"]),
                    str(r["max_cands"]),
                    str(r["two_char_th"]),
                    str(r["lexicon_words"]),
                    f"{r['trigger_rate']:.4f}",
                    f"{r['avg_hits_per_text']:.4f}",
                    str(r["llm_judged"]),
                    f"{r['precision_high_conf']:.4f}",
                    f"{r['precision_all']:.4f}",
                    f"{r['mean_conf']:.4f}",
                    "Y" if r["has_bingbian"] else "N",
                    "Y" if r["has_jixing"] else "N",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="LLM 驱动的离线候选阈值自动调参")
    parser.add_argument("--dataset", required=True, help="真实数据 Excel 路径")
    parser.add_argument(
        "--grid",
        default="50,100,8,1000;20,100,8,1000;20,200,8,1000;20,200,5,1000;10,200,5,1000;10,300,3,1000",
        help="参数组: max_variant,min_radio,max_cands,two_char_threshold;...",
    )
    parser.add_argument("--max-rows", type=int, default=8000, help="参与扫描的最大行数")
    parser.add_argument("--eval-text-limit", type=int, default=10000, help="每组规则评估文本数上限")
    parser.add_argument("--llm-top-k", type=int, default=3, help="进入LLM精评的参数组数量")
    parser.add_argument("--llm-cases-per-group", type=int, default=40, help="每组送 LLM 评审的样本数")
    parser.add_argument("--llm-workers", type=int, default=16, help="LLM并发评审线程数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="output/expanded_candidates_llm_tuning.md")
    args = parser.parse_args()

    random.seed(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    validator = get_llm_validator()
    if not validator.available():
        raise RuntimeError("LLM 服务不可用，请检查 .env 中的 LLM 配置和服务状态")

    print("加载基础资源...")
    radio_vocab = load_radiology_vocab(str(RADIOLOGY_VOCAB))
    same_pinyin = load_same_pinyin_dict()
    medical_vocab = load_medical_vocab(MEDICAL_DICT_PATHS)

    print("读取数据集...")
    df = pd.read_excel(args.dataset, usecols=["描述", "结论"], nrows=args.max_rows)
    text_pool = build_text_pool(df)
    if len(text_pool) > args.eval_text_limit:
        text_pool = random.sample(text_pool, args.eval_text_limit)
    print(f"规则评估文本条数: {len(text_pool)}")

    detector = MedicalTypoDetectorFinal()
    detector.load()

    groups = parse_grid(args.grid)
    results: List[Dict] = []

    coarse_results: List[Dict] = []

    # Stage-1: 规则粗筛（不调用LLM）
    for g in groups:
        print(f"\n评估 {g.name}: {g}")
        lexicon = build_expanded_candidate_lexicon(
            radio_vocab=radio_vocab,
            same_pinyin=same_pinyin,
            medical_vocab=medical_vocab,
            min_radio_freq=g.min_radio_freq,
            two_char_threshold=g.two_char_threshold,
            max_variant_freq=g.max_variant_freq,
            max_candidates_per_wrong=g.max_candidates_per_wrong,
        )

        all_hits: List[Dict] = []
        hit_text_count = 0
        total_hits = 0

        for text in text_pool:
            hits = detect_expanded_only(detector, text, lexicon)
            if hits:
                hit_text_count += 1
                total_hits += len(hits)
                # 每条文本只取最前面两个，避免某条文本主导采样
                all_hits.extend(hits[:2])

        trigger_rate = hit_text_count / max(len(text_pool), 1)
        avg_hits_per_text = total_hits / max(len(text_pool), 1)

        coarse_results.append(
            {
                "group": g.name,
                "max_variant": g.max_variant_freq,
                "min_radio": g.min_radio_freq,
                "max_cands": g.max_candidates_per_wrong,
                "two_char_th": g.two_char_threshold,
                "lexicon_words": len(lexicon),
                "trigger_rate": trigger_rate,
                "avg_hits_per_text": avg_hits_per_text,
                "sample_hits": all_hits,
                "llm_judged": 0,
                "precision_high_conf": 0.0,
                "precision_all": 0.0,
                "mean_conf": 0.0,
                "has_bingbian": "兵变" in lexicon,
                "has_jixing": "积性" in lexicon,
            }
        )

    # Stage-2: 仅对 Top-K 组做 LLM 精评
    topk = choose_topk_for_llm(coarse_results, args.llm_top_k)
    topk_names = {x["group"] for x in topk}
    print(f"\n进入 LLM 精评参数组: {', '.join(sorted(topk_names))}")

    for row in coarse_results:
        if row["group"] not in topk_names:
            results.append(row)
            continue

        sample_hits = row.get("sample_hits", [])
        if len(sample_hits) > args.llm_cases_per_group:
            sample_hits = random.sample(sample_hits, args.llm_cases_per_group)

        judgments = []
        with ThreadPoolExecutor(max_workers=max(args.llm_workers, 1)) as ex:
            futures = [
                ex.submit(
                    llm_judge_typo,
                    validator,
                    h["context"],
                    h["error"],
                    h["suggestion"],
                )
                for h in sample_hits
            ]
            for fut in as_completed(futures):
                try:
                    judgments.append(fut.result())
                except Exception:
                    judgments.append((False, 0.0, "future_error"))

        high_conf_tp = sum(1 for ok, conf, _ in judgments if ok and conf >= 0.7)
        all_tp = sum(1 for ok, _, _ in judgments if ok)
        precision_high_conf = high_conf_tp / max(len(judgments), 1)
        precision_all = all_tp / max(len(judgments), 1)
        mean_conf = statistics.mean([conf for _, conf, _ in judgments]) if judgments else 0.0

        row["llm_judged"] = len(judgments)
        row["precision_high_conf"] = precision_high_conf
        row["precision_all"] = precision_all
        row["mean_conf"] = mean_conf
        results.append(row)

    # 移除临时字段
    for row in results:
        row.pop("sample_hits", None)

    best = choose_best(results)
    table = render_table(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table + "\n\n" + f"best={json.dumps(best, ensure_ascii=False)}\n", encoding="utf-8")

    print("\n参数对比表：")
    print(table)
    print("\n最佳参数：")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""阈值扫描：离线扩展候选词库参数对比表

输出每组参数的候选规模与质量代理指标，帮助快速定参。
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.phase2_mine_blacklist import (
    load_radiology_vocab,
    load_same_pinyin_dict,
    load_medical_vocab,
    build_expanded_candidate_lexicon,
)
from utils.config import RADIOLOGY_VOCAB, MEDICAL_DICT_PATHS


def summarize_lexicon(lexicon):
    wrong_words = len(lexicon)
    total_candidates = sum(len(v) for v in lexicon.values())
    avg_candidates = (total_candidates / wrong_words) if wrong_words else 0.0

    all_items = [item for items in lexicon.values() for item in items]
    offline_scores = [item.get("offline_score", 0.0) for item in all_items]
    wrong_freq_pos = [item for item in all_items if item.get("wrong_freq", 0) > 0]
    correct_freqs = [item.get("correct_freq", 0) for item in all_items]

    return {
        "wrong_words": wrong_words,
        "total_candidates": total_candidates,
        "avg_candidates": round(avg_candidates, 3),
        "p50_offline_score": round(statistics.median(offline_scores), 4) if offline_scores else 0.0,
        "p90_offline_score": round(statistics.quantiles(offline_scores, n=10)[8], 4) if len(offline_scores) >= 10 else 0.0,
        "wrong_freq_gt0_ratio": round((len(wrong_freq_pos) / max(total_candidates, 1)) * 100, 2),
        "median_correct_freq": int(statistics.median(correct_freqs)) if correct_freqs else 0,
        "has_bingbian": "兵变" in lexicon,
        "has_jixing": "积性" in lexicon,
    }


def parse_grid(raw: str):
    groups = []
    for seg in raw.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        parts = [p.strip() for p in seg.split(",")]
        if len(parts) != 4:
            raise ValueError(f"无效参数组: {seg}")
        max_variant_freq, min_radio_freq, max_candidates_per_wrong, two_char_threshold = map(int, parts)
        groups.append({
            "max_variant_freq": max_variant_freq,
            "min_radio_freq": min_radio_freq,
            "max_candidates_per_wrong": max_candidates_per_wrong,
            "two_char_threshold": two_char_threshold,
        })
    return groups


def format_markdown_table(rows):
    headers = [
        "group",
        "max_variant",
        "min_radio",
        "max_cands",
        "two_char_th",
        "wrong_words",
        "total_cands",
        "avg_cands",
        "p50_score",
        "p90_score",
        "wrong_freq>0%",
        "median_correct_freq",
        "has_兵变",
        "has_积性",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for i, r in enumerate(rows, 1):
        lines.append(
            "| " + " | ".join([
                f"G{i}",
                str(r["params"]["max_variant_freq"]),
                str(r["params"]["min_radio_freq"]),
                str(r["params"]["max_candidates_per_wrong"]),
                str(r["params"]["two_char_threshold"]),
                str(r["summary"]["wrong_words"]),
                str(r["summary"]["total_candidates"]),
                str(r["summary"]["avg_candidates"]),
                str(r["summary"]["p50_offline_score"]),
                str(r["summary"]["p90_offline_score"]),
                str(r["summary"]["wrong_freq_gt0_ratio"]),
                str(r["summary"]["median_correct_freq"]),
                "Y" if r["summary"]["has_bingbian"] else "N",
                "Y" if r["summary"]["has_jixing"] else "N",
            ]) + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="离线扩展候选词库阈值扫描")
    parser.add_argument(
        "--grid",
        default="50,100,8,1000;20,100,8,1000;20,200,8,1000;20,200,5,1000;10,200,5,1000;10,300,3,1000",
        help="参数组: max_variant,min_radio,max_candidates,two_char_threshold;...",
    )
    parser.add_argument("--out", default="output/expanded_candidates_tuning_table.md", help="输出 Markdown 表路径")
    args = parser.parse_args()

    print("加载基础资源...")
    radio_vocab = load_radiology_vocab(str(RADIOLOGY_VOCAB))
    same_pinyin = load_same_pinyin_dict()
    medical_vocab = load_medical_vocab(MEDICAL_DICT_PATHS)

    groups = parse_grid(args.grid)
    results = []

    for idx, params in enumerate(groups, 1):
        print(f"\n[{idx}/{len(groups)}] 扫描参数: {params}")
        lexicon = build_expanded_candidate_lexicon(
            radio_vocab=radio_vocab,
            same_pinyin=same_pinyin,
            medical_vocab=medical_vocab,
            min_radio_freq=params["min_radio_freq"],
            two_char_threshold=params["two_char_threshold"],
            max_variant_freq=params["max_variant_freq"],
            max_candidates_per_wrong=params["max_candidates_per_wrong"],
        )
        summary = summarize_lexicon(lexicon)
        results.append({"params": params, "summary": summary})

    table = format_markdown_table(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table + "\n", encoding="utf-8")

    print("\n参数对比表：")
    print(table)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()

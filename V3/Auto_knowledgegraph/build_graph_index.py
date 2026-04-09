from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到路径
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


import json
from pathlib import Path
from typing import Any

import pandas as pd

from V3.Auto_knowledgegraph.config import ensure_runtime_dirs, load_config

LEVEL_COLUMNS = ["一级部位", "二级部位", "三级部位", "四级部位", "五级部位", "六级部位"]


def _safe_split(cell: Any) -> list[str]:
    if pd.isna(cell):
        return []
    return [part.strip() for part in str(cell).split("|") if part and str(part).strip()]


def build_graph_nodes(df: pd.DataFrame, source: str) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        path_parts: list[str] = []
        synonym_levels: list[list[str]] = []
        for column in LEVEL_COLUMNS:
            synonyms = _safe_split(row.get(column))
            if not synonyms:
                break
            path_parts.append(synonyms[0])
            synonym_levels.append(synonyms)
        if not path_parts:
            continue

        node = {
            "source": source,
            "node_id": f"{source}:{'/'.join(path_parts)}",
            "path": path_parts,
            "canonical_name": path_parts[-1],
            "level": len(path_parts),
            "synonym_levels": synonym_levels,
            "start_axis": row.get("起始坐标"),
            "end_axis": row.get("终止坐标"),
            "category": row.get("分类"),
        }
        nodes.append(node)
    return nodes


def build_alias_index(nodes: list[dict[str, Any]]) -> dict[str, list[str]]:
    alias_index: dict[str, list[str]] = {}
    for node in nodes:
        node_id = node["node_id"]
        for synonyms in node.get("synonym_levels", []):
            for synonym in synonyms:
                alias_index.setdefault(synonym, [])
                if node_id not in alias_index[synonym]:
                    alias_index[synonym].append(node_id)
    return alias_index


def main() -> None:
    cfg = load_config()
    ensure_runtime_dirs(cfg)

    report_df = pd.read_excel(cfg.knowledgegraph_path)
    title_book = pd.ExcelFile(cfg.title_knowledgegraph_path)

    report_nodes = build_graph_nodes(report_df, source="report")
    title_nodes: list[dict[str, Any]] = []
    for sheet_name in title_book.sheet_names:
        sheet_df = pd.read_excel(cfg.title_knowledgegraph_path, sheet_name=sheet_name)
        title_nodes.extend(build_graph_nodes(sheet_df, source=f"title:{sheet_name}"))

    all_nodes = report_nodes + title_nodes
    alias_index = build_alias_index(all_nodes)

    (cfg.data_dir / "graph_index.json").write_text(
        json.dumps({"nodes": all_nodes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (cfg.data_dir / "graph_alias_index.json").write_text(
        json.dumps(alias_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
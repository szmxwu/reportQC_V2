#!/usr/bin/env python3
"""Export first 100 rows from result/reports.db::reports into a Markdown table.

Usage:
    python3 tools/export_reports_preview_md.py

This script will create (or overwrite) result/reports_preview.md
"""
import sqlite3
import os
import sys
from typing import List, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result', 'reports.db')
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result', 'reports_preview.md')


def fetch_rows(db_path: str, limit: int = 100) -> Tuple[List[str], List[Tuple]]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # 检查表是否存在
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reports'")
        if not cur.fetchone():
            raise RuntimeError("Table 'reports' does not exist in the database")
        cur.execute("PRAGMA table_info(reports)")
        cols = [row[1] for row in cur.fetchall()]
        sel = cur.execute(f"SELECT * FROM reports LIMIT ?", (limit,))
        rows = sel.fetchall()
        return cols, rows
    finally:
        conn.close()


def to_markdown(cols: List[str], rows: List[Tuple]) -> str:
    # Header
    header = '| ' + ' | '.join(cols) + ' |\n'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |\n'
    body_lines = []
    for r in rows:
        # 转换 None -> ''，并把每个元素转为字符串
        cells = ['' if c is None else str(c) for c in r]
        body_lines.append('| ' + ' | '.join(cells) + ' |')
    return header + sep + '\n'.join(body_lines) + '\n'


def main():
    try:
        cols, rows = fetch_rows(DB_PATH, limit=100)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    md = to_markdown(cols, rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('# Reports preview (first 100 rows)\n\n')
        f.write(md)

    print(f"Wrote preview to: {OUT_PATH}")


if __name__ == '__main__':
    main()

"""
高风险句子对人工标注平台
用于标注 sentence_semantics/output/data/high_risk_review_list.jsonl
"""
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="高风险句子对标注平台")

# 数据文件路径
DATA_FILE = Path("sentence_semantics/output/data/high_risk_review_list.jsonl")

# 内存中缓存的数据
_records: list[dict] = []


def load_data() -> list[dict]:
    records = []
    if DATA_FILE.exists():
        with DATA_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def save_data(records: list[dict]) -> None:
    # 写回原始 jsonl 文件
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DATA_FILE.open("w", encoding="utf-8", newline="") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@app.on_event("startup")
def startup():
    global _records
    _records = load_data()


class AnnotatePayload(BaseModel):
    index: int
    confirm: bool | None  # True=认可, False=不认可, None=未标注


@app.get("/api/data")
def get_data():
    """返回所有记录（仅含必要字段，减少传输量）"""
    global _records
    return {
        "total": len(_records),
        "items": [
            {
                "index": i,
                "review_task": rec.get("review_task", ""),
                "sentence_a": rec.get("sentence_a", ""),
                "sentence_b": rec.get("sentence_b", ""),
                "confirm": rec.get("confirm"),  # 可能不存在
            }
            for i, rec in enumerate(_records)
        ],
    }


@app.post("/api/save")
def save_annotation(payload: AnnotatePayload):
    global _records
    if 0 <= payload.index < len(_records):
        if payload.confirm is None:
            _records[payload.index].pop("confirm", None)
        else:
            _records[payload.index]["confirm"] = payload.confirm
        save_data(_records)
        return {"status": "ok"}
    return {"status": "error", "message": "index out of range"}


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>高风险句子对标注平台</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    background: #f5f7fa;
    color: #333;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #2c3e50;
    color: #fff;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }
  header h1 { font-size: 18px; font-weight: 500; }
  .progress { font-size: 14px; color: #ecf0f1; }
  .progress span { font-weight: bold; color: #1abc9c; }
  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 20px;
    overflow: hidden;
  }
  .task-hint {
    background: #fff3cd;
    border-left: 4px solid #f1c40f;
    padding: 10px 16px;
    margin-bottom: 16px;
    border-radius: 4px;
    font-size: 14px;
    color: #856404;
    flex-shrink: 0;
  }
  .sentences {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    min-height: 0;
  }
  .card {
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    padding: 24px;
    display: flex;
    flex-direction: column;
    overflow: auto;
  }
  .card h3 {
    font-size: 14px;
    color: #7f8c8d;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .card .text {
    font-size: 22px;
    line-height: 1.7;
    color: #2c3e50;
    word-break: break-word;
  }
  .controls {
    margin-top: 16px;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 20px;
    flex-shrink: 0;
  }
  .btn {
    border: none;
    border-radius: 8px;
    padding: 16px 48px;
    font-size: 18px;
    cursor: pointer;
    transition: transform .05s, opacity .2s;
    user-select: none;
  }
  .btn:active { transform: scale(0.98); }
  .btn-true {
    background: #27ae60;
    color: #fff;
  }
  .btn-true:hover { background: #219150; }
  .btn-true.active {
    box-shadow: 0 0 0 4px rgba(39,174,96,0.3);
    outline: 2px solid #27ae60;
  }
  .btn-false {
    background: #c0392b;
    color: #fff;
  }
  .btn-false:hover { background: #a93226; }
  .btn-false.active {
    box-shadow: 0 0 0 4px rgba(192,57,43,0.3);
    outline: 2px solid #c0392b;
  }
  .nav {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .nav button {
    background: #34495e;
    color: #fff;
    border: none;
    padding: 10px 18px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
  }
  .nav button:disabled { background: #95a5a6; cursor: not-allowed; }
  .status {
    font-size: 13px;
    color: #7f8c8d;
    margin-top: 6px;
    text-align: center;
  }
  .status.ok { color: #27ae60; }
  @media (max-width: 900px) {
    .sentences { grid-template-columns: 1fr; }
    .card .text { font-size: 18px; }
  }
</style>
</head>
<body>
<header>
  <h1>📝 高风险句子对人工标注</h1>
  <div class="progress">进度: <span id="progress">0/0</span> &nbsp;|&nbsp; 已标注: <span id="done-count">0</span></div>
</header>
<main>
  <div class="task-hint" id="task-hint">任务提示: 判断下面两个句子是否等价（语义一致）</div>
  <div class="sentences">
    <div class="card">
      <h3>Sentence A</h3>
      <div class="text" id="sentence-a">加载中...</div>
    </div>
    <div class="card">
      <h3>Sentence B</h3>
      <div class="text" id="sentence-b">加载中...</div>
    </div>
  </div>
  <div class="controls">
    <div class="nav">
      <button id="btn-prev">← 上一条 (←)</button>
    </div>
    <button class="btn btn-false" id="btn-false">不认可 (N)</button>
    <button class="btn btn-true" id="btn-true">认可 (Y)</button>
    <div class="nav">
      <button id="btn-next">下一条 (→) →</button>
    </div>
  </div>
  <div class="status" id="status">按 Y 认可，N 不认可，← → 翻页</div>
</main>
<script>
  let items = [];
  let current = 0;

  async function load() {
    const res = await fetch('/api/data');
    const data = await res.json();
    items = data.items || [];
    current = 0;
    // 尝试跳到第一个未标注的位置
    const firstUnlabeled = items.findIndex(i => i.confirm === null || i.confirm === undefined);
    if (firstUnlabeled !== -1) current = firstUnlabeled;
    render();
  }

  function render() {
    if (!items.length) return;
    const item = items[current];
    document.getElementById('sentence-a').textContent = item.sentence_a || '';
    document.getElementById('sentence-b').textContent = item.sentence_b || '';
    document.getElementById('task-hint').textContent = '任务提示: ' + (item.review_task || '请判断以下两个句子的关系');
    document.getElementById('progress').textContent = (current + 1) + '/' + items.length;
    document.getElementById('done-count').textContent = items.filter(i => i.confirm === true || i.confirm === false).length;

    const bt = document.getElementById('btn-true');
    const bf = document.getElementById('btn-false');
    bt.classList.toggle('active', item.confirm === true);
    bf.classList.toggle('active', item.confirm === false);

    document.getElementById('btn-prev').disabled = current === 0;
    document.getElementById('btn-next').disabled = current === items.length - 1;
  }

  async function setConfirm(value) {
    if (!items.length) return;
    items[current].confirm = value;
    render();
    const status = document.getElementById('status');
    status.textContent = '保存中...';
    try {
      const res = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: current, confirm: value })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        status.textContent = '已保存';
        status.classList.add('ok');
        setTimeout(() => { status.classList.remove('ok'); status.textContent = '按 Y 认可，N 不认可，← → 翻页'; }, 800);
      } else {
        status.textContent = '保存失败: ' + (data.message || '');
      }
    } catch (e) {
      status.textContent = '保存失败: ' + e.message;
    }
  }

  function prev() { if (current > 0) { current--; render(); } }
  function next() { if (current < items.length - 1) { current++; render(); } }

  document.getElementById('btn-true').onclick = () => setConfirm(true);
  document.getElementById('btn-false').onclick = () => setConfirm(false);
  document.getElementById('btn-prev').onclick = prev;
  document.getElementById('btn-next').onclick = next;

  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    switch (e.key) {
      case 'ArrowLeft': prev(); break;
      case 'ArrowRight': next(); break;
      case 'y': case 'Y': case '1': setConfirm(true); break;
      case 'n': case 'N': case '0': setConfirm(false); break;
    }
  });

  load();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    print(f"数据文件: {DATA_FILE.resolve()}")
    print("启动标注平台: http://127.0.0.1:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765)

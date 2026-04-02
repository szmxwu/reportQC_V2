"""将抽取结果转换为Doccano(jsonl)需要的格式。

输入：text_extrac_process / report_extrac_process 的结果列表，每个元素为一个包含
 anatomy 与 pathology 信息的字典（参见 output.jsonl 示例）。

输出：List[Dict]，每个元素对应一条 Doccano 记录：
{
  "text": "短句",
  "entities": [
      {"id": 0, "start_offset": 0, "end_offset": 2, "label": "ANAT", "keyword": "肝门", "attributes": {"orientation": ""}},
      {"id": 1, "start_offset": 2, "end_offset": 7, "label": "FIND", "keyword": "区可见肿块"}
  ],
  "relations": [
      {"from_id": 0, "to_id": 1, "type": "ANAT-FIND"}
  ]
}

规则说明：
1. text 取每组的 short_sentence。
2. ANAT 实体：使用字段 start/end 作为偏移；label 固定为 "ANAT"；携带 keyword 与 attributes.orientation。
3. FIND 实体：使用字段 illness_start / illness_end 作为偏移；label 固定为 "FIND"；keyword = illness。
4. 关系：同一 anchor 里的 ANAT 与其对应的 FIND 建立一条 "ANAT-FIND" 关系。
5. 若同一 short_sentence 中有多个解剖实体，会聚合到同一条记录；同一 (illness_start, illness_end, illness) 只生成一次 FIND 实体；多个 ANAT 指向对应 FIND（可能一对多）。
6. 缺失或不合法的偏移将被跳过；若某组最终没有成对的实体则不输出。
"""
from typing import Iterable, List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict


def _valid_span(start: Any, end: Any, text: str) -> bool:
    try:
        return isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)
    except Exception:
        return False


def trans_to_Doccano(entities: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not entities:
        return []

    # 分组：同一 short_sentence（结合 sentence_index 避免不同句子相同内容混合）
    groups: DefaultDict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for item in entities:
        short_sentence = item.get("short_sentence") or item.get("long_sentence") or item.get("original_sentence") or ""
        sentence_index = item.get("sentence_index", -1)
        groups[(short_sentence, sentence_index)].append(item)

    doccano_records: List[Dict[str, Any]] = []

    for (short_sentence, _), items in groups.items():
        text = short_sentence or ""
        if not text:
            continue

        anat_map: Dict[Tuple[int, int, str], int] = {}
        find_map: Dict[Tuple[int, int, str], int] = {}
        entities_out: List[Dict[str, Any]] = []
        relations_out: List[Dict[str, Any]] = []

        # 第一轮：建立实体（保持插入顺序）
        def get_or_add_anat(start: int, end: int, keyword: str, orientation: str) -> int:
            key = (start, end, keyword)
            if key in anat_map:
                return anat_map[key]
            ent_id = len(entities_out)
            entities_out.append({
                "id": ent_id,
                "start_offset": start,
                "end_offset": end,
                "label": "ANAT",
                "keyword": keyword,
                "attributes": {"orientation": orientation or ""}
            })
            anat_map[key] = ent_id
            return ent_id

        def get_or_add_find(start: int, end: int, illness: str) -> int:
            key = (start, end, illness)
            if key in find_map:
                return find_map[key]
            ent_id = len(entities_out)
            entities_out.append({
                "id": ent_id,
                "start_offset": start,
                "end_offset": end,
                "label": "FIND",
                "keyword": illness
            })
            find_map[key] = ent_id
            return ent_id

        for anchor in items:
            # 解剖实体
            a_start = anchor.get("start")
            a_end = anchor.get("end")
            keyword = anchor.get("keyword", "")
            orientation = anchor.get("orientation", "")
            # 病理实体
            f_start = anchor.get("illness_start")
            f_end = anchor.get("illness_end")
            illness = anchor.get("illness", "")

            if not keyword or not illness:
                continue
            if not _valid_span(a_start, a_end, text):
                continue
            if not _valid_span(f_start, f_end, text):
                continue

            anat_id = get_or_add_anat(a_start, a_end, keyword, orientation)
            find_id = get_or_add_find(f_start, f_end, illness)

            # 关系：去重
            rel_key = (anat_id, find_id)
            if not any(r["from_id"] == anat_id and r["to_id"] == find_id for r in relations_out):
                relations_out.append({
                    "from_id": anat_id,
                    "to_id": find_id,
                    "type": "ANAT-FIND"
                })

        if not entities_out or not relations_out:
            continue

        # 重新规范 id（若存在跳号）
        id_mapping = {old_ent["id"]: new_id for new_id, old_ent in enumerate(entities_out)}
        for new_id, ent in enumerate(entities_out):
            ent["id"] = new_id
        for rel in relations_out:
            rel["from_id"] = id_mapping[rel["from_id"]]
            rel["to_id"] = id_mapping[rel["to_id"]]

        doccano_records.append({
            "text": text,
            "entities": entities_out,
            "relations": relations_out,
            "long_sentence": (items[0].get("long_sentence") if items else text)
        })

    return doccano_records

# grammer 模块模型文件说明

## 必需文件（已包含在仓库中）

| 文件 | 大小 | 说明 |
|------|------|------|
| `ac_automaton_v2.pkl` | 4.4MB | AC自动机纠错引擎（主引擎） |
| `medical_confusion.txt` | 4.4MB | 医学拼音混淆词表 |
| `high_risk_general.txt` | 429KB | 高危通用错误词表 |
| `word_order_templates.json` | 238KB | 词序错误检测模板 |
| `radiology_vocab.json` | 834KB | 放射语料词频表 |
| `substring_vocab.json` | 38KB | 子串频次表 |

## 可选文件（推荐下载）

| 文件 | 大小 | 说明 | 获取方式 |
|------|------|------|----------|
| `radiology_ngram.klm` | 84MB | KenLM语言模型 | 下载 `radiology_ngram.klm.gz` 后解压 |

## 训练用文件（不需要上传到仓库）

| 文件 | 大小 | 说明 |
|------|------|------|
| `radiology_corpus.txt` | 2.2GB | 训练语料（分词后） |
| `radiology_ngram.arpa` | 123MB | ARPA格式语言模型（转klm前的中间文件） |
| `ac_automaton.pkl` | 28MB | 旧版引擎（已废弃） |

## 依赖关系图

```
detect_real_data_final.py (推理入口)
├── ac_automaton_v2.pkl ✅ 必需
├── medical_confusion.txt ✅ 必需
├── high_risk_general.txt ✅ 必需
├── word_order_templates.json ✅ 必需
└── radiology_ngram.klm ⚠️ 可选（无则跳过上下文校验）

train/ (训练脚本，运行不需要)
├── radiology_corpus.txt ❌ 训练用
└── radiology_ngram.arpa ❌ 训练中间文件
```

## 注意事项

- 如果 `ac_automaton_v2.pkl` 不存在，代码会尝试从 `medical_confusion.txt` 和 `high_risk_general.txt` 重新生成
- `radiology_ngram.klm` 是可选依赖，缺失时只会影响上下文校验功能，基础错别字检测仍可正常工作

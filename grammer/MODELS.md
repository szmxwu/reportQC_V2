# 模型文件说明

## 推理必需的文件

运行错别字检测需要以下模型文件：

### 1. KenLM 语言模型 (必需)
- **文件**: `models/radiology_ngram.klm`
- **大小**: ~627MB (压缩后分卷)
- **解压命令**:
  ```bash
  cd grammer
  cat models/radiology_ngram.klm.part.* | tar -xzf -
  ```

### 2. AC 自动机 (必需)
- **文件**: `models/ac_automaton_v2.pkl`
- **大小**: ~22MB (压缩后 3.4MB)
- **解压命令**:
  ```bash
  cd grammer/models
  tar -xzf ac_automaton_v2.pkl.tar.gz
  ```

### 3. 医学混淆词表 (必需)
- **文件**: `models/medical_confusion.txt`
- **大小**: ~24MB (压缩后 3.3MB)
- **解压命令**:
  ```bash
  cd grammer/models
  tar -xzf medical_confusion.txt.tar.gz
  ```

### 4. 词序模板 (已包含在 Git 中)
- **文件**: `models/word_order_templates.json`
- **大小**: 243KB
- **说明**: 文件较小，直接提交到 Git

### 5. 词表文件 (已包含在 Git 中)
- **文件**: `models/radiology_vocab.json`
- **大小**: 1MB
- **说明**: 文件较小，直接提交到 Git

### 6. 高危通用词 (已包含在 Git 中)
- **文件**: `models/high_risk_general.txt`
- **大小**: 439KB
- **说明**: 文件较小，直接提交到 Git

## 快速解压所有模型

```bash
cd grammer

# 解压 KLM 模型
cat models/radiology_ngram.klm.part.* | tar -xzf -

# 解压 AC 自动机
cd models && tar -xzf ac_automaton_v2.pkl.tar.gz

# 解压医学混淆词表
tar -xzf medical_confusion.txt.tar.gz
```

## 训练资源文件 (已加入 .gitignore)

以下文件不需要提交到 Git，可以通过训练流程重新生成：

- `train/` - 训练代码和原始语料
- `models/radiology_corpus.txt` - 原始语料库 (~598MB)
- `models/sentence_corpus.sqlite3` - 句子语料库 (~618MB)
- `models/radiology_ngram.arpa` - ARPA 格式模型 (~1.1GB)
- `models/expanded_candidates.json` - 扩展候选词 (~59MB)
- `models/substring_vocab.json` - 子串词表

## 训练流程

如果需要重新训练模型，请执行：

```bash
python -m inference.medical_typo_detector --full-pipeline
```

详细说明请参考 `README.md`

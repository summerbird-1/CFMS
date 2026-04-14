# CFMS: An Explainable Fine-Grained Chinese Multimodal Sarcasm Detection Benchmark
A high-quality benchmark for **interpretable multimodal sarcasm understanding** on Chinese social media, with triple-level annotations and a reinforcement learning-based in-context learning method.

---

## 📢 Updates
- **Jan 2026**: Dataset, code and evaluation scripts are publicly released.

---

## 📌 Overview
Sarcasm is a complex rhetorical phenomenon where literal semantics contradict the actual intent, widely used in social media. In multimodal scenarios, sarcasm is often expressed through **text-image semantic conflicts**.

Existing multimodal sarcasm detection (MSD) benchmarks are mostly English-centric, with coarse-grained annotations and poor coverage of Chinese-specific sarcastic expressions (e.g., implicit satire, passive-aggressive rhetoric).

**CFMS** is a **fine-grained Chinese multimodal sarcasm dataset** built from real-world social media content. It moves beyond binary classification and enables **interpretable sarcasm reasoning** via a structured annotation pipeline.

---

## 📂 Dataset Details
CFMS consists of **2,796 high-quality image-text pairs** with rigorous data cleaning and human-machine collaborative annotation.

### Core Features
- **Triple-level Annotation Framework**
  1. Sarcasm Identification (binary classification)
  2. Sarcasm Target Recognition (entity localization)
  3. Sarcasm Explanation Generation (rhetorical mechanism interpretation)
- **High Annotation Quality**: Human-in-the-loop verification with GPT-4o pre-labeling, substantial inter-annotator agreement (Kappa=0.69)
- **Standard Split**: Train (1,956) / Validation (420) / Test (420)
- **Cross-lingual Subset**: 200 Chinese-English parallel metaphor-sarcasm samples for cross-cultural research

### Sarcasm Target Categories
Social phenomena (41%), individual behavior (23%), interpersonal relations (17%), institutional rules (12%), others (7%)

---

## 🚀 Method: PGDS (Policy-Guided Demonstration Selection)
Traditional similarity-based retrieval for in-context learning (ICL) fails to capture deep semantic conflicts in sarcasm.

**PGDS** is a **reinforcement learning-augmented ICL strategy** that dynamically selects optimal in-context exemplars **without model fine-tuning**.

### Key Mechanism
1. Multimodal encoding with BGE (text) and CLIP (image)
2. Top-50 candidate retrieval via cosine similarity
3. Lightweight MLP policy network for probability-based exemplar sampling
4. Multi-dimensional reward optimization (classification, target recognition, explanation quality)
5. Policy update with REINFORCE algorithm

### Advantages
- Model-agnostic, plug-and-play for all multimodal large language models (MLLMs)
- Outperforms random retrieval and RAG-based ICL
- Excels at metaphorical and culturally-grounded sarcasm reasoning

---

## 📊 Experimental Results
PGDS consistently achieves state-of-the-art performance on mainstream open-source MLLMs under 1-shot settings.

| Model | Method | Accuracy | F1-Score | Target Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-VL-7B-Instruct | RAG 1-shot | 76.88 | 76.13 | 45.67 |
| | **PGDS (Ours)** | **78.01** | **76.34** | **48.68** |
| InternVL2.5-8B | RAG 1-shot | 71.05 | 75.47 | 46.15 |
| | **PGDS (Ours)** | **74.76** | **76.13** | **50.89** |

### Key Findings
1. Metaphor recognition is significantly more challenging than sarcasm detection
2. Chinese-native MLLMs outperform general closed-source models on local rhetorical understanding
3. Sarcasm target localization remains the core bottleneck for current models

---

## 🎨 Sarcasm-Aware AIGC
The fine-grained explanation annotations in CFMS can serve as structured prompts for text-to-image models.
- **76% success rate** in generating images with explicit sarcastic intent
- Enables controllable and interpretable sarcastic content generation

---

## 🛠️ Quick Start
### Installation
```bash
git clone https://anonymous.4open.science/r/CFMS-E8F9.git
cd CFMS
pip install -r requirements.txt
```

### Evaluation
```bash
# Run PGDS on InternVL2.5-8B
python eval_pgds.py --model internvl2.5 --split test

# Zero-shot baseline evaluation
python eval_baseline.py --model qwen2.5-vl --mode zero_shot
```

---

## ✨ Highlights
1. **Chinese-Centric**: Covers unique implicit sarcasm in Chinese social media
2. **Interpretable**: Triple annotations for full-stack sarcasm understanding
3. **Efficient**: PGDS boosts MLLM performance without fine-tuning
4. **Multifunctional**: Supports detection, reasoning and controllable generation

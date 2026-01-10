
# CFMS: Towards Explainable and Fine-Grained Chinese Multimodal Sarcasm Detection Benchmark

> **Abstract:** Multimodal sarcasm detection (MSD) has progressed significantly, yet existing benchmarks suffer from coarse-grained annotations and limited cultural coverage. To address this, we introduce **CFMS**, the first fine-grained multimodal sarcasm dataset tailored for Chinese social media. It features a triple-level annotation framework (identification, target recognition, and explanation) to support interpretable reasoning. Furthermore, we propose **Policy-Guided Demonstration Selection (PGDS)**, a reinforcement learning-augmented In-Context Learning strategy to optimize exemplar selection for MLLMs.

---

## 🔔 News
* **[2026-01-06]** Code and Data are released.

---

## 📖 Introduction

Sarcasm is a sophisticated linguistic phenomenon where literal meanings deviate from true intents. In the multimodal era, this often manifests through conflict between text and imagery. Existing datasets often overlook unique Chinese sarcastic forms (e.g., “阴阳怪气”) and lack fine-grained reasoning annotations.

**CFMS** addresses these gaps by shifting focus from simple classification to deep interpretation.

**[IMAGE PLACEHOLDER 1]**


## 📂 The CFMS Dataset

CFMS is constructed from real-world Chinese social media, focusing on high-quality image-text pairs with strong semantic correlations.

### Key Statistics

* 
**Total Samples:** 2,796 high-quality image-text pairs.


* **Annotation Granularity:** Triple-level framework (Identification, Target Recognition, Explanation Generation).
* 
**Quality Control:** Human-in-the-loop pipeline with GPT-4o pre-annotation and expert verification.



### Data Examples

**[IMAGE PLACEHOLDER 2]**


### Comparison with Existing Datasets

We also curated a high-consistency parallel **Chinese-English metaphor subset** (200 entries each) to facilitate cross-lingual semantic studies.

---

## 🚀 Methodology: PGDS

To overcome the limitations of traditional similarity-based retrieval in In-Context Learning (ICL), we propose **Policy-Guided Demonstration Selection (PGDS)**.

**[IMAGE PLACEHOLDER 3]**


### Core Mechanism

PGDS uses a lightweight policy network  to dynamically optimize the selection of in-context exemplars. The optimization is driven by a multi-dimensional reward function:

This allows the model to select examples that improve reasoning chains rather than just surface-level similarity.




## 📊 Experimental Results

Our extensive experiments show that PGDS significantly outperforms random and RAG-based 1-shot baselines, bridging the gap between zero-shot and fine-tuning approaches.

**[IMAGE PLACEHOLDER 4]**


| Model | Method | Accuracy | F1 | Target Acc |
| --- | --- | --- | --- | --- |
| **Qwen2.5-VL** | RAG 1-shot | 76.88 | 76.13 | 45.67 |
|  | **PGDS (Ours)** | **78.01** | **76.34** | **48.68** |
| **InternVL2.5** | RAG 1-shot | 71.05 | 75.47 | 46.15 |
|  | **PGDS (Ours)** | **74.76** | **76.13** | **50.89** |
| <br>(Selected results from the paper )

 |  |  |  |  |

---

## 🎨 AI-Generated Sarcasm

We also explore using CFMS explanations as prompts for Text-to-Image models. The fine-grained explanations effectively guide AI in generating images with explicit sarcastic intent.

**[IMAGE PLACEHOLDER 5]**

# SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards  
<p align="center">
  <a href="https://arxiv.org/abs/2511.07403">
    <img src="https://img.shields.io/badge/arXiv-2511.07403-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://pixl.cs.ox.ac.uk/spatial-thinker">
    <img src="https://img.shields.io/badge/🌐%20Project%20Page-blue.svg" alt="Project Page">
  </a>
  <a href="https://huggingface.co/collections/OX-PIXL/spatialthinker">
    <img src="https://img.shields.io/badge/🤗%20Models%20%26%20Dataset-orange.svg" alt="Hugging Face Models">
  </a>
</p>

### 💡 Abstract
Multimodal large language models (MLLMs) have achieved remarkable progress
in vision–language tasks, but they continue to struggle with spatial understanding.
Existing spatial MLLMs often rely on explicit 3D inputs or architecture-specific
modifications, and remain constrained by large-scale datasets or sparse supervision.
To address these limitations, we introduce SPATIALTHINKER, a 3D-aware MLLM
trained with RL to integrate structured spatial grounding with multi-step reasoning.
The model simulates human-like spatial perception by constructing a scene graph
of task-relevant objects and spatial relations, and reasoning towards an answer via
dense spatial rewards. SPATIALTHINKER consists of two key contributions: (1)
a data synthesis pipeline that generates STVQA-7K, a high-quality spatial VQA
dataset, and (2) online RL with a multi-objective dense spatial reward enforcing
spatial grounding. SPATIALTHINKER-7B outperforms supervised fine-tuning and
the sparse RL baseline on spatial understanding and real-world VQA benchmarks,
nearly doubling the base-model gain compared to sparse RL, and surpassing GPT4o. These results showcase the effectiveness of combining spatial supervision with
reward-aligned reasoning in enabling robust 3D spatial understanding with limited
data and advancing MLLMs towards human-level visual reasoning.

<p align="center">
  <img src="assets/spatialthinker.jpg" width="60%" alt="SpatialThinker Overview">
</p>

---

### ✨ Updates
- [2025/11/11] 🔥 Code base released.
- [2025/11/08] 🔥 Model Checkpoints and Dataset released.
---

**SpatialThinker** introduces *Spatially-Aware Policy Optimization (SAPO)* — a reinforcement learning framework that integrates structured spatial grounding with lexicographic, multi-objective reward shaping to teach multimodal large language models (MLLMs) fine-grained **3D spatial reasoning**.

---

### 🧩 Requirements

- Python 3.9+
- `transformers >= 4.49.0`
- `flash-attn >= 2.4.3`
- `vllm >= 0.7.3` (0.8.0 recommended)

---

### ⚙️ Installation

```bash
pip install -e .
```

---

### 🚀 Training

#### Train **SpatialThinker Models** with STVQA-7K, Dense Spatial Rewards + GRPO

```bash
bash scripts/spatialthinker_3b_grpo.sh
```
```bash
bash scripts/spatialthinker_7b_grpo.sh
```

#### Train **Baseline Models** (Vanilla GRPO) with STVQA-7K

```bash
bash scripts/qwen_2_5_3b_stvqa_vanilla_grpo.sh
```
```bash
bash scripts/qwen_2_5_7b_stvqa_vanilla_grpo.sh
```

---

### 🧠 Merge Checkpoints to Hugging Face Format
```bash
python3 scripts/model_merger.py --local_dir path_to_your_last_actor_checkpoint
```
---
### ✅ TODOs

- [x] Release Training Code  
- [x] Release Evaluation Code  
- [x] Release Model Checkpoints  
- [x] Release STVQA-7K Training Dataset  
- [ ] Release STVQA-7K Data Generation Pipeline

---
### 📘 Citation

If you find this repository useful in your project, please consider giving a ⭐ and citing:

```bibtex
@misc{batra2025spatialthinkerreinforcing3dreasoning,  
 title={SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards},  
 author={Hunar Batra and Haoqin Tu and Hardy Chen and Yuanze Lin and Cihang Xie and Ronald Clark},  
 year={2025},  
 eprint={2511.07403},  
 archivePrefix={arXiv},  
 primaryClass={cs.CV},  
 url={https://arxiv.org/abs/2511.07403},  
}
```
---

### 🌟 Acknowledgements
This project builds upon the following open-source frameworks and works:
- [**EasyR1**](https://github.com/hiyouga/EasyR1) — An efficient, scalable, multi-modality RL training framework based on veRL  
- [**LLaMA-Factory**](https://github.com/hunarbatra/LLaMA-Factory) — Unified efficient fine-tuning of 100+ LLMs & VLMs  
- [**Qwen2.5-VL**](https://arxiv.org/abs/2502.13923) — Multimodal LLM series from the Qwen family  

---

💡 *For more details, visit the [project page](https://pixl.cs.ox.ac.uk/spatial-thinker) and our [paper on arXiv](https://arxiv.org/abs/2511.07403).*

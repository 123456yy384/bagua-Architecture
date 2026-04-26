# BaGua Architecture (八卦架构)

**Polarity-Driven Fully Dynamic Impedance Neural Network | Non-Transformer Base Architecture**

Author: Yang Enshuo (阳恩硕) | 17 | Independent Researcher  
Contact: Oyes13619690046@outlook.com

[HuggingFace](https://huggingface.co/kasuenshuo/BaGua-Architecture-0.5B-Base) | [GitHub](https://github.com/123456yy384/bagua-Architecture)

---

## What Is This

BaGua Architecture is a neural network architecture redesigned from the ground up — not built on Transformers, and does not use fixed attention mechanisms.

The core idea draws from the polarity principles of the I Ching (易经八卦): eight trigram partitions interact through real-time computed polarity vectors that dynamically determine information flow impedance between them. On every forward pass, the network topology is completely regenerated — there are no fixed connection paths.

Every mainstream LLM today — GPT, Claude, Gemini, Llama — is fundamentally built on Transformer-based fixed-topology attention. BaGua Architecture is a ground-up rethinking.

---

## Nine Core Modules

| Module | Function |
|--------|---------|
| Dynamic BaGua Array | Eight trigram partitions process in parallel; impedance matrix dynamically regulates information flow |
| Hexagram Polarity | Generates polarity vectors in real-time; opposite polarities = low-resistance passage; same polarity = high-resistance isolation |
| Elimination Audit | Micro-level: evaluates output quality of each trigram head, penalizes vague predictions |
| Low-Efficiency Elimination | Low-value trigram outputs are directly zeroed out for dynamic sparse computation |
| Compute Buffer | Gate-controlled smoothing of signal jumps caused by path pruning, ensuring training stability |
| Decentralized Self-Computation | Global survival rate tracking; auto-boosts when survival drops to prevent over-pruning |
| Left Ear In, Right Ear Out | Memory accumulates within a sequence, fully resets between sequences — natural defense against cross-sequence overfitting |
| Nine Provinces Encoding | Three-level hierarchical position awareness, computed via formulas in real-time, zero memory overhead |
| **Task Self-Awareness** | **Identifies task type from the first token across 23 preset scenarios; dynamically switches between unidirectional/bidirectional information flow** |

---

## Essential Differences from Transformer

| Aspect | Transformer | BaGua Architecture |
|--------|------------|--------------------|
| Network Topology | Fixed, identical every pass | Dynamic, rebuilt from scratch every pass |
| Attention Mechanism | All heads operate independently | Polarity-driven interaction between trigrams |
| Overfitting Defense | Relies on dropout and external methods | Native architectural defense |
| Task Adaptation | Separate models for generation vs understanding | Task self-awareness auto-switches |
| Memory Mechanism | KV Cache accumulates across sequences | Left-Ear-In-Right-Ear-Out clears between sequences |
| Position Encoding | Unified external plugin | Neuron-level three-tier hierarchical awareness |

---

## Experimental Results

### Experiment 1: Concept Validation (Random Data)

| Model | Parameters | Loss |
|-------|-----------|------|
| BaGua Architecture | **0.468M** | **6.3201** |
| Standard MHA | 0.784M | 6.3304 |

Achieved lower loss with **40% fewer parameters**.

### Experiment 2: SST-2 Sentiment Classification

| Model | Accuracy | Final Loss | Overfitting |
|-------|---------|-----------|-------------|
| BaGua Architecture (11M) | 73.4% | **Stable 0.51** | **None** |
| BERT-like (11M) | 79.7% | 0.43 → 1.35 | **Severe** |

Key finding: BERT-like severely overfits from epoch 8; BaGua remains stable throughout.

### Experiment 3: AG News Classification (20 Epochs)

| Model | Accuracy | Final Loss |
|-------|---------|-----------|
| BaGua Architecture | 89.54% | **0.31** |
| BERT-like | 91.75% | 0.43 |

Accuracy is 2.21 points behind (BERT used 3B-word pretrained embeddings; BaGua started from random initialization), but loss is **28% lower** with zero overfitting throughout.

### Experiment 4: LLM Pretraining — 258M Version

- Architecture: 258M params, 12 layers, dim=768
- Data: OpenWebText (40GB English) + Chinese Wikipedia (2GB)
- Hardware: Tesla V100-SXM2-32GB
- Progress: PPL dropped from ~116,054 (random init) to ~319 (11,000 steps)
- Capability: Generates coherent English text; Chinese in early training stages

### Experiment 5: LLM Pretraining — 0.5B Version (Latest)

- Architecture: 504M params, 24 layers, dim=1024
- Data: OpenWebText (40GB English) + Chinese Wikipedia (1.7GB)
- Hardware: Dual RTX 4090D (cloud) + Tesla V100-SXM2-32GB (local)
- Steps: ~170,000
- Best Validation PPL: ~106
- [HuggingFace Model](https://huggingface.co/kasuenshuo/BaGua-Architecture-0.5B-Base)

---

## Quick Start

```bash
pip install torch transformers numpy tqdm matplotlib

# Classification experiments
python bagua_multitask.py

# LLM training
python bagua_preprocess.py      # Data preprocessing
python bagua_llm_train_v4.py    # Training
python bagua_chat.py            # Chat inference

# 0.5B Hugging Face model inference
# See https://huggingface.co/kasuenshuo/BaGua-Architecture-0.5B-Base
```

---

## Roadmap

1. **Macro Elimination Audit**: Sentence-level output quality evaluation; low-confidence sentences auto-regenerate
2. **Expanded Chinese Data**: Target 10GB+ to enable genuine Chinese reasoning
3. **Instruction Fine-tuning**: Upgrade from text continuation to conversational model
4. **Code Compilation Verification**: Generate + verify + fix loop to guarantee runnable output
5. **Deeper Training**: Target PPL below 100 (0.5B: currently ~106)
6. **Scaling**: 1.5B+ parameter versions

---

## Honest Assessment

- Independent research — no institutional backing, no supervisor
- All experimental data is real and unmodified
- Classification accuracy 2–3 points below BERT, but anti-overfitting capability is demonstrably stronger
- LLM at early training stage (0.5B: PPL ~106; fluent conversation typically requires PPL < 50)
- Not instruction-tuned — model currently does text continuation, not Q&A
- Chinese output quality is lower due to data imbalance (40GB English vs 1.7GB Chinese)

---

## Citation

```bibtex
@misc{yang2026bagua,
  title   = {BaGua Architecture: Polarity-Driven Dynamic Impedance Neural Network},
  author  = {Yang, Enshuo},
  year    = {2026},
  url     = {https://github.com/123456yy384/bagua-Architecture}
}
```

---

*"始于AI，不止于AI。"*  
*"Started from AI, going beyond AI."*

*BaGua Architecture — from ancient Chinese philosophy, toward the future of computing.*

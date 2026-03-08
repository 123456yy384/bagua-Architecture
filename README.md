BaGua Architecture（八卦架构）

Key observation: BaGua Architecture's loss is 28% lower than BERT-like despite comparable accuracy, indicating superior feature representation quality. BERT-like shows clear overfitting (loss rising after epoch 8); BaGua remains stable through epoch 20.

Task 2: Text Coherence Judgment (2-class)
Given a context passage, identify the genuine next sentence

| Model | Parameters | Final Accuracy | Final Loss |
|-------|-----------|---------------|-----------|
| BaGua Architecture | 10.9M | 55.00% | — |
| BERT-like Encoder | 11.0M | 56.83% | — |

Note: Both models perform near chance on this task, suggesting it requires capabilities (e.g., fine-grained semantic matching) beyond what either 10M-parameter model currently provides.

Task 3: Proof of Concept (Random Sequences)
| Model | Parameters | Final Loss |
|-------|-----------|-----------|
| BaGua Architecture | 0.468M | 6.3201 |
| Standard MHA | 0.784M | 6.3304 |

BaGua achieved lower loss with 40% fewer parameters.

Honest Assessment

BaGua Architecture does not currently outperform BERT-like on accuracy metrics in these experiments. What it does demonstrate:

- Significantly lower validation loss (better internal representations)
- Superior stability — no overfitting observed across all experiments
- Competitive accuracy with the same parameter count
- A fundamentally new architectural paradigm with room for growth

This is version 1.0 of a new architecture. The Transformer itself took years of community refinement to reach its current form.

Files（The first experiment）

taichu.py          # the original architecture code
taichu_results.png    # the experiment result chart
terminal_output.png                  # the terminal output screenshot

Files（The second experiment）

bagua_multitask.py          # the multi-task comprehensive experiment code
Figure_1.png    # the final result chart

Design Philosophy

I have no formal training in machine learning. After middle school I was placed into a vocational school.

I was inspired by the Eight Trigrams (八卦) — an ancient Chinese system describing dynamic relationships between eight fundamental forces. The insight: if opposing polarities determine energy flow in the universe, why not in a neural network?

Every module grew from that question. The polarity clash. The elimination of weak pathways. The memory that resets between sequences. The hierarchical position encoding. All of it is one idea: let structure emerge from data, not be imposed on it.

Citation

@misc{yang2026bagua,
  title     = {BaGua Architecture: Polarity-Driven Dynamic Impedance for Self-Evolving Neural Networks},
  author    = {Yang, Enshuo},
  year      = {2026},
  month     = {March},
  publisher = {GitHub},
  url       = {https://github.com/123456yy384/bagua-Architecture
}

Contact

Independent researcher seeking collaboration.  
Email: Oyes13619690046@outlook.com

BaGua Architecture was originally named TaiChu (太初) — The Great Beginning.
The name changed. The idea did not.

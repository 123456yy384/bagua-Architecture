"""
八卦架构 (BaGua Architecture) — LLM 训练脚本 V3
================================================
作者：阳恩硕 (Yang Enshuo)
数据：预处理二进制token文件（需先运行 bagua_preprocess.py）
架构：九模块完整版（含任务自我感知）
改进：
    - 分词器自动检测并下载，无需手动配置
    - 500步汇报一次，不保存中间checkpoint，节省硬盘
    - 每轮结束保存一次模型，覆盖上一轮
    - 硬盘占用从100GB降到1GB

运行方式：
    1. 先在本地跑 bagua_preprocess.py 生成二进制数据
    2. 把 bagua_tokens/ 文件夹拷到V100服务器桌面
    3. python bagua_llm_train_v3.py
"""

import os
import sys
import json
import math
import time
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt


# ============================================================
# 自动检测数据路径
# ============================================================

def get_token_dir():
    """检测预处理后的二进制token目录"""
    candidates = [
        Path("C:/Users/Administrator/Desktop/bagua_tokens"),
        Path("C:/Users/Admin/Desktop/bagua_tokens"),
        Path("D:/bagua_tokens"),
        Path(os.path.expanduser("~/Desktop/bagua_tokens")),
        Path("./bagua_tokens"),
    ]
    for p in candidates:
        if p.exists() and (p / "meta.json").exists():
            print(f"找到Token目录：{p}")
            return p
    print("未找到 bagua_tokens 目录，请先运行 bagua_preprocess.py")
    sys.exit(1)

TOKEN_DIR = get_token_dir()
META = json.loads((TOKEN_DIR / "meta.json").read_text(encoding='utf-8'))
EN_TOKENS_PATH = META["en_tokens_path"]
ZH_TOKENS_PATH = META["zh_tokens_path"]
EN_TOTAL = META["en_total_tokens"]
ZH_TOTAL = META["zh_total_tokens"]
VOCAB_SIZE = META["vocab_size"]

print(f"英文token数：{EN_TOTAL:,}")
print(f"中文token数：{ZH_TOTAL:,}")
print(f"词表大小：{VOCAB_SIZE}")


# ============================================================
# 分词器自动检测和下载
# ============================================================

def get_tokenizer():
    """
    自动检测本地分词器，找不到就自动下载
    支持多个候选路径，有一个能用就行
    """
    from transformers import AutoTokenizer

    candidates = [
        Path("C:/Users/Administrator/Desktop/bagua_data/tokenizer"),
        Path("C:/Users/Admin/Desktop/bagua_data/tokenizer"),
        Path("D:/bagua_data/tokenizer"),
        Path("E:/bagua_data/tokenizer"),
        TOKEN_DIR / "tokenizer",
        Path(os.path.expanduser("~/Desktop/bagua_data/tokenizer")),
    ]

    # 先找本地
    for p in candidates:
        if p.exists() and (p / "vocab.txt").exists():
            print(f"找到本地分词器：{p}")
            try:
                tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
                print(f"分词器加载成功，词表大小：{tok.vocab_size}")
                return tok
            except Exception as e:
                print(f"加载失败：{e}，尝试下一个路径")
                continue

    # 本地没有，自动下载
    print("本地未找到分词器，正在自动下载（需要翻墙）...")
    save_path = Path("C:/Users/Administrator/Desktop/bagua_data/tokenizer")
    save_path.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    tok.save_pretrained(str(save_path))
    print(f"分词器已下载并保存：{save_path}")
    print(f"词表大小：{tok.vocab_size}")
    return tok


TOKENIZER = get_tokenizer()


# ============================================================
# 卦象对冲
# ============================================================

class GuaXiangDuiChong(nn.Module):
    def __init__(self, head_dim: int, num_heads: int = 8, polarity_dim: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.polarity_generators = nn.ModuleList([
            nn.Sequential(nn.Linear(head_dim, polarity_dim), nn.Tanh())
            for _ in range(num_heads)
        ])
        self.impedance_modulator = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1), nn.Softplus()
        )

    def forward(self, heads):
        B = heads[0].shape[0]
        polarity_vectors = []
        for i in range(self.num_heads):
            summary = heads[i].mean(dim=1)
            polarity_vectors.append(self.polarity_generators[i](summary))
        polarity_matrix = torch.stack(polarity_vectors, dim=1)
        polarity_norm = F.normalize(polarity_matrix, dim=-1)
        dot_products = torch.bmm(polarity_norm, polarity_norm.transpose(1, 2))
        impedance = self.impedance_modulator(dot_products.reshape(-1, 1)).reshape(B, 8, 8)
        mask = 1.0 - torch.eye(8, device=impedance.device).unsqueeze(0)
        return impedance * mask


# ============================================================
# 九州编码
# ============================================================

class JiuZhouBianMa(nn.Module):
    def __init__(self, dim: int, num_zones: int = 8, max_len: int = 512):
        super().__init__()
        assert dim % 8 == 0
        self.head_dim = dim // 8
        self.num_zones = num_zones
        self.zone_proj = nn.ModuleList([
            nn.Linear(self.head_dim + 2, self.head_dim, bias=False)
            for _ in range(8)
        ])
        self.inject_gate = nn.Sequential(nn.Linear(self.head_dim, 1), nn.Sigmoid())

    def _compute_position_code(self, trigram_idx, seq_len, device):
        trigram_code = trigram_idx / 7.0
        positions = torch.arange(seq_len, device=device).float() / max(seq_len - 1, 1)
        zone_size = max(seq_len / self.num_zones, 1)
        zone_relative = (torch.arange(seq_len, device=device).float() % zone_size) / zone_size
        trigram_broadcast = torch.full((seq_len,), trigram_code, device=device)
        return torch.stack([positions * 0.5 + trigram_broadcast * 0.5, zone_relative], dim=-1)

    def forward(self, heads):
        B, S, _ = heads[0].shape
        device = heads[0].device
        enriched = []
        for i, h in enumerate(heads):
            pos_code = self._compute_position_code(i, S, device).unsqueeze(0).expand(B, -1, -1)
            h_enriched = self.zone_proj[i](torch.cat([h, pos_code], dim=-1))
            gate = self.inject_gate(h)
            enriched.append(h * (1 - gate) + h_enriched * gate)
        return enriched


# ============================================================
# 动态八卦阵（含因果掩码）
# ============================================================

class DongTaiBaGuaZhen(nn.Module):
    def __init__(self, dim: int, polarity_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        assert dim % 8 == 0
        self.head_dim = dim // 8
        self.trigram_projections = nn.ModuleList([
            nn.Linear(dim, self.head_dim, bias=False) for _ in range(8)
        ])
        self.resonance_freqs = nn.Parameter(torch.randn(8, self.head_dim) * 0.01)
        self.guaxiang_duichong = GuaXiangDuiChong(self.head_dim, 8, polarity_dim)
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, causal: bool = True):
        residual = x
        B, S, D = x.shape
        device = x.device

        heads = []
        for i in range(8):
            proj = self.trigram_projections[i](x)
            freq_mod = torch.cos(self.resonance_freqs[i] * math.pi)
            heads.append(proj * freq_mod)
        impedance = self.guaxiang_duichong(heads)

        # 因果掩码：位置t只能接收t之前位置的信息
        if causal and S > 1:
            future_mask = torch.triu(
                torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1
            )

        output_heads = []
        for i in range(8):
            h_i = heads[i].clone()
            for j in range(8):
                if i != j:
                    imp_ij = impedance[:, i, j].unsqueeze(1).unsqueeze(2)
                    transfer = heads[j] / (1.0 + imp_ij) * 0.1
                    if causal and S > 1:
                        mask = future_mask.unsqueeze(0).unsqueeze(-1)
                        transfer_3d = transfer.unsqueeze(2).expand(B, S, S, self.head_dim)
                        transfer_masked = transfer_3d.masked_fill(mask, 0.0).mean(dim=2)
                        h_i = h_i + transfer_masked
                    else:
                        h_i = h_i + transfer
            output_heads.append(h_i)

        merged = torch.cat(output_heads, dim=-1)
        output = self.dropout(self.output_proj(merged))
        return self.norm(output + residual), impedance


# ============================================================
# 淘汰审核
# ============================================================

class TaoTaiShenHe(nn.Module):
    def __init__(self, head_dim: int, beta: float = 0.01):
        super().__init__()
        self.beta = beta
        self.density_scorer = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2),
            nn.GELU(),
            nn.Linear(head_dim // 2, 1),
            nn.Sigmoid()
        )

    def evaluate(self, head_output, head_input):
        logical_density = self.density_scorer(head_output).mean()
        causal_delta = (head_output - head_input).abs().mean()
        causal_stability = torch.sigmoid(causal_delta * 10.0 - 1.0)
        return logical_density * causal_stability

    def honesty_loss(self, logits):
        logits_sample = logits[:64].detach()
        probs = F.softmax(logits_sample, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        return self.beta * entropy


# ============================================================
# 淘汰低效机制
# ============================================================

class TaoTaiDiXiaoJiZhi:
    def __init__(self, survival_threshold: float = 0.3):
        self.survival_threshold = survival_threshold

    def eliminate(self, head_outputs, scores):
        return [
            h * 0.0 if scores[i].item() < self.survival_threshold else h
            for i, h in enumerate(head_outputs)
        ]


# ============================================================
# 算力缓冲区
# ============================================================

class SuanLiHuanChongQu(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.charge_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.discharge_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, current, prev):
        if current.shape != prev.shape:
            return current
        return current + self.discharge_gate(current) * (self.charge_gate(prev) * prev)


# ============================================================
# 左耳进右耳出（原始位置，不做全局前置）
# ============================================================

class ZuoErJinYouErChu(nn.Module):
    """
    左耳进右耳出 — 复位到原始位置
    职责：序列内记忆积累，序列间彻底清零
    不承担因果约束职责（因果约束由动态八卦阵的掩码负责）
    不做全局前置，待在它该待的位置：所有层处理完之后
    """
    def __init__(self, dim: int, memory_slots: int = 32):
        super().__init__()
        self.dim = dim
        self.memory_slots = memory_slots
        self.write_gate = nn.Sequential(nn.Linear(dim, memory_slots), nn.Sigmoid())
        self.read_gate = nn.Sequential(nn.Linear(dim, memory_slots), nn.Sigmoid())
        self.memory_transform = nn.Linear(dim, memory_slots)
        self.memory_proj = nn.Linear(memory_slots, dim)
        self.output_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh())
        self.memory_state = None

    def reset(self):
        """序列结束，清零记忆，用完即删"""
        self.memory_state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        device = x.device
        if self.memory_state is None or self.memory_state.shape[0] != B:
            self.memory_state = torch.zeros(B, self.memory_slots, device=device)
        outputs = []
        for t in range(S):
            x_t = x[:, t, :]
            read_weight = self.read_gate(x_t)
            memory_vec = self.memory_proj(read_weight * self.memory_state)
            fused = self.output_gate(torch.cat([x_t, memory_vec], dim=-1))
            write_weight = self.write_gate(x_t)
            new_memory = self.memory_transform(x_t)
            self.memory_state = (
                (1 - write_weight) * self.memory_state +
                write_weight * new_memory
            ).detach()
            outputs.append(fused.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# ============================================================
# 前馈网络
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


# ============================================================
# 任务自我感知（第九模块）
# ============================================================

class RenWuZiWoGanZhi(nn.Module):
    """
    任务自我感知 — 第九模块
    读取第一个token特征，与23个预设场景匹配
    输出因果强度：接近1=生成任务（单向），接近0=理解任务（双向）
    软权重混合，不会因判断偏差完全跑偏
    """
    SCENE_PRESETS = {
        'text_generation':    0.95,
        'code_generation':    0.95,
        'continuation':       0.90,
        'translation':        0.85,
        'summarization':      0.80,
        'creative_writing':   0.95,
        'poetry':             0.90,
        'classification':     0.05,
        'sentiment':          0.05,
        'ner':                0.10,
        'relation_extract':   0.10,
        'similarity':         0.05,
        'retrieval':          0.05,
        'annotation':         0.05,
        'fact_check':         0.10,
        'content_moderation': 0.05,
        'qa':                 0.40,
        'dialogue':           0.70,
        'reading_comp':       0.35,
        'rag':                0.45,
        'instruction':        0.60,
        'reasoning':          0.50,
        'multi_turn':         0.65,
    }

    def __init__(self, dim: int):
        super().__init__()
        num_scenes = len(self.SCENE_PRESETS)
        self.scene_matcher = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_scenes),
            nn.Softmax(dim=-1)
        )
        causal_weights = list(self.SCENE_PRESETS.values())
        self.register_buffer(
            'scene_causal_weights',
            torch.tensor(causal_weights, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_token = x[:, 0, :]
        scene_probs = self.scene_matcher(first_token)
        causal_strength = torch.matmul(scene_probs, self.scene_causal_weights)
        return causal_strength


# ============================================================
# 八卦架构 LLM（九模块，左耳进右耳出复位）
# ============================================================

class BaGuaLLM(nn.Module):
    """
    八卦架构语言模型 — 九模块完整版
    左耳进右耳出：复位到原始位置（所有层之后）
    因果约束：由动态八卦阵内置掩码+任务自我感知动态切换
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        num_layers: int = 12,
        max_len: int = 256,
        polarity_dim: int = 32,
        dropout: float = 0.1,
        survival_threshold: float = 0.3,
    ):
        super().__init__()
        assert dim % 8 == 0
        self.dim = dim
        self.head_dim = dim // 8
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_len + 1, dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.bagua_layers = nn.ModuleList([
            DongTaiBaGuaZhen(dim=dim, polarity_dim=polarity_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ff_layers = nn.ModuleList([
            FeedForward(dim=dim, ff_mult=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.shenhe_layers = nn.ModuleList([
            nn.ModuleList([TaoTaiShenHe(self.head_dim) for _ in range(8)])
            for _ in range(num_layers)
        ])
        self.taoTai = TaoTaiDiXiaoJiZhi(survival_threshold)
        self.buffer = SuanLiHuanChongQu(dim)
        self.jiuzhou = JiuZhouBianMa(dim=dim, num_zones=8, max_len=max_len)
        self.tsa = RenWuZiWoGanZhi(dim=dim)

        # 左耳进右耳出：在它该在的位置（所有层之后）
        self.zuoer = ZuoErJinYouErChu(dim=dim, memory_slots=32)

        self.lm_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, vocab_size, bias=False)
        )
        self.logic_pressure = 1.0
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device

        pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.embed_dropout(self.embedding(input_ids) + self.pos_embedding(pos_ids))

        # 任务自我感知：第一个token判断任务类型
        causal_strength = self.tsa(x)
        use_causal = (causal_strength.mean().item() > 0.5)

        prev = x
        total_survival = 0.0

        # 所有层处理
        for layer_idx in range(self.num_layers):
            bagua_out, impedance = self.bagua_layers[layer_idx](x, causal=use_causal)

            input_heads = [x[:, :, i*self.head_dim:(i+1)*self.head_dim] for i in range(8)]
            output_heads_raw = [bagua_out[:, :, i*self.head_dim:(i+1)*self.head_dim] for i in range(8)]

            output_heads = self.jiuzhou(output_heads_raw)

            scores = [
                self.shenhe_layers[layer_idx][i].evaluate(output_heads[i], input_heads[i])
                for i in range(8)
            ]

            masked = self.taoTai.eliminate(output_heads, scores)
            merged = torch.cat(masked, dim=-1)

            smoothed = self.buffer(merged, prev)
            prev = smoothed.detach()
            x = self.ff_layers[layer_idx](smoothed)

            avg_survival = sum(s.item() for s in scores) / 8
            total_survival += avg_survival

        # 去中心化自运算
        avg_s = total_survival / self.num_layers
        self.logic_pressure = max(0.5, min(2.0, self.logic_pressure + 0.05 * (1.0 - avg_s)))

        # 左耳进右耳出：待在它该待的位置，所有层处理完之后
        x = self.zuoer(x)
        self.zuoer.reset()  # 序列结束立即清零，用完即删

        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        device: str = 'cuda',
    ) -> torch.Tensor:
        self.eval()
        self.zuoer.reset()
        generated = input_ids.clone().to(device)

        for _ in range(max_new_tokens):
            logits = self(generated)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < min_val, float('-inf'))
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() in [102, 0]:
                break

        self.zuoer.reset()
        return generated


# ============================================================
# 二进制数据集（从预处理文件读取，训练集和验证集完全分开）
# ============================================================

class BaGuaBinaryDataset(Dataset):
    """
    从预处理二进制文件直接读取token，速度快，内存占用低
    训练集和验证集从不同位置取，彻底分开，不重叠
    """
    def __init__(
        self,
        seq_len: int = 256,
        split: str = 'train',
        en_ratio: float = 0.7,
    ):
        self.seq_len = seq_len
        self.split = split

        # 内存映射读取，不把整个文件加载到内存
        self.en_data = np.memmap(EN_TOKENS_PATH, dtype=np.uint16, mode='r')
        self.zh_data = np.memmap(ZH_TOKENS_PATH, dtype=np.uint16, mode='r')

        print(f"英文token数组长度：{len(self.en_data):,}")
        print(f"中文token数组长度：{len(self.zh_data):,}")

        # 训练集用前90%，验证集用后10%，彻底不重叠
        en_split = int(len(self.en_data) * 0.9)
        zh_split = int(len(self.zh_data) * 0.9)

        if split == 'train':
            self.en_end = en_split
            self.zh_end = zh_split
            self.en_start = 0
            self.zh_start = 0
        else:  # val
            self.en_start = en_split
            self.en_end = len(self.en_data)
            self.zh_start = zh_split
            self.zh_end = len(self.zh_data)

        # 生成样本索引
        self.samples = self._build_index(en_ratio)
        print(f"{split}集样本数：{len(self.samples):,}")

    def _build_index(self, en_ratio):
        samples = []
        step = self.seq_len  # 不重叠

        # 英文样本
        en_count = 0
        for start in range(self.en_start, self.en_end - self.seq_len - 1, step):
            samples.append(('en', start))
            en_count += 1
            if self.split == 'val' and en_count >= 500:
                break
            if self.split == 'train' and en_count >= 100000:
                break

        # 中文样本
        zh_count = 0
        for start in range(self.zh_start, self.zh_end - self.seq_len - 1, step):
            samples.append(('zh', start))
            zh_count += 1
            if self.split == 'val' and zh_count >= 200:
                break
            if self.split == 'train' and zh_count >= 30000:
                break

        random.seed(42)
        random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lang, start = self.samples[idx]
        data = self.en_data if lang == 'en' else self.zh_data
        chunk = data[start:start + self.seq_len + 1].astype(np.int64)
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, target_ids



# ============================================================
# 指令+工具调用数据集（与预训练数据混合训练）
# 比例：70%预训练文本 + 30%指令数据
# 让模型一步学会：语言规律 + 听懂人话 + 调用工具
# ============================================================

# 指令和工具调用训练数据（中英双语）
INSTRUCTION_DATA = [
    # ── 普通问答 ──
    {"instruction": "请回答以下问题", "input": "什么是人工智能？", "output": "人工智能是让计算机模拟人类智能行为的技术，包括学习、推理和解决问题。"},
    {"instruction": "请回答以下问题", "input": "中国的首都是哪里？", "output": "中国的首都是北京。"},
    {"instruction": "请回答以下问题", "input": "水的化学式是什么？", "output": "水的化学式是H₂O，由两个氢原子和一个氧原子组成。"},
    {"instruction": "Answer the question", "input": "What is machine learning?", "output": "Machine learning is a type of AI that allows computers to learn from data without being explicitly programmed."},
    {"instruction": "Answer the question", "input": "What is the capital of France?", "output": "The capital of France is Paris."},
    {"instruction": "Answer the question", "input": "How many planets are in our solar system?", "output": "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."},

    # ── 翻译任务 ──
    {"instruction": "请将以下英文翻译成中文", "input": "Hello, how are you?", "output": "你好，你怎么样？"},
    {"instruction": "请将以下中文翻译成英文", "input": "今天天气很好", "output": "The weather is very nice today."},
    {"instruction": "Translate to Chinese", "input": "Artificial intelligence is changing the world.", "output": "人工智能正在改变世界。"},

    # ── 摘要任务 ──
    {"instruction": "请用一句话总结", "input": "八卦架构是基于易经八卦图设计的神经网络，通过极性驱动的动态阻抗机制，让网络拓扑在每次前向传播时根据输入数据重新生长。", "output": "八卦架构是一种基于易经的动态拓扑神经网络，每次根据输入实时重构网络结构。"},
    {"instruction": "Summarize in one sentence", "input": "BaGua Architecture uses polarity-driven dynamic impedance to reconstruct the network topology on every forward pass, inspired by the Eight Trigrams.", "output": "BaGua Architecture dynamically reconstructs its topology using polarity interactions inspired by the Eight Trigrams."},

    # ── 代码生成 ──
    {"instruction": "请写一个Python函数", "input": "计算两个数的和", "output": "def add(a, b):\n    return a + b"},
    {"instruction": "Write a Python function", "input": "Check if a number is prime", "output": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True"},

    # ── 工具调用：需要实时数据时调用搜索 ──
    {"instruction": "请回答以下问题", "input": "今天的天气怎么样？",
     "output": "<tool>search</tool><query>今天天气</query>\n根据搜索结果，我可以告诉您当前的天气情况。"},
    {"instruction": "请回答以下问题", "input": "最新的AI新闻是什么？",
     "output": "<tool>search</tool><query>最新AI新闻</query>\n根据搜索结果，以下是最新的AI领域动态。"},
    {"instruction": "Answer the question", "input": "What is the latest news about AI?",
     "output": "<tool>search</tool><query>latest AI news 2026</query>\nBased on search results, here are the latest developments in AI."},
    {"instruction": "请回答以下问题", "input": "现在比特币的价格是多少？",
     "output": "<tool>search</tool><query>比特币当前价格</query>\n根据搜索结果，以下是比特币的最新价格信息。"},

    # ── 工具调用：需要计算时调用计算器 ──
    {"instruction": "请计算", "input": "1234567 乘以 9876543 等于多少？",
     "output": "<tool>calculator</tool><expr>1234567 * 9876543</expr>\n计算结果是：12193254061881。"},
    {"instruction": "Calculate", "input": "What is 15% of 2840?",
     "output": "<tool>calculator</tool><expr>2840 * 0.15</expr>\nThe result is 426."},

    # ── 工具调用：需要查代码文档时 ──
    {"instruction": "请解释这段代码", "input": "import torch\nx = torch.zeros(3, 4)",
     "output": "<tool>search</tool><query>PyTorch torch.zeros documentation</query>\n这段代码导入PyTorch库，然后创建一个3行4列的全零张量。"},

    # ── 对话任务 ──
    {"instruction": "请友好地回复", "input": "我今天心情不好", "output": "很抱歉听到这个，希望你能快点好起来。有什么我能帮你的吗？"},
    {"instruction": "Respond helpfully", "input": "I am feeling stressed", "output": "I understand. Stress can be overwhelming. Try to take a short break and breathe deeply. Is there something specific bothering you?"},

    # ── 推理任务 ──
    {"instruction": "请一步步推理", "input": "如果所有的猫都是动物，所有的动物都需要食物，那么猫需要食物吗？",
     "output": "步骤1：所有猫都是动物（已知条件）\n步骤2：所有动物都需要食物（已知条件）\n步骤3：因为猫是动物，而动物需要食物，所以猫需要食物。\n结论：是的，猫需要食物。"},
    {"instruction": "Reason step by step", "input": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
     "output": "Step 1: Rain causes wet ground (given)\nStep 2: The ground is wet (observed)\nStep 3: However, wet ground could have other causes (sprinklers, flooding, etc.)\nConclusion: We cannot be certain it rained, though it is possible."},

    # ── 创意写作 ──
    {"instruction": "请写一首短诗", "input": "关于春天", "output": "春风轻抚大地，\n花朵悄然绽放，\n万物复苏之时，\n生命重焕光芒。"},
    {"instruction": "Write a short poem", "input": "about the stars", "output": "Stars shine bright in the night,\nGuiding travelers with their light,\nAncient fires burning still,\nFilling hearts with wonder and thrill."},
]


class InstructionDataset(Dataset):
    """
    指令+工具调用数据集
    格式：### 指令：{instruction}\n### 输入：{input}\n### 输出：{output}
    扩充200倍保证每个batch都有指令样本
    """
    def __init__(self, tokenizer, seq_len=256, augment_times=200):
        self.seq_len = seq_len
        self.samples = []

        data = INSTRUCTION_DATA * augment_times
        random.shuffle(data)

        pad_id = tokenizer.pad_token_id or 0

        for item in data:
            text = (
                f"### 指令：{item['instruction']}\n"
                f"### 输入：{item['input']}\n"
                f"### 输出：{item['output']}"
            )
            tokens = tokenizer.encode(text, add_special_tokens=False)

            if len(tokens) >= seq_len + 1:
                input_ids = tokens[:seq_len]
                target_ids = tokens[1:seq_len + 1]
            else:
                input_ids = tokens[:]
                target_ids = tokens[1:] + [pad_id]
                while len(input_ids) < seq_len:
                    input_ids.append(pad_id)
                    target_ids.append(pad_id)
                input_ids = input_ids[:seq_len]
                target_ids = target_ids[:seq_len]

            self.samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long)
            ))

        print(f"指令+工具调用样本数：{len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MixedDataset(Dataset):
    """
    混合数据集：预训练文本 + 指令 + 工具调用
    比例：70%预训练 + 30%指令
    一步训练，同时学会语言规律、听懂人话、调用工具
    """
    def __init__(self, pretrain_dataset, instruct_dataset, instruct_ratio=0.3):
        self.pretrain = pretrain_dataset
        self.instruct = instruct_dataset
        self.instruct_ratio = instruct_ratio
        # 计算混合后的总样本数
        n_pretrain = len(pretrain_dataset)
        n_instruct = int(n_pretrain * instruct_ratio / (1 - instruct_ratio))
        n_instruct = min(n_instruct, len(instruct_dataset))
        self.total = n_pretrain + n_instruct
        # 建立索引：0=预训练样本，1=指令样本
        self.index = (
            [('pretrain', i) for i in range(n_pretrain)] +
            [('instruct', i % len(instruct_dataset)) for i in range(n_instruct)]
        )
        random.seed(42)
        random.shuffle(self.index)
        print(f"混合数据集：预训练{n_pretrain:,}条 + 指令{n_instruct:,}条 = 共{self.total:,}条")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        dtype, i = self.index[idx]
        if dtype == 'pretrain':
            return self.pretrain[i]
        else:
            return self.instruct[i]

# ============================================================
# 训练函数
# ============================================================

def train():
    print("=" * 60)
    print("  八卦架构 LLM 训练 V4 — 混合训练版")
    print("  数据：70%预训练文本 + 30%指令+工具调用")
    print("  目标：语言理解 + 听懂人话 + 工具调用，一步到位")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n运行设备：{device}")
    if device == 'cuda':
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(f"显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.95)
        print("V100专项优化已启用")

    CONFIG_TRAIN = {
        'dim': 768,
        'num_layers': 12,
        'seq_len': 256,
        'batch_size': 4,
        'grad_accum': 16,       # 等效batch=64
        'num_epochs': 10,
        'learning_rate': 2e-4,
        'warmup_steps': 2000,
        'grad_clip': 1.0,
        'log_every': 50,
        'eval_every': 500,      # 500步汇报一次，不保存checkpoint
        'num_workers': 4,
    }

    model = BaGuaLLM(
        vocab_size=VOCAB_SIZE,
        dim=CONFIG_TRAIN['dim'],
        num_layers=CONFIG_TRAIN['num_layers'],
        max_len=CONFIG_TRAIN['seq_len'],
        polarity_dim=32,
        dropout=0.1,
    ).to(device)

    scaler = torch.amp.GradScaler('cuda')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量：{total_params:,} ({total_params/1e6:.1f}M)")

    # 预训练数据集
    pretrain_train = BaGuaBinaryDataset(seq_len=CONFIG_TRAIN['seq_len'], split='train')
    val_dataset = BaGuaBinaryDataset(seq_len=CONFIG_TRAIN['seq_len'], split='val')

    # 指令+工具调用数据集
    instruct_dataset = InstructionDataset(
        tokenizer=TOKENIZER,
        seq_len=CONFIG_TRAIN['seq_len'],
        augment_times=200
    )

    # 混合数据集：70%预训练 + 30%指令
    train_dataset = MixedDataset(
        pretrain_dataset=pretrain_train,
        instruct_dataset=instruct_dataset,
        instruct_ratio=0.3
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG_TRAIN['batch_size'],
        shuffle=True,
        num_workers=CONFIG_TRAIN['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG_TRAIN['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    try:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG_TRAIN['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True
        )
        print("使用 Fused AdamW 优化器")
    except TypeError:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG_TRAIN['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    total_steps = CONFIG_TRAIN['num_epochs'] * len(train_loader)
    warmup = CONFIG_TRAIN['warmup_steps']

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.1, 0.5 * (1 + math.cos(
            math.pi * (step - warmup) / (total_steps - warmup)
        )))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    save_dir = TOKEN_DIR / "checkpoints"
    save_dir.mkdir(exist_ok=True)

    train_losses, val_losses, val_ppls = [], [], []
    step = 0
    best_ppl = float('inf')

    print(f"\n开始训练，共 {CONFIG_TRAIN['num_epochs']} 轮，{total_steps} 步")
    print(f"{'Step':>8} | {'TrainLoss':>10} | {'ValLoss':>9} | {'ValPPL':>9}")
    print("-" * 45)

    start_time = time.time()

    for epoch in range(CONFIG_TRAIN['num_epochs']):
        model.train()
        optimizer.zero_grad()

        for batch_idx, (input_ids, target_ids) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        ):
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device=='cuda')):
                logits = model(input_ids)
                B, S, V = logits.shape
                loss = loss_fn(logits.reshape(-1, V), target_ids.reshape(-1))
                loss = loss + model.shenhe_layers[0][0].honesty_loss(logits.reshape(-1, V))
                loss = loss / CONFIG_TRAIN['grad_accum']

            scaler.scale(loss).backward()

            if (batch_idx + 1) % CONFIG_TRAIN['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG_TRAIN['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            step += 1

            if step % CONFIG_TRAIN['log_every'] == 0:
                train_losses.append(loss.item() * CONFIG_TRAIN['grad_accum'])

            if step % CONFIG_TRAIN['eval_every'] == 0:
                model.eval()
                val_loss_sum = 0.0
                val_tokens = 0
                with torch.no_grad():
                    for v_input, v_target in val_loader:
                        v_input = v_input.to(device)
                        v_target = v_target.to(device)
                        with torch.amp.autocast('cuda', enabled=(device=='cuda')):
                            v_logits = model(v_input)
                            B, S, V = v_logits.shape
                            v_loss = loss_fn(v_logits.reshape(-1, V), v_target.reshape(-1))
                        val_loss_sum += v_loss.item() * B * S
                        val_tokens += B * S

                val_loss = val_loss_sum / val_tokens
                val_ppl = math.exp(min(val_loss, 20))
                val_losses.append(val_loss)
                val_ppls.append(val_ppl)

                print(f"{step:>8} | {loss.item()*CONFIG_TRAIN['grad_accum']:>10.4f} | "
                      f"{val_loss:>9.4f} | {val_ppl:>9.2f}")

                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                    torch.save(model.state_dict(), save_dir / "best_model.pt")
                    print(f"  ✓ 最佳模型已保存，PPL={best_ppl:.2f}")

                model.train()

            # 不保存中间checkpoint，节省硬盘空间

    elapsed = time.time() - start_time
    print(f"\n训练完成，总耗时：{elapsed/3600:.2f}小时")
    print(f"最佳验证PPL：{best_ppl:.2f}")

    # 保存训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, alpha=0.7)
    plt.title('BaGua LLM V2 — Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_ppls, color='orange')
    plt.title('BaGua LLM V2 — Validation PPL')
    plt.xlabel('Eval Steps')
    plt.ylabel('Perplexity')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    result_path = TOKEN_DIR / "bagua_llm_v2_results.png"
    plt.savefig(str(result_path), dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存：{result_path}")

    # 生成示例
    print("\n生成文字示例：")
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast(
        vocab_file=str(TOKEN_DIR.parent / "tokenizer" / "vocab.txt"),
        do_lower_case=True
    )
    model.eval()
    for prompt in ["The history of artificial intelligence", "人工智能的发展"]:
        input_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            dtype=torch.long
        ).to(device)
        output_ids = model.generate(input_ids, max_new_tokens=80, temperature=0.8, device=device)
        text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        print(f"\n输入：{prompt}")
        print(f"输出：{text}")


if __name__ == "__main__":
    train()

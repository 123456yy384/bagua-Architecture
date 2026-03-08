"""
八卦架构 (BaGua Architecture) — 多任务综合测试
================================================
作者：阳恩硕 (Yang Enshuo)
任务一：AG News 新闻主题分类（4分类，全局语义理解）
任务二：WikiText-103 文本连贯性判断（上下文理解，贴近大模型实际能力）
对比：八卦架构 vs BERT-like 双向编码器（公平对比）
规模：小参数版（~10M）和大参数版（~100M）各跑一次

运行方式：
    pip install torch datasets transformers matplotlib tqdm
    python bagua_multitask.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from typing import List, Tuple
import math
from tqdm import tqdm
import random


# ============================================================
# 卦象对冲 — 全动态阻抗驱动器
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

    def forward(self, heads: List[torch.Tensor]) -> torch.Tensor:
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
# 九州编码 — 神经元层级位置感知
# ============================================================

class JiuZhouBianMa(nn.Module):
    """
    九州编码 — 层级位置感知
    第一级：卦象分区（8个大区）
    第二级：序列位置小分区
    第三级：组内相对位置
    用公式实时计算，用完即扔，零额外显存
    """
    def __init__(self, dim: int, num_zones: int = 8, max_len: int = 128):
        super().__init__()
        assert dim % 8 == 0
        self.head_dim = dim // 8
        self.num_zones = num_zones
        self.zone_proj = nn.ModuleList([
            nn.Linear(self.head_dim + 2, self.head_dim, bias=False)
            for _ in range(8)
        ])
        self.inject_gate = nn.Sequential(nn.Linear(self.head_dim, 1), nn.Sigmoid())

    def _compute_position_code(self, trigram_idx: int, seq_len: int, device):
        trigram_code = trigram_idx / 7.0
        positions = torch.arange(seq_len, device=device).float() / max(seq_len - 1, 1)
        zone_size = seq_len / self.num_zones
        zone_relative = (torch.arange(seq_len, device=device).float() % max(zone_size, 1)) / max(zone_size, 1)
        trigram_broadcast = torch.full((seq_len,), trigram_code, device=device)
        return torch.stack([positions * 0.5 + trigram_broadcast * 0.5, zone_relative], dim=-1)

    def forward(self, heads: list) -> list:
        B, S, _ = heads[0].shape
        device = heads[0].device
        enriched = []
        for i, h in enumerate(heads):
            pos_code = self._compute_position_code(i, S, device).unsqueeze(0).expand(B, -1, -1)
            h_with_pos = torch.cat([h, pos_code], dim=-1)
            h_enriched = self.zone_proj[i](h_with_pos)
            gate = self.inject_gate(h)
            enriched.append(h * (1 - gate) + h_enriched * gate)
        return enriched


# ============================================================
# 动态八卦阵 — 核心信息处理单元
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        heads = []
        for i in range(8):
            proj = self.trigram_projections[i](x)
            freq_mod = torch.cos(self.resonance_freqs[i] * math.pi)
            heads.append(proj * freq_mod)
        impedance = self.guaxiang_duichong(heads)
        output_heads = []
        for i in range(8):
            h_i = heads[i].clone()
            for j in range(8):
                if i != j:
                    imp_ij = impedance[:, i, j].unsqueeze(1).unsqueeze(2)
                    h_i = h_i + heads[j] / (1.0 + imp_ij) * 0.1
            output_heads.append(h_i)
        merged = torch.cat(output_heads, dim=-1)
        output = self.dropout(self.output_proj(merged))
        return self.norm(output + residual), impedance


# ============================================================
# 淘汰审核（合并版：卦象编码+淘汰审核）
# ============================================================

class TaoTaiShenHe(nn.Module):
    """
    淘汰审核 — 评估卦象价值 + 惩罚模糊预测
    原卦象编码和淘汰审核合并为一个模块
    """
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

    def honesty_loss(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.detach(), dim=-1)
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
# 左耳进右耳出 — 序列内记忆，序列间清零
# ============================================================

class ZuoErJinYouErChu(nn.Module):
    """
    左耳进右耳出 — 八卦架构专属记忆模块
    序列内：维持记忆，保证前后文连贯
    序列间：处理完立即清零，用完即删，天然杜绝跨序列过拟合
    """
    def __init__(self, dim: int, memory_slots: int = 16):
        super().__init__()
        self.dim = dim
        self.memory_slots = memory_slots
        self.write_gate = nn.Sequential(nn.Linear(dim, memory_slots), nn.Sigmoid())
        self.read_gate = nn.Sequential(nn.Linear(dim, memory_slots), nn.Sigmoid())
        self.memory_transform = nn.Linear(dim, dim)
        self.memory_proj = nn.Linear(memory_slots, dim)
        self.output_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh())
        self.memory_state = None

    def reset(self):
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
            self.memory_state = (
                (1 - write_weight) * self.memory_state +
                write_weight * self.memory_transform(x_t).mean(dim=-1, keepdim=True).expand_as(write_weight)
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
# 八卦架构完整模型（八个模块全齐）
# ============================================================

class BaGuaModel(nn.Module):
    """
    八卦架构 — 八模块完整版
    1. 动态八卦阵
    2. 卦象对冲
    3. 淘汰审核（含原卦象编码）
    4. 淘汰低效机制
    5. 算力缓冲区
    6. 去中心化自运算
    7. 左耳进右耳出
    8. 九州编码
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 4,
        max_len: int = 128,
        polarity_dim: int = 32,
        dropout: float = 0.1,
        survival_threshold: float = 0.3,
    ):
        super().__init__()
        assert dim % 8 == 0
        self.dim = dim
        self.head_dim = dim // 8
        self.num_layers = num_layers

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

        # 淘汰审核（合并版）
        self.shenhe_layers = nn.ModuleList([
            nn.ModuleList([TaoTaiShenHe(self.head_dim) for _ in range(8)])
            for _ in range(num_layers)
        ])

        # 淘汰低效机制
        self.taoTai = TaoTaiDiXiaoJiZhi(survival_threshold)

        # 算力缓冲区
        self.buffer = SuanLiHuanChongQu(dim)

        # 九州编码
        self.jiuzhou = JiuZhouBianMa(dim=dim, num_zones=8, max_len=max_len)

        # 左耳进右耳出
        self.zuoer = ZuoErJinYouErChu(dim=dim, memory_slots=16)

        # 分类头（八卦架构专用，直接线性映射）
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # 去中心化自运算
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

        prev = x
        total_survival = 0.0

        for layer_idx in range(self.num_layers):
            bagua_out, impedance = self.bagua_layers[layer_idx](x)

            input_heads = [x[:, :, i*self.head_dim:(i+1)*self.head_dim] for i in range(8)]
            output_heads_raw = [bagua_out[:, :, i*self.head_dim:(i+1)*self.head_dim] for i in range(8)]

            # 九州编码：注入位置感知
            output_heads = self.jiuzhou(output_heads_raw)

            # 淘汰审核：评估每个卦象的价值
            scores = [
                self.shenhe_layers[layer_idx][i].evaluate(output_heads[i], input_heads[i])
                for i in range(8)
            ]

            # 淘汰低效机制
            masked = self.taoTai.eliminate(output_heads, scores)
            merged = torch.cat(masked, dim=-1)

            # 算力缓冲区
            smoothed = self.buffer(merged, prev)
            prev = smoothed.detach()

            x = self.ff_layers[layer_idx](smoothed)

            avg_survival = sum(s.item() for s in scores) / 8
            total_survival += avg_survival

        # 去中心化自运算
        avg_s = total_survival / self.num_layers
        delta = (1.0 - avg_s) * 0.05
        self.logic_pressure = max(0.5, min(2.0, self.logic_pressure + delta))

        # 左耳进右耳出
        x = self.zuoer(x)
        self.zuoer.reset()

        pooled = x.mean(dim=1)
        return self.classifier(pooled)

    def honesty_loss(self, logits):
        return self.shenhe_layers[0][0].honesty_loss(logits)


# ============================================================
# BERT-like 对比模型
# ============================================================

class BERTlikeModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_classes: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_len + 1, dim)
        self.embed_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.embed_dropout(self.embedding(input_ids) + self.pos_embedding(pos_ids))
        x = self.encoder(x)
        return self.classifier(x.mean(dim=1))


# ============================================================
# 数据集一：AG News 新闻主题分类
# ============================================================

def load_agnews(max_len: int = 128, batch_size: int = 32):
    print("正在加载 AG News 数据集...")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    class AGNewsDataset(Dataset):
        def __init__(self, data, tokenizer, max_len):
            self.data = list(data)
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            encoded = self.tokenizer(
                item['text'], max_length=self.max_len,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            return encoded['input_ids'].squeeze(0), torch.tensor(item['label'], dtype=torch.long)

    train_ds = AGNewsDataset(dataset['train'], tokenizer, max_len)
    val_ds = AGNewsDataset(dataset['test'], tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"训练集：{len(train_ds)} 条 | 验证集：{len(val_ds)} 条 | 类别：4（科技/体育/商业/世界）")
    return train_loader, val_loader, tokenizer.vocab_size


# ============================================================
# 数据集二：WikiText-103 文本连贯性判断
# 给一段上文，判断哪一句是真正的下一句（二分类：真/假）
# 贴近大模型的实际文字理解能力
# ============================================================

def load_coherence(max_len: int = 128, batch_size: int = 32):
    print("正在加载 WikiText-103 连贯性数据集...")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 提取有效段落
    def get_paragraphs(split):
        paragraphs = []
        current = []
        for item in dataset[split]:
            text = item['text'].strip()
            if text == '':
                if len(current) >= 3:
                    paragraphs.append(current)
                current = []
            elif not text.startswith('='):
                current.append(text)
        return paragraphs

    train_paras = get_paragraphs('train')
    val_paras = get_paragraphs('validation')

    class CoherenceDataset(Dataset):
        """
        正样本：上文 + 真实下一句（标签=1）
        负样本：上文 + 随机句子（标签=0）
        考验模型对文本连贯性的理解
        """
        def __init__(self, paragraphs, tokenizer, max_len, size=20000):
            self.samples = []
            self.tokenizer = tokenizer
            self.max_len = max_len
            all_sentences = [s for para in paragraphs for s in para]
            random.seed(42)
            para_list = [p for p in paragraphs if len(p) >= 2]
            for _ in range(size):
                para = random.choice(para_list)
                idx = random.randint(0, len(para) - 2)
                context = ' '.join(para[max(0, idx-2):idx+1])
                if random.random() > 0.5:
                    next_sent = para[idx + 1]
                    label = 1
                else:
                    next_sent = random.choice(all_sentences)
                    label = 0
                self.samples.append((context, next_sent, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            context, next_sent, label = self.samples[idx]
            encoded = self.tokenizer(
                context, next_sent,
                max_length=self.max_len, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            return encoded['input_ids'].squeeze(0), torch.tensor(label, dtype=torch.long)

    train_ds = CoherenceDataset(train_paras, tokenizer, max_len, size=30000)
    val_ds = CoherenceDataset(val_paras, tokenizer, max_len, size=3000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"训练集：{len(train_ds)} 条 | 验证集：{len(val_ds)} 条 | 类别：2（连贯/不连贯）")
    return train_loader, val_loader, tokenizer.vocab_size


# ============================================================
# 训练与验证
# ============================================================

def evaluate(model, val_loader, device):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            total_loss += loss_fn(logits, labels).item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total, total_loss / len(val_loader)


def run_task(
    task_name: str,
    train_loader,
    val_loader,
    vocab_size: int,
    num_classes: int,
    dim: int,
    num_layers: int,
    num_epochs: int = 15,
    learning_rate: float = 3e-4,
    device: str = 'cuda',
    scale_name: str = '小参数',
):
    print(f"\n{'='*60}")
    print(f"  任务：{task_name} | {scale_name}规模 | dim={dim}, layers={num_layers}")
    print(f"{'='*60}")

    bagua = BaGuaModel(
        vocab_size=vocab_size, dim=dim, num_layers=num_layers,
        num_classes=num_classes, max_len=128, polarity_dim=32, dropout=0.1
    ).to(device)

    bert = BERTlikeModel(
        vocab_size=vocab_size, dim=dim, num_layers=num_layers,
        num_heads=8, num_classes=num_classes, max_len=128, dropout=0.1
    ).to(device)

    bagua_params = sum(p.numel() for p in bagua.parameters())
    bert_params = sum(p.numel() for p in bert.parameters())
    print(f"参数量 — 八卦：{bagua_params/1e6:.1f}M | BERT-like：{bert_params/1e6:.1f}M")

    opt_b = optim.AdamW(bagua.parameters(), lr=learning_rate, weight_decay=0.01)
    opt_t = optim.AdamW(bert.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = num_epochs * len(train_loader)
    warmup = total_steps // 10

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup))))

    sched_b = optim.lr_scheduler.LambdaLR(opt_b, lr_lambda)
    sched_t = optim.lr_scheduler.LambdaLR(opt_t, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    bagua_accs, bert_accs, epochs_list = [], [], []

    print(f"\n{'Epoch':>6} | {'八卦Acc':>8} | {'BERTAcc':>8} | {'八卦Loss':>9} | {'BERTLoss':>9}")
    print("-" * 55)

    for epoch in range(num_epochs):
        for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            input_ids, labels = input_ids.to(device), labels.to(device)

            opt_b.zero_grad()
            logits_b = bagua(input_ids)
            loss_b = loss_fn(logits_b, labels) + bagua.honesty_loss(logits_b)
            loss_b.backward()
            torch.nn.utils.clip_grad_norm_(bagua.parameters(), 1.0)
            opt_b.step()
            sched_b.step()

            opt_t.zero_grad()
            logits_t = bert(input_ids)
            loss_fn(logits_t, labels).backward()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
            opt_t.step()
            sched_t.step()

        b_acc, b_loss = evaluate(bagua, val_loader, device)
        t_acc, t_loss = evaluate(bert, val_loader, device)
        bagua_accs.append(b_acc)
        bert_accs.append(t_acc)
        epochs_list.append(epoch + 1)
        print(f"{epoch+1:>6} | {b_acc:>8.4f} | {t_acc:>8.4f} | {b_loss:>9.4f} | {t_loss:>9.4f}")

    print("-" * 55)
    winner = '八卦架构更优 ✓' if bagua_accs[-1] > bert_accs[-1] else 'BERT-like更优'
    print(f"最终：八卦 {bagua_accs[-1]*100:.2f}% | BERT {bert_accs[-1]*100:.2f}% | {winner}")

    return {
        'task': task_name, 'scale': scale_name,
        'bagua_accs': bagua_accs, 'bert_accs': bert_accs,
        'epochs': epochs_list,
        'bagua_final': bagua_accs[-1], 'bert_final': bert_accs[-1],
        'bagua_params': bagua_params, 'bert_params': bert_params,
    }


# ============================================================
# 绘图
# ============================================================

def plot_results(all_results):
    n = len(all_results)
    fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 6))
    fig.suptitle('BaGua Architecture vs BERT-like\nMulti-Task Comprehensive Evaluation',
                 fontsize=13, fontweight='bold')

    colors_b, colors_t = '#E84545', '#2B4EFF'

    for i, r in enumerate(all_results):
        ax = axes[i]
        ax.plot(r['epochs'], [a*100 for a in r['bagua_accs']], 'o-',
                label='BaGua', color=colors_b, linewidth=2)
        ax.plot(r['epochs'], [a*100 for a in r['bert_accs']], 's--',
                label='BERT-like', color=colors_t, linewidth=2)
        ax.set_title(f"{r['task']}\n({r['scale']})")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(40, 100)

    # 最终准确率汇总柱状图
    ax = axes[-1]
    x = np.arange(len(all_results))
    w = 0.35
    bagua_finals = [r['bagua_final']*100 for r in all_results]
    bert_finals = [r['bert_final']*100 for r in all_results]
    bars1 = ax.bar(x - w/2, bagua_finals, w, label='BaGua', color=colors_b, alpha=0.85)
    bars2 = ax.bar(x + w/2, bert_finals, w, label='BERT-like', color=colors_t, alpha=0.85)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_title('Final Accuracy\nAll Tasks')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['task']}\n{r['scale']}" for r in all_results], fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(40, 100)

    plt.tight_layout()
    plt.savefig('bagua_multitask_results.png', dpi=150, bbox_inches='tight')
    print("\n结果图表已保存：bagua_multitask_results.png")
    plt.show()


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"运行设备：{device}")
    if device == 'cuda':
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    all_results = []

    # ── 任务一：AG News 新闻主题分类 ──
    # 全局语义理解，八卦架构主场，小参数20轮
    train_loader, val_loader, vocab_size = load_agnews(max_len=128, batch_size=32)
    result = run_task(
        task_name='AG News 主题分类',
        train_loader=train_loader, val_loader=val_loader,
        vocab_size=vocab_size, num_classes=4,
        dim=256, num_layers=4,
        num_epochs=20, learning_rate=3e-4,
        device=device, scale_name='小参数',
    )
    all_results.append(result)

    # ── 任务二：WikiText-103 文本连贯性判断 ──
    # 贴近大模型实际能力，左耳进右耳出记忆模块主场，小参数20轮
    train_loader, val_loader, vocab_size = load_coherence(max_len=128, batch_size=32)
    result = run_task(
        task_name='文本连贯性判断',
        train_loader=train_loader, val_loader=val_loader,
        vocab_size=vocab_size, num_classes=2,
        dim=256, num_layers=4,
        num_epochs=20, learning_rate=3e-4,
        device=device, scale_name='小参数',
    )
    all_results.append(result)

    # 汇总绘图
    plot_results(all_results)

    print("\n" + "="*60)
    print("  八卦架构多任务综合测试完成")
    for r in all_results:
        winner = '八卦✓' if r['bagua_final'] > r['bert_final'] else 'BERT'
        print(f"  {r['task']} {r['scale']}：八卦 {r['bagua_final']*100:.2f}% | BERT {r['bert_final']*100:.2f}% | {winner}")
    print("="*60)

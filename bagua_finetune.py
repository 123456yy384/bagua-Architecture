"""
八卦架构 (BaGua Architecture) — 指令微调脚本
============================================
作者：阳恩硕 (Yang Enshuo)
用途：在预训练模型基础上进行指令微调
      让模型能听懂"请翻译"、"请总结"等指令
      从续写模型升级为对话模型

前置条件：
    已有 best_model.pt（预训练完成）

运行方式：
    python bagua_finetune.py
"""

import os
import sys
import json
import math
import time
import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# 路径检测
# ============================================================

def get_token_dir():
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
    print("未找到bagua_tokens目录")
    sys.exit(1)

TOKEN_DIR = get_token_dir()
META = json.loads((TOKEN_DIR / "meta.json").read_text(encoding='utf-8'))
VOCAB_SIZE = META["vocab_size"]
PRETRAINED_MODEL = TOKEN_DIR / "checkpoints" / "best_model.pt"
TOKENIZER_PATH = "C:/Users/Administrator/Desktop/bagua_data/tokenizer"

print(f"预训练模型：{PRETRAINED_MODEL}")
print(f"词表大小：{VOCAB_SIZE}")


# ============================================================
# 完整架构（和训练脚本完全一致，不能改动）
# ============================================================

class GuaXiangDuiChong(nn.Module):
    def __init__(self, head_dim, num_heads=8, polarity_dim=32):
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
        pv = [self.polarity_generators[i](heads[i].mean(dim=1)) for i in range(self.num_heads)]
        pm = F.normalize(torch.stack(pv, dim=1), dim=-1)
        dp = torch.bmm(pm, pm.transpose(1, 2))
        imp = self.impedance_modulator(dp.reshape(-1, 1)).reshape(B, 8, 8)
        return imp * (1.0 - torch.eye(8, device=imp.device).unsqueeze(0))


class JiuZhouBianMa(nn.Module):
    def __init__(self, dim, num_zones=8, max_len=512):
        super().__init__()
        assert dim % 8 == 0
        self.head_dim = dim // 8
        self.num_zones = num_zones
        self.zone_proj = nn.ModuleList([
            nn.Linear(self.head_dim + 2, self.head_dim, bias=False) for _ in range(8)
        ])
        self.inject_gate = nn.Sequential(nn.Linear(self.head_dim, 1), nn.Sigmoid())

    def _pos_code(self, i, S, device):
        tc = i / 7.0
        pos = torch.arange(S, device=device).float() / max(S-1, 1)
        zs = max(S / self.num_zones, 1)
        zr = (torch.arange(S, device=device).float() % zs) / zs
        return torch.stack([pos*0.5 + torch.full((S,), tc, device=device)*0.5, zr], dim=-1)

    def forward(self, heads):
        B, S, _ = heads[0].shape
        device = heads[0].device
        out = []
        for i, h in enumerate(heads):
            pc = self._pos_code(i, S, device).unsqueeze(0).expand(B, -1, -1)
            he = self.zone_proj[i](torch.cat([h, pc], dim=-1))
            g = self.inject_gate(h)
            out.append(h * (1-g) + he * g)
        return out


class DongTaiBaGuaZhen(nn.Module):
    def __init__(self, dim, polarity_dim=32, dropout=0.1):
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

    def forward(self, x, causal=True):
        residual = x
        B, S, D = x.shape
        device = x.device
        heads = [self.trigram_projections[i](x) * torch.cos(self.resonance_freqs[i] * math.pi)
                 for i in range(8)]
        impedance = self.guaxiang_duichong(heads)
        if causal and S > 1:
            fm = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        output_heads = []
        for i in range(8):
            h_i = heads[i].clone()
            for j in range(8):
                if i != j:
                    imp_ij = impedance[:, i, j].unsqueeze(1).unsqueeze(2)
                    transfer = heads[j] / (1.0 + imp_ij) * 0.1
                    if causal and S > 1:
                        t3d = transfer.unsqueeze(2).expand(B, S, S, self.head_dim)
                        transfer = t3d.masked_fill(fm.unsqueeze(0).unsqueeze(-1), 0.0).mean(dim=2)
                    h_i = h_i + transfer
            output_heads.append(h_i)
        merged = torch.cat(output_heads, dim=-1)
        return self.norm(self.dropout(self.output_proj(merged)) + residual), impedance


class TaoTaiShenHe(nn.Module):
    def __init__(self, head_dim, beta=0.01):
        super().__init__()
        self.beta = beta
        self.density_scorer = nn.Sequential(
            nn.Linear(head_dim, head_dim//2), nn.GELU(),
            nn.Linear(head_dim//2, 1), nn.Sigmoid()
        )

    def evaluate(self, ho, hi):
        return self.density_scorer(ho).mean() * torch.sigmoid((ho-hi).abs().mean() * 10.0 - 1.0)

    def honesty_loss(self, logits):
        p = F.softmax(logits[:64].detach(), dim=-1)
        return self.beta * (-torch.sum(p * torch.log(p + 1e-8), dim=-1).mean())


class TaoTaiDiXiaoJiZhi:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def eliminate(self, heads, scores):
        return [h * 0.0 if scores[i].item() < self.threshold else h for i, h in enumerate(heads)]


class SuanLiHuanChongQu(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.charge_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.discharge_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, cur, prev):
        if cur.shape != prev.shape:
            return cur
        return cur + self.discharge_gate(cur) * (self.charge_gate(prev) * prev)


class ZuoErJinYouErChu(nn.Module):
    def __init__(self, dim, memory_slots=32):
        super().__init__()
        self.memory_slots = memory_slots
        self.write_gate = nn.Sequential(nn.Linear(dim, memory_slots), nn.Sigmoid())
        self.read_gate = nn.Sequential(nn.Linear(dim, memory_slots), nn.Sigmoid())
        self.memory_transform = nn.Linear(dim, memory_slots)
        self.memory_proj = nn.Linear(memory_slots, dim)
        self.output_gate = nn.Sequential(nn.Linear(dim*2, dim), nn.Tanh())
        self.memory_state = None

    def reset(self):
        self.memory_state = None

    def forward(self, x):
        B, S, D = x.shape
        device = x.device
        if self.memory_state is None or self.memory_state.shape[0] != B:
            self.memory_state = torch.zeros(B, self.memory_slots, device=device)
        outputs = []
        for t in range(S):
            xt = x[:, t, :]
            mv = self.memory_proj(self.read_gate(xt) * self.memory_state)
            fused = self.output_gate(torch.cat([xt, mv], dim=-1))
            ww = self.write_gate(xt)
            self.memory_state = ((1-ww)*self.memory_state + ww*self.memory_transform(xt)).detach()
            outputs.append(fused.unsqueeze(1))
        return torch.cat(outputs, dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim*ff_mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim*ff_mult, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class RenWuZiWoGanZhi(nn.Module):
    SCENE_PRESETS = {
        'text_generation': 0.95, 'code_generation': 0.95, 'continuation': 0.90,
        'translation': 0.85, 'summarization': 0.80, 'creative_writing': 0.95,
        'poetry': 0.90, 'classification': 0.05, 'sentiment': 0.05, 'ner': 0.10,
        'relation_extract': 0.10, 'similarity': 0.05, 'retrieval': 0.05,
        'annotation': 0.05, 'fact_check': 0.10, 'content_moderation': 0.05,
        'qa': 0.40, 'dialogue': 0.70, 'reading_comp': 0.35, 'rag': 0.45,
        'instruction': 0.60, 'reasoning': 0.50, 'multi_turn': 0.65,
    }

    def __init__(self, dim):
        super().__init__()
        n = len(self.SCENE_PRESETS)
        self.scene_matcher = nn.Sequential(
            nn.Linear(dim, dim//2), nn.GELU(), nn.Linear(dim//2, n), nn.Softmax(dim=-1)
        )
        self.register_buffer('scene_causal_weights',
            torch.tensor(list(self.SCENE_PRESETS.values()), dtype=torch.float32))

    def forward(self, x):
        return torch.matmul(self.scene_matcher(x[:, 0, :]), self.scene_causal_weights)


class BaGuaLLM(nn.Module):
    def __init__(self, vocab_size, dim=768, num_layers=12, max_len=256,
                 polarity_dim=32, dropout=0.1, survival_threshold=0.3):
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
            DongTaiBaGuaZhen(dim, polarity_dim, dropout) for _ in range(num_layers)
        ])
        self.ff_layers = nn.ModuleList([FeedForward(dim, 4, dropout) for _ in range(num_layers)])
        self.shenhe_layers = nn.ModuleList([
            nn.ModuleList([TaoTaiShenHe(self.head_dim) for _ in range(8)])
            for _ in range(num_layers)
        ])
        self.taoTai = TaoTaiDiXiaoJiZhi(survival_threshold)
        self.buffer = SuanLiHuanChongQu(dim)
        self.jiuzhou = JiuZhouBianMa(dim, 8, max_len)
        self.tsa = RenWuZiWoGanZhi(dim)
        self.zuoer = ZuoErJinYouErChu(dim, 32)
        self.lm_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, vocab_size, bias=False))
        self.logic_pressure = 1.0

    def forward(self, input_ids):
        B, S = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.embed_dropout(self.embedding(input_ids) + self.pos_embedding(pos_ids))
        use_causal = (self.tsa(x).mean().item() > 0.5)
        prev = x
        total_survival = 0.0
        for layer_idx in range(self.num_layers):
            bagua_out, impedance = self.bagua_layers[layer_idx](x, causal=use_causal)
            input_heads = [x[:, :, i*self.head_dim:(i+1)*self.head_dim] for i in range(8)]
            output_heads_raw = [bagua_out[:, :, i*self.head_dim:(i+1)*self.head_dim] for i in range(8)]
            output_heads = self.jiuzhou(output_heads_raw)
            scores = [self.shenhe_layers[layer_idx][i].evaluate(output_heads[i], input_heads[i])
                      for i in range(8)]
            masked = self.taoTai.eliminate(output_heads, scores)
            merged = torch.cat(masked, dim=-1)
            smoothed = self.buffer(merged, prev)
            prev = smoothed.detach()
            x = self.ff_layers[layer_idx](smoothed)
            total_survival += sum(s.item() for s in scores) / 8
        avg_s = total_survival / self.num_layers
        self.logic_pressure = max(0.5, min(2.0, self.logic_pressure + 0.05 * (1.0 - avg_s)))
        x = self.zuoer(x)
        self.zuoer.reset()
        return self.lm_head(x)


# ============================================================
# 指令微调数据集
# 格式：指令 + 输入 + 输出，用特殊分隔符拼接
# 模型学会：看到指令格式就执行指令，而不是随机续写
# ============================================================

# 内置指令微调数据（中英双语，覆盖常见任务）
INSTRUCTION_DATA = [
    # 翻译任务
    {"instruction": "请将以下英文翻译成中文", "input": "Hello, how are you?", "output": "你好，你怎么样？"},
    {"instruction": "请将以下中文翻译成英文", "input": "今天天气很好", "output": "The weather is very nice today."},
    {"instruction": "Translate to Chinese", "input": "Artificial intelligence is changing the world.", "output": "人工智能正在改变世界。"},
    {"instruction": "Translate to English", "input": "八卦架构是一个创新的神经网络架构", "output": "BaGua Architecture is an innovative neural network architecture."},

    # 问答任务
    {"instruction": "请回答以下问题", "input": "什么是人工智能？", "output": "人工智能是让计算机模拟人类智能行为的技术，包括学习、推理和解决问题。"},
    {"instruction": "请回答以下问题", "input": "中国的首都是哪里？", "output": "中国的首都是北京。"},
    {"instruction": "Answer the question", "input": "What is machine learning?", "output": "Machine learning is a type of AI that allows computers to learn from data without being explicitly programmed."},
    {"instruction": "Answer the question", "input": "What is the capital of France?", "output": "The capital of France is Paris."},

    # 摘要任务
    {"instruction": "请用一句话总结以下内容", "input": "八卦架构是基于易经八卦图设计的神经网络架构，通过极性驱动的动态阻抗机制，让网络拓扑在每次前向传播时根据输入数据重新生长，从根本上解决了传统Transformer固定拓扑的局限性。", "output": "八卦架构是一种基于易经的动态拓扑神经网络，每次根据输入实时重构网络结构。"},
    {"instruction": "Summarize in one sentence", "input": "BaGua Architecture is a neural network inspired by the Eight Trigrams of Chinese philosophy, using polarity-driven dynamic impedance to reconstruct the network topology on every forward pass.", "output": "BaGua Architecture dynamically reconstructs its topology using polarity interactions inspired by the Eight Trigrams."},

    # 续写任务
    {"instruction": "请续写以下故事", "input": "从前有一个小村庄，村子里住着一位年轻的发明家", "output": "他每天都在工作室里研究新的发明，希望有一天能造出改变世界的东西。村民们都说他是个怪人，但他从不在意，只是默默地坚持着自己的梦想。"},
    {"instruction": "Continue the story", "input": "Once upon a time, there was a young engineer who", "output": "dreamed of building something that would change the world. Every day, he worked tirelessly in his small workshop, experimenting with new ideas and learning from his failures."},

    # 解释任务
    {"instruction": "请用简单的语言解释", "input": "什么是神经网络？", "output": "神经网络是模仿人脑工作方式的计算机程序，由很多相互连接的节点组成，通过学习大量数据来解决问题。"},
    {"instruction": "Explain simply", "input": "What is deep learning?", "output": "Deep learning is a type of AI that uses many layers of neural networks to learn complex patterns from data, similar to how humans learn."},

    # 对话任务
    {"instruction": "请用友好的方式回复", "input": "我今天心情不好", "output": "很抱歉听到这个消息，希望你能好起来。有时候心情不好是正常的，给自己一些时间和空间。有什么我能帮你的吗？"},
    {"instruction": "Respond helpfully", "input": "I'm feeling stressed about my work", "output": "I understand how stressful work can be sometimes. It's important to take breaks and not overwhelm yourself. Is there anything specific that's causing you stress?"},

    # 代码任务
    {"instruction": "请写一个Python函数", "input": "计算两个数的和", "output": "def add(a, b):\n    return a + b"},
    {"instruction": "Write a Python function", "input": "Check if a number is even", "output": "def is_even(n):\n    return n % 2 == 0"},

    # 创意写作
    {"instruction": "请写一首关于的短诗", "input": "春天", "output": "春风轻抚大地，\n花朵悄然绽放，\n万物复苏之时，\n生命重焕光芒。"},
    {"instruction": "Write a short poem about", "input": "the ocean", "output": "Waves crash upon the shore,\nEndless blue stretches far,\nThe ocean's mighty roar,\nBeneath the morning star."},
]


class InstructionDataset(Dataset):
    """
    指令微调数据集
    格式：### 指令：{instruction}\n### 输入：{input}\n### 输出：{output}
    模型学会：看到这种格式就执行指令
    """
    def __init__(self, tokenizer, seq_len=256, augment_times=50):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.samples = []

        # 把内置数据扩充augment_times倍（通过随机顺序打乱）
        data = INSTRUCTION_DATA * augment_times
        random.seed(42)
        random.shuffle(data)

        for item in data:
            # 拼接成训练格式
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
                pad_id = tokenizer.pad_token_id or 0
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

        print(f"指令微调样本数：{len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# 指令微调训练
# ============================================================

def finetune():
    print("=" * 60)
    print("  八卦架构 LLM — 指令微调")
    print("  让模型从续写升级为能听懂指令的对话模型")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n运行设备：{device}")
    if device == 'cuda':
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 加载分词器
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    print(f"分词器加载完成，词表大小：{tokenizer.vocab_size}")

    # 加载预训练模型
    print(f"\n加载预训练模型：{PRETRAINED_MODEL}")
    model = BaGuaLLM(
        vocab_size=VOCAB_SIZE, dim=768, num_layers=12,
        max_len=256, polarity_dim=32, dropout=0.1,
    ).to(device)

    state_dict = torch.load(PRETRAINED_MODEL, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("预训练模型加载完成")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params/1e6:.1f}M")

    # 指令微调配置（学习率要比预训练小很多，避免灾难性遗忘）
    CONFIG = {
        'num_epochs': 10,
        'batch_size': 4,
        'learning_rate': 5e-5,   # 比预训练小10倍
        'grad_clip': 0.5,
        'warmup_steps': 100,
        'log_every': 20,
    }

    dataset = InstructionDataset(tokenizer, seq_len=256, augment_times=100)
    loader = DataLoader(
        dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=2, pin_memory=True
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    total_steps = CONFIG['num_epochs'] * len(loader)
    warmup = CONFIG['warmup_steps']

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup))))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    loss_fn = nn.CrossEntropyLoss()

    save_dir = TOKEN_DIR / "checkpoints"
    save_dir.mkdir(exist_ok=True)

    best_loss = float('inf')
    step = 0

    print(f"\n开始指令微调，共{CONFIG['num_epochs']}轮")
    print(f"{'Step':>6} | {'Loss':>8} | {'LR':>10}")
    print("-" * 30)

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0.0

        for input_ids, target_ids in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids)
                    B, S, V = logits.shape
                    loss = loss_fn(logits.reshape(-1, V), target_ids.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids)
                B, S, V = logits.shape
                loss = loss_fn(logits.reshape(-1, V), target_ids.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            step += 1

            if step % CONFIG['log_every'] == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"{step:>6} | {loss.item():>8.4f} | {lr:>10.2e}")

        avg_loss = epoch_loss / len(loader)
        print(f"\nEpoch {epoch+1} 完成，平均Loss：{avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_dir / "instruct_model.pt")
            print(f"✓ 最佳指令模型已保存，Loss={best_loss:.4f}")

    print(f"\n指令微调完成，最佳Loss：{best_loss:.4f}")
    print(f"模型保存至：{save_dir / 'instruct_model.pt'}")
    print("\n现在可以运行 bagua_chat.py 测试对话效果")
    print("（将 best_model.pt 替换为 instruct_model.pt）")


if __name__ == "__main__":
    finetune()

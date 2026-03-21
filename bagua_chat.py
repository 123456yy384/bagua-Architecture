"""
八卦架构 (BaGua Architecture) — 对话推理脚本
============================================
作者：阳恩硕 (Yang Enshuo)
用途：加载训练好的模型，进行交互式对话测试

运行方式：
    python bagua_chat.py

模型文件位置：
    C:/Users/Administrator/Desktop/bagua_tokens/checkpoints/best_model.pt
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 自动检测路径
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
            return p
    print("未找到 bagua_tokens 目录")
    sys.exit(1)

TOKEN_DIR = get_token_dir()
META = json.loads((TOKEN_DIR / "meta.json").read_text(encoding='utf-8'))
VOCAB_SIZE = META["vocab_size"]
MODEL_PATH = TOKEN_DIR / "checkpoints" / "best_model.pt"

print(f"模型路径：{MODEL_PATH}")
print(f"词表大小：{VOCAB_SIZE}")


# ============================================================
# 完整架构定义（和训练脚本完全一致）
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
        pm = torch.stack(pv, dim=1)
        pn = F.normalize(pm, dim=-1)
        dp = torch.bmm(pn, pn.transpose(1, 2))
        imp = self.impedance_modulator(dp.reshape(-1, 1)).reshape(B, 8, 8)
        mask = 1.0 - torch.eye(8, device=imp.device).unsqueeze(0)
        return imp * mask


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
        tb = torch.full((S,), tc, device=device)
        return torch.stack([pos*0.5 + tb*0.5, zr], dim=-1)

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
        ld = self.density_scorer(ho).mean()
        cd = (ho - hi).abs().mean()
        cs = torch.sigmoid(cd * 10.0 - 1.0)
        return ld * cs

    def honesty_loss(self, logits):
        p = F.softmax(logits[:64].detach(), dim=-1)
        return self.beta * (-torch.sum(p * torch.log(p + 1e-8), dim=-1).mean())


class TaoTaiDiXiaoJiZhi:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def eliminate(self, heads, scores):
        return [h * 0.0 if scores[i].item() < self.threshold else h
                for i, h in enumerate(heads)]


class SuanLiHuanChongQu(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cg = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.dg = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, cur, prev):
        if cur.shape != prev.shape:
            return cur
        return cur + self.dg(cur) * (self.cg(prev) * prev)


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
            rw = self.read_gate(xt)
            mv = self.memory_proj(rw * self.memory_state)
            fused = self.output_gate(torch.cat([xt, mv], dim=-1))
            ww = self.write_gate(xt)
            nm = self.memory_transform(xt)
            self.memory_state = ((1-ww)*self.memory_state + ww*nm).detach()
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

        causal_strength = self.tsa(x)
        use_causal = (causal_strength.mean().item() > 0.5)

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

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200, temperature=0.8,
                 top_k=50, top_p=0.9, device='cuda'):
        self.eval()
        self.zuoer.reset()
        generated = input_ids.clone().to(device)

        for _ in range(max_new_tokens):
            # 超过最大长度时只取最后256个token
            ctx = generated[:, -256:] if generated.shape[1] > 256 else generated
            logits = self(ctx)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k过滤
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, -1:]] = float('-inf')

            # Top-p过滤
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                next_logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() in [102, 0, 3]:
                break

        self.zuoer.reset()
        return generated


# ============================================================
# 加载模型
# ============================================================

def load_model(device):
    print("正在加载模型...")
    model = BaGuaLLM(
        vocab_size=VOCAB_SIZE,
        dim=768,
        num_layers=12,
        max_len=256,
        polarity_dim=32,
        dropout=0.0,  # 推理时关闭dropout
    ).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型加载完成，参数量：{total_params/1e6:.1f}M")
    return model


def load_tokenizer():
    from transformers import BertTokenizerFast
    vocab_file = TOKEN_DIR.parent / "tokenizer" / "vocab.txt"
    if not vocab_file.exists():
        # 尝试E盘
        vocab_file = Path("E:/bagua_data/tokenizer/vocab.txt")
    tokenizer = BertTokenizerFast(
        vocab_file=str(vocab_file),
        do_lower_case=True
    )
    return tokenizer


# ============================================================
# 交互式对话
# ============================================================

def chat():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"运行设备：{device}")

    model = load_model(device)
    tokenizer = load_tokenizer()

    print("\n" + "="*55)
    print("  八卦架构 LLM — 对话测试")
    print("  输入文字后按回车，模型接着生成")
    print("  输入 'quit' 退出")
    print("  输入 'temp:0.5' 调整温度（越低越保守，越高越随机）")
    print("="*55 + "\n")

    temperature = 0.8
    max_tokens = 150

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出对话")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("退出对话")
            break

        if user_input.startswith('temp:'):
            try:
                temperature = float(user_input.split(':')[1])
                print(f"温度已设置为：{temperature}")
            except:
                print("格式错误，示例：temp:0.7")
            continue

        if user_input.startswith('max:'):
            try:
                max_tokens = int(user_input.split(':')[1])
                print(f"最大生成长度已设置为：{max_tokens}")
            except:
                print("格式错误，示例：max:200")
            continue

        # 分词并生成
        input_ids = tokenizer.encode(user_input, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                device=device,
            )

        # 只显示新生成的部分
        new_ids = output_ids[0][len(input_ids):].tolist()
        response = tokenizer.decode(new_ids, skip_special_tokens=True)

        print(f"八卦：{response}\n")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    chat()

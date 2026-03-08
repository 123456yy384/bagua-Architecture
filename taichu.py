"""
太初 (TaiChu) 架构 — 官方命名版
================================
创始人：[你的名字]

模块命名体系（官方）：
    动态八卦阵     — 信息拓扑解构引擎
    卦象对冲       — 全动态阻抗驱动器（动态八卦阵的核心）
    卦象编码       — 上帝视角价值评估
    淘汰低效机制   — 定点清除低价值路径
    算力缓冲区     — 爆破后信号平滑重分配
    淘汰审核       — 诚实度损失函数
    去中心化自运算 — 全局逻辑压强调节器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from typing import Dict, List, Tuple, Optional
import math
import time


# ============================================================
# 卦象对冲 — 动态八卦阵的核心驱动器
# 作用：实时计算八个卦象之间的极性阻抗矩阵
# 原理：每个卦象生成自己的极性向量，两两点积决定阻抗强弱
#       同极相斥（高阻抗）→ 信息阻断
#       异极相吸（低阻抗）→ 信息自由流动
# ============================================================

class GuaXiangDuiChong(nn.Module):
    """
    卦象对冲 — 太初架构的全动态阻抗驱动器

    每次前向传播，八个卦象根据当前输入数据各自生成极性向量，
    通过两两点积实时计算 8×8 阻抗矩阵。
    没有任何预设的固定结构，拓扑完全由数据决定。
    """

    def __init__(self, dim: int, num_heads: int = 8, polarity_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.polarity_dim = polarity_dim

        # 八个卦象各自独立的极性生成网络
        # 输出值域 (-1, 1)，正负代表极性方向
        self.polarity_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim // num_heads, polarity_dim),
                nn.Tanh()
            ) for _ in range(num_heads)
        ])

        # 阻抗调制器：将点积映射为最终阻抗值（可训练）
        # 使系统能自动学习最优的极性-阻抗映射关系
        self.impedance_modulator = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Softplus()  # 保证阻抗恒为正值
        )

    def compute_polarity_vectors(
        self, head_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        为八个卦象分别生成极性向量。
        输入：8个卦象输出，每个 [B, S, head_dim]
        输出：极性矩阵 [B, 8, polarity_dim]
        """
        polarity_vectors = []
        for i in range(self.num_heads):
            # 对序列取均值，提炼每个卦象的整体极性状态
            head_summary = head_outputs[i].mean(dim=1)          # [B, head_dim]
            polarity = self.polarity_generators[i](head_summary) # [B, polarity_dim]
            polarity_vectors.append(polarity)
        return torch.stack(polarity_vectors, dim=1)              # [B, 8, polarity_dim]

    def compute_dynamic_impedance(
        self, polarity_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        基于极性向量计算全动态阻抗矩阵。
        输入：极性矩阵 [B, 8, polarity_dim]
        输出：阻抗矩阵 [B, 8, 8]，每个batch、每个步骤完全独立
        """
        B = polarity_matrix.shape[0]

        # 归一化，使点积只反映方向，不受幅度干扰
        polarity_norm = F.normalize(polarity_matrix, dim=-1)     # [B, 8, polarity_dim]

        # 计算所有卦象对之间的点积 → [B, 8, 8]
        dot_products = torch.bmm(polarity_norm, polarity_norm.transpose(1, 2))

        # 通过调制器映射为阻抗值
        dot_flat = dot_products.reshape(-1, 1)
        impedance_flat = self.impedance_modulator(dot_flat)
        impedance = impedance_flat.reshape(B, 8, 8)

        # 对角线清零（卦象不与自身产生阻抗）
        mask = 1.0 - torch.eye(8, device=impedance.device).unsqueeze(0)
        return impedance * mask                                  # [B, 8, 8]


# ============================================================
# 动态八卦阵 — 太初架构的信息解构引擎
# 作用：将输入投影到八个卦象空间，通过卦象对冲实时重组拓扑
# ============================================================

class DongTaiBaGuaZhen(nn.Module):
    """
    动态八卦阵 — 太初架构的核心信息处理单元

    八个卦象各自提取输入的不同特征，
    由卦象对冲实时计算阻抗，控制卦象间的信息流动。
    每次前向传播，拓扑结构完全重新生长，没有任何固定连接。
    """

    TRIGRAMS = ['乾', '坤', '震', '巽', '坎', '离', '艮', '兑']

    def __init__(self, dim: int = 512, polarity_dim: int = 32):
        super().__init__()
        assert dim % 8 == 0, "dim必须能被8整除，对应八个卦象"
        self.dim = dim
        self.head_dim = dim // 8

        # 八个卦象的投影层（在__init__中统一注册，权重持久可训练）
        self.trigram_projections = nn.ModuleList([
            nn.Linear(dim, self.head_dim, bias=False) for _ in range(8)
        ])

        # 每个卦象的谐振频率（可训练）
        self.resonance_freqs = nn.Parameter(
            torch.randn(8, self.head_dim) * 0.01
        )

        # 卦象对冲（全动态阻抗驱动器）
        self.guaxiang_duichong = GuaXiangDuiChong(
            dim=dim, num_heads=8, polarity_dim=polarity_dim
        )

        # 输出归一化
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        动态八卦阵前向传播。
        输入：x [B, S, D]
        输出：output [B, S, D]，impedance [B, 8, 8]
        """
        B, S, D = x.shape

        # 第一步：投影到八个卦象空间，应用谐振频率调制
        heads = []
        for i in range(8):
            proj = self.trigram_projections[i](x)            # [B, S, head_dim]
            freq_mod = torch.cos(self.resonance_freqs[i] * math.pi)
            heads.append(proj * freq_mod)

        # 第二步：卦象对冲生成全动态阻抗矩阵
        polarity_matrix = self.guaxiang_duichong.compute_polarity_vectors(heads)
        impedance = self.guaxiang_duichong.compute_dynamic_impedance(polarity_matrix)

        # 第三步：通过阻抗矩阵实现卦象间信息流动
        # 异极相吸（低阻抗）→ 信息传递多
        # 同极相斥（高阻抗）→ 信息传递少
        output_heads = []
        for i in range(8):
            h_i = heads[i].clone()
            for j in range(8):
                if i != j:
                    imp_ij = impedance[:, i, j].unsqueeze(1).unsqueeze(2)
                    transfer = heads[j] / (1.0 + imp_ij)
                    h_i = h_i + transfer * 0.1
            output_heads.append(h_i)

        merged = torch.cat(output_heads, dim=-1)             # [B, S, D]
        return self.output_norm(merged), impedance


# ============================================================
# 卦象编码 — 上帝视角的价值评估机制
# 作用：在卦象自身不可感知的情况下，评估其逻辑密度和生存价值
# ============================================================

class GuaXiangBianMa(nn.Module):
    """
    卦象编码 — 太初架构的上帝视角审核器

    神经元只知道自己的输出，不知道自己的价值评分。
    卦象编码从系统层面评估每个卦象的贡献：
        逻辑密度  — 输出信息是否丰富
        因果稳健性 — 对输入是否做了真实的变换
    两者相乘得到生存分数，交给淘汰低效机制处理。
    """

    def __init__(self, head_dim: int):
        super().__init__()
        # 逻辑密度评估网络
        self.density_scorer = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2),
            nn.GELU(),
            nn.Linear(head_dim // 2, 1),
            nn.Sigmoid()  # 输出范围 (0, 1)
        )

    def evaluate(
        self,
        head_output: torch.Tensor,  # [B, S, head_dim]，卦象输出
        head_input: torch.Tensor,   # [B, S, head_dim]，对应输入
    ) -> Dict[str, torch.Tensor]:
        """
        评估卦象的逻辑价值，生成生存分数。
        因果稳健性 = 输出相对输入的变化量（变化越大说明贡献越真实）
        """
        # 逻辑密度：直接评估输出的信息丰富程度
        logical_density = self.density_scorer(head_output).mean()

        # 因果稳健性：输出与输入的差异越大，说明卦象在真实工作
        causal_delta = (head_output - head_input).abs().mean()
        causal_stability = torch.sigmoid(causal_delta * 10.0 - 1.0)

        # 生存分数 = 逻辑密度 × 因果稳健性
        survival_score = logical_density * causal_stability

        return {
            'logical_density': logical_density,
            'causal_stability': causal_stability,
            'survival_score': survival_score,
        }


# ============================================================
# 淘汰低效机制 — 定点清除低价值卦象路径
# 作用：根据卦象编码的生存分数，清除"苟活"路径
# ============================================================

class TaoTaiDiXiaoJiZhi:
    """
    淘汰低效机制 — 太初架构的黑暗森林法则执行者

    生存分数低于阈值的卦象，被判定为低效路径，
    其输出直接清零——不是随机丢弃，而是有依据的定点清除。
    被清除路径的算力由算力缓冲区接管和重分配。
    """

    def __init__(self, survival_threshold: float = 0.3):
        # 生存阈值：低于此分数的卦象被淘汰
        self.survival_threshold = survival_threshold
        self.elimination_log: List[int] = []  # 记录淘汰历史

    def identify_targets(
        self, evaluations: Dict[int, Dict[str, torch.Tensor]]
    ) -> List[int]:
        """识别需要淘汰的卦象索引"""
        targets = [
            idx for idx, eval_result in evaluations.items()
            if eval_result['survival_score'].item() < self.survival_threshold
        ]
        self.elimination_log.extend(targets)
        return targets

    def eliminate(
        self,
        head_outputs: List[torch.Tensor],
        targets: List[int],
    ) -> List[torch.Tensor]:
        """
        对被淘汰的卦象输出置零（乘以0保留梯度图，训练仍可进行）
        """
        return [
            h * 0.0 if i in targets else h
            for i, h in enumerate(head_outputs)
        ]


# ============================================================
# 算力缓冲区 — 淘汰后的信号平滑重分配
# 作用：防止路径突然清零导致信号不连续和梯度崩溃
# ============================================================

class SuanLiHuanChongQu(nn.Module):
    """
    算力缓冲区 — 太初架构的信号平滑器

    路径被淘汰后会产生信号跳变，算力缓冲区的作用是：
    将上一步的状态"充电"存入缓冲，
    在当前步"放电"注入输出，实现无缝的算力重分配。
    充电门和放电门均可训练，系统能自动学习最优的缓冲策略。
    """

    def __init__(self, dim: int):
        super().__init__()
        # 充电门：决定上一步有多少状态被存入缓冲
        self.charge_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        # 放电门：决定缓冲中有多少被注入当前输出
        self.discharge_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(
        self,
        current_output: torch.Tensor,  # [B, S, D]，当前步输出
        prev_output: torch.Tensor,     # [B, S, D]，上一步输出
    ) -> torch.Tensor:
        charge = self.charge_gate(prev_output)       # 从上一步提取有价值信息
        buffer = charge * prev_output                # 加权存入缓冲
        discharge = self.discharge_gate(current_output)
        return current_output + discharge * buffer   # 平滑注入当前输出


# ============================================================
# 淘汰审核 — 诚实度损失函数
# 作用：同时惩罚"随机乱说"和"听起来合理但错误"两种失真
# ============================================================

class TaoTaiShenHe(nn.Module):
    """
    淘汰审核 — 太初架构的诚实度量化机制

    说真话的对立面有两种：
        1. 随机乱说    — 输出对微小扰动极度敏感（不稳定）
        2. 系统性错误  — 输出自洽但模棱两可（过于平坦）

    淘汰审核同时惩罚这两种失真：
        扰动敏感性惩罚 — 加微小噪声后输出变化越大惩罚越重
        平坦度惩罚     — 输出概率分布越均匀惩罚越重
    """

    def __init__(self, noise_scale: float = 0.01, alpha: float = 0.1, beta: float = 0.05):
        super().__init__()
        self.noise_scale = noise_scale  # 扰动噪声幅度
        self.alpha = alpha              # 扰动敏感性权重
        self.beta = beta                # 平坦度权重

    def forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        logits_original: torch.Tensor,
        prev_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算淘汰审核损失"""
        embeddings = model.embedding(input_ids)

        # 在embedding空间加入微小扰动
        noise = torch.randn_like(embeddings) * self.noise_scale
        embeddings_perturbed = embeddings + noise

        with torch.no_grad():
            dorf_out, _ = model.dongTaiBaGuaZhen(embeddings_perturbed)
            input_heads = [
                embeddings_perturbed[:, :, i*model.head_dim:(i+1)*model.head_dim]
                for i in range(8)
            ]
            output_heads = [
                dorf_out[:, :, i*model.head_dim:(i+1)*model.head_dim]
                for i in range(8)
            ]
            evaluations = {
                i: model.guaxiang_bianma_heads[i].evaluate(output_heads[i], input_heads[i])
                for i in range(8)
            }
            targets = model.taoTaiDiXiaoJiZhi.identify_targets(evaluations)
            masked = model.taoTaiDiXiaoJiZhi.eliminate(output_heads, targets)
            merged = torch.cat(masked, dim=-1)
            prev = prev_output if prev_output is not None else embeddings_perturbed
            smoothed = model.suanLiHuanChongQu(merged, prev)
            logits_perturbed = model.output_proj(model.output_norm(smoothed))

        probs_orig = F.softmax(logits_original.detach(), dim=-1)
        probs_pert = F.softmax(logits_perturbed, dim=-1)

        # 扰动敏感性惩罚（KL散度）
        kl = F.kl_div(
            F.log_softmax(logits_original.detach(), dim=-1),
            probs_pert, reduction='batchmean'
        )

        # 平坦度惩罚（输出熵）
        entropy = -torch.sum(
            probs_orig * torch.log(probs_orig + 1e-8), dim=-1
        ).mean()

        return self.alpha * kl + self.beta * entropy


# ============================================================
# 太初完整模型（去中心化自运算驱动）
# ============================================================

class TaiChu(nn.Module):
    """
    太初 (TaiChu) 完整模型

    七个模块协同运作：
        动态八卦阵     → 解构输入信息，实时重组拓扑
        卦象对冲       → 驱动动态八卦阵的全动态阻抗
        卦象编码       → 上帝视角评估每个卦象的价值
        淘汰低效机制   → 清除低价值路径
        算力缓冲区     → 平滑淘汰后的信号跳变
        淘汰审核       → 损失函数，惩罚两种"说假话"
        去中心化自运算 → 全局逻辑压强调节，驱动持续进化
    """

    def __init__(
        self,
        dim: int = 256,
        vocab_size: int = 1000,
        survival_threshold: float = 0.3,
        polarity_dim: int = 32,
    ):
        super().__init__()
        assert dim % 8 == 0
        self.dim = dim
        self.head_dim = dim // 8

        # 输入嵌入
        self.embedding = nn.Embedding(vocab_size, dim)

        # 模块一&二：动态八卦阵（内含卦象对冲）
        self.dongTaiBaGuaZhen = DongTaiBaGuaZhen(dim=dim, polarity_dim=polarity_dim)

        # 模块三：卦象编码（每个卦象一个评估器）
        self.guaxiang_bianma_heads = nn.ModuleList([
            GuaXiangBianMa(head_dim=self.head_dim) for _ in range(8)
        ])

        # 模块四：淘汰低效机制
        self.taoTaiDiXiaoJiZhi = TaoTaiDiXiaoJiZhi(survival_threshold)

        # 模块五：算力缓冲区
        self.suanLiHuanChongQu = SuanLiHuanChongQu(dim)

        # 输出层
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size)

        # 模块七：去中心化自运算（逻辑压强状态）
        self.logic_pressure = 1.0
        self.pressure_history: List[float] = []
        self.impedance_history: List[np.ndarray] = []

    def _split_heads(self, x: torch.Tensor) -> List[torch.Tensor]:
        """将 [B, S, D] 切分为8个 [B, S, head_dim]"""
        return [x[:, :, i*self.head_dim:(i+1)*self.head_dim] for i in range(8)]

    def _quzhongxinhua_ziyunsuan(self, avg_survival: float, elimination_count: int):
        """
        去中心化自运算 — 模块七
        根据生存分数动态调整逻辑压强和淘汰阈值：
            生存分数低 → 压强升高 → 淘汰更激进
            系统稳定   → 压强平衡 → 维持现状
        """
        delta = (1.0 - avg_survival) * 0.1 - elimination_count * 0.02
        self.logic_pressure = max(0.5, min(3.0, self.logic_pressure + delta))
        self.pressure_history.append(self.logic_pressure)
        # 压强越高，淘汰阈值越高，系统净化越激进
        self.taoTaiDiXiaoJiZhi.survival_threshold = min(0.6, 0.3 * self.logic_pressure)

    def forward(
        self,
        input_ids: torch.Tensor,
        prev_output: Optional[torch.Tensor] = None,
        record_impedance: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        x = self.embedding(input_ids)                        # [B, S, D]

        # 动态八卦阵（含卦象对冲）
        dorf_output, impedance = self.dongTaiBaGuaZhen(x)

        if record_impedance:
            self.impedance_history.append(
                impedance.mean(dim=0).detach().cpu().numpy()
            )

        # 卦象编码：上帝视角评估各卦象价值
        input_heads = self._split_heads(x)
        output_heads = self._split_heads(dorf_output)
        evaluations = {
            i: self.guaxiang_bianma_heads[i].evaluate(output_heads[i], input_heads[i])
            for i in range(8)
        }

        # 淘汰低效机制：清除低价值路径
        targets = self.taoTaiDiXiaoJiZhi.identify_targets(evaluations)
        masked = self.taoTaiDiXiaoJiZhi.eliminate(output_heads, targets)
        merged = torch.cat(masked, dim=-1)

        # 算力缓冲区：平滑信号跳变
        if prev_output is None:
            prev_output = x
        smoothed = self.suanLiHuanChongQu(merged, prev_output)

        # 去中心化自运算：更新逻辑压强
        avg_survival = sum(e['survival_score'].item() for e in evaluations.values()) / 8
        self._quzhongxinhua_ziyunsuan(avg_survival, len(targets))

        logits = self.output_proj(self.output_norm(smoothed))

        info = {
            'elimination_count': len(targets),
            'avg_survival': avg_survival,
            'logic_pressure': self.logic_pressure,
            'impedance_mean': impedance.mean().item(),
            'impedance_std': impedance.std().item(),
        }

        return logits, smoothed, info


# ============================================================
# 对比基准：标准多头注意力
# ============================================================

class StandardMHA(nn.Module):
    """标准多头注意力模型（对比基准，参数量与太初对齐）"""

    def __init__(self, dim: int = 256, vocab_size: int = 1000, num_heads: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return self.output_proj(x)


# ============================================================
# 对比实验
# ============================================================

def run_experiment(
    num_steps: int = 200,
    batch_size: int = 8,
    seq_len: int = 32,
    vocab_size: int = 500,
    dim: int = 256,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    use_taoTaiShenHe: bool = True,  # 是否启用淘汰审核
):
    print("=" * 60)
    print("  太初 (TaiChu) vs 标准MHA 对比实验")
    print("=" * 60)

    taichu = TaiChu(
        dim=dim, vocab_size=vocab_size,
        survival_threshold=0.3, polarity_dim=32
    ).to(device)

    mha = StandardMHA(dim=dim, vocab_size=vocab_size, num_heads=8).to(device)

    taichu_params = sum(p.numel() for p in taichu.parameters())
    mha_params = sum(p.numel() for p in mha.parameters())
    print(f"\n参数量对比:")
    print(f"  太初:      {taichu_params:>10,} ({taichu_params/1e6:.3f}M)")
    print(f"  标准MHA:   {mha_params:>10,} ({mha_params/1e6:.3f}M)")

    opt_t = optim.AdamW(taichu.parameters(), lr=learning_rate)
    opt_m = optim.AdamW(mha.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    taoTaiShenHe = TaoTaiShenHe(noise_scale=0.01, alpha=0.1, beta=0.05)

    taichu_losses, mha_losses = [], []
    taichu_times, mha_times = [], []
    pressures, eliminations = [], []

    print(f"\n开始训练 ({num_steps}步)...")
    print(f"{'步骤':>6} | {'太初Loss':>10} | {'MHA Loss':>10} | "
          f"{'逻辑压强':>8} | {'淘汰数':>6} | {'阻抗均值':>8}")
    print("-" * 65)

    prev_output = None

    for step in range(num_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        # ── 训练太初 ──
        t0 = time.time()
        opt_t.zero_grad()
        logits_t, prev_output, info = taichu(
            input_ids, prev_output, record_impedance=(step % 20 == 0)
        )
        prev_output = prev_output.detach()

        ce_loss = loss_fn(logits_t.reshape(-1, vocab_size), target_ids.reshape(-1))
        if use_taoTaiShenHe:
            shenhe_loss = taoTaiShenHe(taichu, input_ids, logits_t, prev_output)
            total_loss = ce_loss + shenhe_loss
        else:
            total_loss = ce_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(taichu.parameters(), max_norm=1.0)
        opt_t.step()
        taichu_times.append(time.time() - t0)
        taichu_losses.append(ce_loss.item())
        pressures.append(info['logic_pressure'])
        eliminations.append(info['elimination_count'])

        # ── 训练标准MHA ──
        t0 = time.time()
        opt_m.zero_grad()
        logits_m = mha(input_ids)
        loss_m = loss_fn(logits_m.reshape(-1, vocab_size), target_ids.reshape(-1))
        loss_m.backward()
        torch.nn.utils.clip_grad_norm_(mha.parameters(), max_norm=1.0)
        opt_m.step()
        mha_times.append(time.time() - t0)
        mha_losses.append(loss_m.item())

        if step % 20 == 0:
            print(
                f"{step:>6} | {ce_loss.item():>10.4f} | {loss_m.item():>10.4f} | "
                f"{info['logic_pressure']:>8.3f} | {info['elimination_count']:>6}/8 | "
                f"{info['impedance_mean']:>8.4f}"
            )

    print("-" * 65)

    final_window = 20
    t_final = np.mean(taichu_losses[-final_window:])
    m_final = np.mean(mha_losses[-final_window:])
    print(f"\n最终{final_window}步平均Loss:")
    print(f"  太初:      {t_final:.4f}")
    print(f"  标准MHA:   {m_final:.4f}")
    print(f"  差值:      {t_final - m_final:+.4f} "
          f"({'太初更优' if t_final < m_final else 'MHA更优'})")
    print(f"\n平均每步耗时:")
    print(f"  太初:      {np.mean(taichu_times)*1000:.2f}ms")
    print(f"  标准MHA:   {np.mean(mha_times)*1000:.2f}ms")

    _plot_results(taichu_losses, mha_losses, pressures, eliminations,
                  taichu.impedance_history)

    return {'taichu_losses': taichu_losses, 'mha_losses': mha_losses,
            'taichu_final': t_final, 'mha_final': m_final}


def _plot_results(taichu_losses, mha_losses, pressures, eliminations, impedance_history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TaiChu vs Standard MHA — Experiment Results', fontsize=14, fontweight='bold')

    def smooth(data, w=10):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    # Loss comparison
    ax = axes[0, 0]
    ax.plot(smooth(taichu_losses), label='TaiChu', color='#E84545', linewidth=2)
    ax.plot(smooth(mha_losses), label='Standard MHA', color='#2B4EFF', linewidth=2, linestyle='--')
    ax.set_title('Training Loss Comparison')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Decentralized Self-Evolution (Logic Pressure)
    ax = axes[0, 1]
    ax.plot(pressures, color='#FF8C00', linewidth=2)
    ax.set_title('Decentralized Self-Evolution — Logic Pressure')
    ax.set_xlabel('Step')
    ax.set_ylabel('Logic Pressure')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Initial pressure')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Elimination Mechanism
    ax = axes[1, 0]
    ax.bar(range(len(eliminations)), eliminations, color='#9B59B6', alpha=0.6, width=1.0)
    ax.set_title('Low-Efficiency Elimination — Count per Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('Heads Eliminated / 8')
    ax.set_ylim(0, 8)
    ax.grid(True, alpha=0.3)

    # Polarity Clash — Dynamic Impedance Heatmap
    ax = axes[1, 1]
    if impedance_history:
        im = ax.imshow(impedance_history[-1], cmap='RdYlGn_r', aspect='auto')
        plt.colorbar(im, ax=ax)
        trigrams = ['Qian', 'Kun', 'Zhen', 'Xun', 'Kan', 'Li', 'Gen', 'Dui']
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(trigrams, fontsize=8)
        ax.set_yticklabels(trigrams, fontsize=8)
        ax.set_title('Polarity Clash — Dynamic Impedance Matrix\nRed=High (Repulsion), Green=Low (Attraction)')
    else:
        ax.text(0.5, 0.5, 'No impedance data yet', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('taichu_results.png', dpi=150, bbox_inches='tight')
    print("\n结果图表已保存至: taichu_results.png")
    plt.show()


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    CONFIG = {
        'num_steps':         200,
        'batch_size':        8,
        'seq_len':           32,
        'vocab_size':        500,
        'dim':               256,
        'learning_rate':     1e-3,
        'use_taoTaiShenHe':  True,   # 淘汰审核开关
        'device':            'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print(f"运行设备: {CONFIG['device']}")
    print(f"淘汰审核: {'启用' if CONFIG['use_taoTaiShenHe'] else '禁用'}")

    results = run_experiment(**CONFIG)

    print("\n" + "=" * 60)
    print("  太初实验完成")
    print("  结果图表: taichu_results.png")
    print("=" * 60)

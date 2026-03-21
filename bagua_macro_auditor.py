"""
八卦架构 — 宏观淘汰审核模块
============================
作者：阳恩硕 (Yang Enshuo)

这是淘汰审核从微观（卦象级别）到宏观（句子级别）的延伸：

微观淘汰审核：评估每个卦象头的输出质量，淘汰低效路径
宏观淘汰审核：评估整句话的置信度，低于阈值则重新生成

设计参数：
- 触发时机：遇到句号/问号/感叹号（句子结束）
- 通过阈值：平均置信度 ≥ 0.3
- 最大重试：3次
- 不通过处理：重新生成整句（不修补局部，避免语序断裂）
- 生效范围：训练时筛样本 + 推理时过滤输出

使用方式：
    把此文件里的 HongGuanTaoTaiShenHe 类
    复制到 bagua_llm_train_v4.py 里的架构定义部分
    然后在 BaGuaLLM.__init__ 里注册
    在训练循环和generate函数里调用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ============================================================
# 宏观淘汰审核类
# ============================================================

class HongGuanTaoTaiShenHe(nn.Module):
    """
    宏观淘汰审核 — 句子级别输出质量评估

    工作原理：
    1. 接收一句话的logits（模型对每个token的预测分布）
    2. 计算每个token的最大置信度（softmax后的最大概率）
    3. 对整句话取平均置信度
    4. 低于阈值则判定为低质量，需要重新生成

    训练时：低质量样本直接跳过，不做梯度更新
    推理时：低质量句子重新生成，最多重试3次
    """

    # 句子结束标点的token ID（bert-base-multilingual-cased）
    # 句号=119、问号=136、感叹号=106、英文句号=119
    SENTENCE_END_TOKENS = {119, 136, 106, 1012, 1029, 1013}

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        max_retries: int = 3,
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

    def compute_sentence_confidence(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算一句话的平均置信度

        Args:
            logits: [B, S, V] 模型输出的logits
            token_ids: [B, S] 对应的token ID序列

        Returns:
            confidence: [B] 每个样本的平均置信度（0到1之间）
        """
        # softmax得到概率分布
        probs = F.softmax(logits, dim=-1)  # [B, S, V]

        # 每个位置的最大概率就是该token的置信度
        max_probs, _ = probs.max(dim=-1)  # [B, S]

        # 找到句子结束位置
        # 简单处理：对整个序列取平均置信度
        confidence = max_probs.mean(dim=-1)  # [B]

        return confidence

    def should_regenerate(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Tuple[bool, float]:
        """
        判断这句话是否需要重新生成

        Returns:
            (need_regen, confidence_score)
        """
        with torch.no_grad():
            confidence = self.compute_sentence_confidence(logits, token_ids)
            avg_confidence = confidence.mean().item()

        need_regen = avg_confidence < self.confidence_threshold
        return need_regen, avg_confidence

    def filter_training_sample(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> bool:
        """
        训练时使用：判断这个样本是否值得做梯度更新

        Returns:
            True = 样本质量够，正常训练
            False = 样本质量差，跳过这个batch
        """
        need_regen, confidence = self.should_regenerate(logits, token_ids)
        return not need_regen  # 不需要重新生成 = 可以训练


# ============================================================
# 如何集成到 BaGuaLLM 类
# ============================================================

INTEGRATION_GUIDE = """
# ── 第一步：在 BaGuaLLM.__init__ 里注册宏观淘汰审核 ──

    # 宏观淘汰审核（句子级别）
    self.honguan_shenhe = HongGuanTaoTaiShenHe(
        confidence_threshold=0.3,
        max_retries=3,
    )


# ── 第二步：训练循环里加入样本过滤 ──

    with torch.amp.autocast('cuda', enabled=(device=='cuda')):
        logits = model(input_ids)
        B, S, V = logits.shape

        # 宏观淘汰审核：低质量样本跳过
        if not model.honguan_shenhe.filter_training_sample(
            logits.detach(), input_ids
        ):
            optimizer.zero_grad()
            continue  # 跳过这个batch，不做梯度更新

        loss = loss_fn(logits.reshape(-1, V), target_ids.reshape(-1))
        loss = loss + model.shenhe_layers[0][0].honesty_loss(logits.reshape(-1, V))
        loss = loss / CONFIG_TRAIN['grad_accum']


# ── 第三步：generate函数里加入推理时过滤 ──

    # 在generate函数里，每生成一句话检查一次
    # 遇到句号/问号/感叹号时触发

    sentence_logits = []  # 收集当前句子的logits
    sentence_tokens = []  # 收集当前句子的token
    best_sentence = None
    best_confidence = -1
    retry_count = 0

    for _ in range(max_new_tokens):
        logits = self(generated)
        next_logits = logits[:, -1, :]
        # ... 采样得到next_token ...

        sentence_logits.append(logits[:, -1:, :])
        sentence_tokens.append(next_token)

        # 遇到句子结束标点，触发宏观审核
        if next_token.item() in HongGuanTaoTaiShenHe.SENTENCE_END_TOKENS:
            sent_logits = torch.cat(sentence_logits, dim=1)
            sent_tokens = torch.cat(sentence_tokens, dim=1)

            need_regen, confidence = self.honguan_shenhe.should_regenerate(
                sent_logits, sent_tokens
            )

            if confidence > best_confidence:
                best_confidence = confidence
                best_sentence = (sentence_logits.copy(), sentence_tokens.copy())

            if need_regen and retry_count < self.honguan_shenhe.max_retries:
                # 回退到句子开始位置，重新生成
                retry_count += 1
                generated = generated[:, :-len(sentence_tokens)]
                sentence_logits = []
                sentence_tokens = []
                continue
            else:
                # 通过审核或达到重试上限，继续生成下一句
                retry_count = 0
                sentence_logits = []
                sentence_tokens = []
"""

if __name__ == "__main__":
    print("宏观淘汰审核模块")
    print(f"置信度阈值：0.3")
    print(f"最大重试次数：3")
    print()
    print("集成指南：")
    print(INTEGRATION_GUIDE)

    # 简单测试
    module = HongGuanTaoTaiShenHe(confidence_threshold=0.3, max_retries=3)

    # 模拟高置信度输出
    high_conf_logits = torch.zeros(1, 10, 1000)
    high_conf_logits[:, :, 42] = 10.0  # 某个token概率极高
    token_ids = torch.zeros(1, 10, dtype=torch.long)

    need_regen, conf = module.should_regenerate(high_conf_logits, token_ids)
    print(f"高置信度测试：置信度={conf:.3f}，需要重新生成={need_regen}")

    # 模拟低置信度输出（均匀分布）
    low_conf_logits = torch.zeros(1, 10, 1000)
    need_regen, conf = module.should_regenerate(low_conf_logits, token_ids)
    print(f"低置信度测试：置信度={conf:.3f}，需要重新生成={need_regen}")

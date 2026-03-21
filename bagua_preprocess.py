"""
八卦架构 LLM 数据预处理脚本
============================
作者：阳恩硕 (Yang Enshuo)
功能：把40GB文本提前分词，存成二进制格式
好处：训练时CPU只读数字，不需要解码分词
     速度提升10倍，GPU功率从50W拉到200W+

运行环境：本地电脑（不需要GPU）
运行时间：约2-3小时
运行方式：python bagua_preprocess.py

运行完成后：
    把生成的 bagua_tokens/ 文件夹拷贝到V100服务器桌面
    训练脚本会自动检测并使用二进制数据
"""

import os
import json
import struct
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 强制所有缓存指向D盘，不占用C盘空间
os.environ['HF_HOME'] = 'D:/bagua_cache'
os.environ['HF_DATASETS_CACHE'] = 'D:/bagua_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/bagua_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/bagua_cache'
Path('D:/bagua_cache').mkdir(exist_ok=True)


# ============================================================
# 自动检测路径
# ============================================================

def get_data_dir():
    candidates = [
        Path("E:/bagua_data"),           # 机械硬盘（主要路径）
        Path("D:/bagua_data"),
        Path("C:/Users/Administrator/Desktop/bagua_data"),
        Path("C:/Users/Admin/Desktop/bagua_data"),
        Path(os.path.expanduser("~/Desktop/bagua_data")),
        Path("./bagua_data"),
    ]
    for p in candidates:
        if p.exists() and (p / "config.json").exists():
            print(f"找到数据目录：{p}")
            return p
    print("未找到bagua_data目录")
    import sys; sys.exit(1)

DATA_DIR = get_data_dir()
CONFIG = json.loads((DATA_DIR / "config.json").read_text(encoding='utf-8'))
TOKENIZER_PATH = str(DATA_DIR / "tokenizer")
OWT_PATH = str(DATA_DIR / "openwebtext")
ZH_PATH = str(DATA_DIR / "chinese_wiki")
print(f'数据源目录：{DATA_DIR}')
print(f'Token输出目录：D:/bagua_tokens')

# 输出目录：存到D盘，不占C盘空间
TOKEN_DIR = Path('D:/bagua_tokens')
TOKEN_DIR.mkdir(exist_ok=True)
print(f'Token文件将保存到：{TOKEN_DIR}')


# ============================================================
# 分词器加载
# ============================================================

def load_tokenizer():
    from transformers import BertTokenizerFast, AutoTokenizer
    
    vocab_file = DATA_DIR / "tokenizer" / "vocab.txt"
    tokenizer_dir = DATA_DIR / "tokenizer"
    
    if vocab_file.exists():
        # 本地已有，直接加载
        print("本地分词器已找到，直接加载...")
        tokenizer = BertTokenizerFast(
            vocab_file=str(vocab_file),
            do_lower_case=True
        )
    else:
        # 本地没有，自动下载并保存
        print("本地没有分词器，正在从网络下载（需要翻墙）...")
        tokenizer_dir.mkdir(exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )
        tokenizer.save_pretrained(str(tokenizer_dir))
        print(f"分词器已保存到：{tokenizer_dir}")
    
    print(f"分词器加载完成，词表大小：{tokenizer.vocab_size}")
    return tokenizer


# ============================================================
# 核心处理函数：文本 → 二进制token文件
# ============================================================

def process_file(
    input_path: str,
    output_path: str,
    tokenizer,
    chunk_size: int = 100_000,  # 每次读取100KB文本
    max_bytes: int = None,
    lang: str = 'en'
):
    """
    把文本文件分词后存成二进制
    格式：每个token存为uint16（2字节），节省空间
    """
    print(f"\n正在处理：{input_path}")
    print(f"输出到：{output_path}")

    input_path = Path(input_path)
    output_path = Path(output_path)

    total_size = input_path.stat().st_size
    if max_bytes:
        total_size = min(total_size, max_bytes)

    total_tokens = 0
    buffer = []
    FLUSH_SIZE = 1_000_000  # 每100万个token写一次磁盘

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_path, 'wb') as fout:

        pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"处理{lang}文本")
        bytes_read = 0

        while True:
            if max_bytes and bytes_read >= max_bytes:
                break

            chunk = fin.read(chunk_size)
            if not chunk:
                break

            bytes_read += len(chunk.encode('utf-8', errors='ignore'))
            pbar.update(min(len(chunk), total_size - pbar.n))

            # 分词
            tokens = tokenizer.encode(chunk, add_special_tokens=False)
            buffer.extend(tokens)
            total_tokens += len(tokens)

            # 批量写入磁盘
            if len(buffer) >= FLUSH_SIZE:
                arr = np.array(buffer, dtype=np.uint16)
                fout.write(arr.tobytes())
                buffer = []

        # 写入剩余
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            fout.write(arr.tobytes())

        pbar.close()

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"完成：{total_tokens:,} 个token，文件大小：{file_size:.0f}MB")
    return total_tokens


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  八卦架构 LLM 数据预处理工具")
    print("  文本 → 二进制token，训练速度提升10倍")
    print("=" * 55)

    tokenizer = load_tokenizer()

    # 处理英文数据（取前20GB，约200亿token）
    en_tokens = process_file(
        input_path=f"{OWT_PATH}/train.txt",
        output_path=str(TOKEN_DIR / "en_tokens.bin"),
        tokenizer=tokenizer,
        max_bytes=20_000_000_000,  # 20GB
        lang='英文'
    )

    # 处理中文数据（全量，约2GB）
    zh_tokens = process_file(
        input_path=f"{ZH_PATH}/train.txt",
        output_path=str(TOKEN_DIR / "zh_tokens.bin"),
        tokenizer=tokenizer,
        max_bytes=None,  # 全量
        lang='中文'
    )

    # 保存元信息
    meta = {
        "en_tokens_path": str(TOKEN_DIR / "en_tokens.bin"),
        "zh_tokens_path": str(TOKEN_DIR / "zh_tokens.bin"),
        "en_total_tokens": en_tokens,
        "zh_total_tokens": zh_tokens,
        "total_tokens": en_tokens + zh_tokens,
        "dtype": "uint16",
        "vocab_size": tokenizer.vocab_size,
        "preprocessed": True,
    }
    meta_path = TOKEN_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    total_gb = (en_tokens + zh_tokens) * 2 / 1024**3
    print("\n" + "=" * 55)
    print("  预处理完成！")
    print(f"  英文token数：{en_tokens:,}")
    print(f"  中文token数：{zh_tokens:,}")
    print(f"  合计：{en_tokens + zh_tokens:,} 个token")
    print(f"  二进制文件大小：{total_gb:.1f}GB")
    print(f"  保存目录：{TOKEN_DIR}")
    print("  把 bagua_tokens/ 文件夹拷贝到V100服务器桌面")
    print("=" * 55)

import json
import re
import regex
from typing import Dict, List, Tuple, Iterator, Iterable, Union, Optional, IO
from collections import Counter

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """
    BPE (Byte-Pair Encoding) Tokenizer 实现
    
    支持以下功能：
    - 从预训练的 vocab 和 merges 加载
    - 对文本进行编码/解码
    - 处理特殊 token
    - 内存高效流式编码
    """
    
    def __init__(
        self, 
        vocab: Dict[int, bytes], 
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None
    ):
        """
        初始化 tokenizer
        
        Args:
            vocab: {token_id: token_bytes} 词汇表
            merges: [(token1, token2), ...] 合并规则列表，按训练顺序排列
            special_tokens: 特殊 token 列表
        """
        self.vocab = vocab
        self.merges = merges
        
        # 构建反向映射
        self.vocab_r = {v: k for k, v in vocab.items()}  # {token_bytes: token_id}
        
        # 处理特殊 token
        self.special_tokens = special_tokens or []
        self.special_tokens_r = {token.encode('utf-8'): len(self.vocab) + i 
                                  for i, token in enumerate(self.special_tokens)}
        
        # 将特殊 token 添加到 vocab 中
        next_id = len(self.vocab)
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in self.vocab_r:
                self.vocab[next_id] = token_bytes
                self.vocab_r[token_bytes] = next_id
                next_id += 1
        
        # 构建 merges 映射 { (token1, token2) -> token_id }
        self.merge_dict = {}
        for token1, token2 in self.merges:
            merged_token = token1 + token2
            if merged_token in self.vocab_r:
                self.merge_dict[(token1, token2)] = self.vocab_r[merged_token]
        
        # 预编译特殊 token 正则表达式
        if self.special_tokens:
            # 转义特殊字符并按长度降序排列，避免短 token 匹配长 token 的子串
            escaped_special_tokens = [re.escape(token) for token in self.special_tokens]
            escaped_special_tokens.sort(key=len, reverse=True)
            self.special_token_regex = re.compile('|'.join(escaped_special_tokens))
        else:
            self.special_token_regex = None
    
    @classmethod
    def from_files(
        cls, 
        vocab_filepath: Union[str, IO], 
        merges_filepath: Union[str, IO],
        special_tokens: Optional[List[str]] = None
    ):
        """
        从文件加载 tokenizer
        
        Args:
            vocab_filepath: 词汇表文件路径 (JSON 格式)
            merges_filepath: 合并规则文件路径 (txt 格式)
            special_tokens: 特殊 token 列表
        """
        # 读取词汇表
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # 转换为 {id: bytes}
        vocab = {int(k): v.encode('utf-8') if isinstance(v, str) else v 
                 for k, v in vocab_data.items()}
        
        # 读取合并规则
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        token1, token2 = parts
                        merges.append((token1.encode('utf-8'), token2.encode('utf-8')))
        
        return cls(vocab, merges, special_tokens)
    
    def _pretokenize(self, text: str) -> List[str]:
        """
        预分词：将文本分割成预token，特殊token保持完整
        
        Args:
            text: 输入文本
        Returns:
            预token列表
        """
        tokens = []
        if self.special_token_regex:
            # 使用正则表达式分割，保留特殊 token
            last_end = 0
            for match in self.special_token_regex.finditer(text):
                start, end = match.span()
                # 添加特殊 token 之前的部分
                if start > last_end:
                    chunk = text[last_end:start]
                    tokens.extend(regex.findall(GPT2_SPLIT_PATTERN, chunk))
                # 添加特殊 token
                tokens.append(match.group(0))
                last_end = end
            # 添加最后的部分
            if last_end < len(text):
                chunk = text[last_end:]
                tokens.extend(regex.findall(GPT2_SPLIT_PATTERN, chunk))
        else:
            tokens = regex.findall(GPT2_SPLIT_PATTERN, text)
        
        # 过滤空字符串
        tokens = [t for t in tokens if t]
        
        return tokens
    
    def _encode_bytes(self, text_bytes: bytes) -> List[int]:
        """
        将字节序列编码为 token ID 序列（BPE 核心逻辑）
        
        Args:
            text_bytes: 输入字节序列
        Returns:
            token ID 列表
        """
        # 将字节序列转换为初始 token 序列（每个字节作为一个 token）
        tokens = list(text_bytes)
        
        # 应用合并规则
        while len(tokens) > 1:
            # 找到最佳合并位置（最早出现的可合并相邻对）
            best_idx = -1
            best_score = float('inf')
            
            for i in range(len(tokens) - 1):
                token1 = bytes([tokens[i]])
                token2 = bytes([tokens[i + 1]])
                
                # 检查这对是否在合并规则中
                if (token1, token2) in self.merge_dict:
                    # 使用合并规则的索引作为优先级（越早训练的合并规则优先级越高）
                    merge_idx = -1
                    for idx, (m1, m2) in enumerate(self.merges):
                        if m1 == token1 and m2 == token2:
                            merge_idx = idx
                            break
                    
                    if merge_idx != -1 and merge_idx < best_score:
                        best_idx = i
                        best_score = merge_idx
            
            # 如果没有可合并的对，退出
            if best_idx == -1:
                break
            
            # 执行合并
            token1 = bytes([tokens[best_idx]])
            token2 = bytes([tokens[best_idx + 1]])
            merged_token = token1 + token2
            
            # 替换两个 token 为一个合并后的 token
            new_token_id = self.vocab_r[merged_token]
            tokens = tokens[:best_idx] + [new_token_id] + tokens[best_idx + 2:]
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token ID 序列
        
        Args:
            text: 输入文本
        Returns:
            token ID 列表
        """
        token_ids = []
        
        # 预分词
        pretokens = self._pretokenize(text)
        
        for pretoken in pretokens:
            if pretoken in self.special_tokens:
                # 特殊 token：直接映射到 ID
                token_id = self.vocab_r[pretoken.encode('utf-8')]
                token_ids.append(token_id)
            else:
                # 普通文本：转为字节，然后 BPE 编码
                text_bytes = pretoken.encode('utf-8')
                ids = self._encode_bytes(text_bytes)
                token_ids.extend(ids)
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        将可迭代的字符串序列编码为 token ID 流（内存高效）
        
        Args:
            iterable: 字符串可迭代对象
        Yields:
            单个 token ID
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    
    def decode(self, ids: List[int]) -> str:
        """
        将 token ID 序列解码为文本
        
        Args:
            ids: token ID 列表
        Returns:
            解码后的文本
        """
        # 将 ID 转换为字节序列
        byte_sequence = b''
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]
            else:
                # 如果 ID 不在词汇表中，跳过
                continue
        
        # 尝试解码为字符串，处理可能的编码错误
        try:
            return byte_sequence.decode('utf-8')
        except UnicodeDecodeError:
            # 使用替换字符处理错误
            return byte_sequence.decode('utf-8', errors='replace')


def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on a text file.
    
    Args:
        input_path: Path to the training text file.
        vocab_size: Target vocabulary size.
        special_tokens: List of special tokens.
        
    Returns:
        Tuple of (vocab, merges).
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x

    # 1. Read Data
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Pre-tokenize
    special_token_regex = None
    if special_tokens:
        escaped = [re.escape(t) for t in special_tokens]
        escaped.sort(key=len, reverse=True)
        special_token_regex = re.compile('|'.join(escaped))

    words = []
    if special_token_regex:
        last_end = 0
        for match in special_token_regex.finditer(text):
            start, end = match.span()
            if start > last_end:
                chunk = text[last_end:start]
                words.extend(regex.findall(GPT2_SPLIT_PATTERN, chunk))
            last_end = end
        if last_end < len(text):
            chunk = text[last_end:]
            words.extend(regex.findall(GPT2_SPLIT_PATTERN, chunk))
    else:
        words = regex.findall(GPT2_SPLIT_PATTERN, text)
    
    # 3. Count word frequencies
    # Convert words to bytes for consistency
    words_bytes = [w.encode('utf-8') for w in words]
    word_counts = Counter(words_bytes)

    # 4. Initial vocab
    # Start with all 256 bytes
    vocab = {i: bytes([i]) for i in range(256)}
    # Add special tokens
    next_id = 256
    for st in special_tokens:
        st_bytes = st.encode('utf-8')
        # Check if special token is already in vocab (unlikely for bytes 0-255 unless 1 char)
        if st_bytes not in vocab.values():
            vocab[next_id] = st_bytes
            next_id += 1
            
    # 5. Prepare split words
    # {word_bytes: [b'w', b'o', b'r', b'd']}
    split_words = {wb: [bytes([b]) for b in wb] for wb in word_counts}
    
    merges = []
    
    # 6. Merge loop
    pbar = tqdm(total=vocab_size - len(vocab))
    while len(vocab) < vocab_size:
        pairs = {}
        for wb, count in word_counts.items():
            split = split_words[wb]
            if len(split) < 2:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                pairs[pair] = pairs.get(pair, 0) + count
        
        if not pairs:
            break
            
        best_pair = max(pairs, key=pairs.get)
        
        # Add to merges
        merges.append(best_pair)
        
        # Add to vocab
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        
        # Update split_words
        for wb in split_words:
            split = split_words[wb]
            if len(split) < 2:
                continue
            
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == best_pair[0] and split[i+1] == best_pair[1]:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            split_words[wb] = new_split
            
        pbar.update(1)
    pbar.close()
            
    return vocab, merges
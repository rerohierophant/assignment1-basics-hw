import json
import re
import regex
from typing import Dict, List, Tuple, Iterator, Iterable, Union, Optional, IO
from collections import Counter

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


from functools import lru_cache

@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_byte_encoder() -> Dict[int, str]:
    return bytes_to_unicode()

def get_byte_decoder() -> Dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}


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
        self.ranks = {}  # 缓存合并规则的优先级
        for i, (token1, token2) in enumerate(self.merges):
            self.ranks[(token1, token2)] = i
            merged_token = token1 + token2
            if merged_token in self.vocab_r:
                self.merge_dict[(token1, token2)] = self.vocab_r[merged_token]
        
        # 缓存已编码的词
        self.cache = {}
        
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
        # 准备 decoder
        byte_decoder = get_byte_decoder()
        
        def unicode_to_bytes(s: str) -> bytes:
            # 对于不在映射表中的字符（可能是特殊token或其他），尝试用 utf-8 编码
            # 但对于 BPE 词表中的项，它们应该都在映射表中
            return bytes([byte_decoder[c] for c in s])

        # 读取词汇表
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # 转换为 {id: bytes}
        vocab = {}
        for k, v in vocab_data.items():
            if isinstance(v, str):
                try:
                    vocab[int(k)] = unicode_to_bytes(v)
                except KeyError:
                    # Fallback: 如果映射表中没有，可能是特殊字符或者旧格式，尝试 utf-8
                    vocab[int(k)] = v.encode('utf-8')
            else:
                vocab[int(k)] = v
        
        # 读取合并规则
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        try:
                            merges.append((unicode_to_bytes(parts[0]), unicode_to_bytes(parts[1])))
                        except KeyError:
                             # Fallback
                            merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))
                            
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
        tokens = [bytes([b]) for b in text_bytes]
        
        # 应用合并规则
        while len(tokens) > 1:
            # 找到最佳合并位置（最早出现的可合并相邻对）
            best_idx = -1
            best_score = float('inf')
            
            for i in range(len(tokens) - 1):
                token1 = tokens[i]
                token2 = tokens[i + 1]
                
                # Check rank using cached dictionary - O(1) lookup
                merge_idx = self.ranks.get((token1, token2), -1)
                
                if merge_idx != -1 and merge_idx < best_score:
                    best_idx = i
                    best_score = merge_idx
            
            # 如果没有可合并的对，退出
            if best_idx == -1:
                break
            
            # 执行合并
            token1 = tokens[best_idx]
            token2 = tokens[best_idx + 1]
            merged_token = token1 + token2
            
            # 替换两个 token 为一个合并后的 token
            tokens = tokens[:best_idx] + [merged_token] + tokens[best_idx + 2:]
        
        return [self.vocab_r[t] for t in tokens]
    
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
                
                # Check cache
                if text_bytes in self.cache:
                    ids = self.cache[text_bytes]
                else:
                    ids = self._encode_bytes(text_bytes)
                    self.cache[text_bytes] = ids
                
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
    
    # Pre-calculate pair stats
    # pairs: (t1, t2) -> count
    pairs = Counter()
    # where_pair: (t1, t2) -> set(word_bytes)
    # This might be memory intensive, but faster than iterating all words
    where_pair = {} 
    
    for wb, count in word_counts.items():
        split = split_words[wb]
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            pairs[pair] += count
            if pair not in where_pair:
                where_pair[pair] = set()
            where_pair[pair].add(wb)
            
    while len(vocab) < vocab_size:
        if not pairs:
            break
            
        # Tie-breaking: prefer most frequent, then lexicographically greater
        # pairs.most_common(1) is not deterministic for ties
        # We need to find all pairs with the max count and pick the best one
        max_count = max(pairs.values())
        candidates = [p for p, c in pairs.items() if c == max_count]
        best_pair = max(candidates)
        
        # Add to merges
        merges.append(best_pair)
        
        # Add to vocab
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        
        # Update split_words only for words containing the best_pair
        # And incrementally update pairs stats
        words_to_update = where_pair.get(best_pair, set()).copy()
        
        # We don't need this pair in stats/index anymore
        del pairs[best_pair]
        del where_pair[best_pair]
        
        # Refined loop for correctness and simplicity
        for wb in words_to_update:
            split = split_words[wb]
            count = word_counts[wb]
            
            # 1. Remove old pairs
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                pairs[pair] -= count
                if pairs[pair] <= 0:
                    del pairs[pair]
                if pair in where_pair:
                    where_pair[pair].discard(wb)
                    if not where_pair[pair]:
                        del where_pair[pair]
            
            # 2. Update split
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
            
            # 3. Add new pairs
            for i in range(len(new_split) - 1):
                pair = (new_split[i], new_split[i+1])
                pairs[pair] += count
                if pair not in where_pair:
                    where_pair[pair] = set()
                where_pair[pair].add(wb)
            
        pbar.update(1)
    pbar.close()
            
    return vocab, merges
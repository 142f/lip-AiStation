import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def default_bpe():
    # 虽然我们不再直接使用这个函数的返回值，但保留它以避免其他地方可能出现的引用错误
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # ==================================================================
        # [核心修改] 智能加载策略：自动适配路径 + 自动识别压缩格式
        # ==================================================================
        import os
        import gzip

        # 1. 获取当前脚本所在的目录 (models/clip/)，用于构建相对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. 定义所有可能的“藏宝地点”和“文件形态”
        # 程序会按顺序挨个去找，找到哪个用哪个
        candidate_paths = [
            # 优先级 A: 你传入的参数路径 (如果外部指定了)
            bpe_path,
            
            # 优先级 B: Kaggle 环境 / 本地相对路径 (最推荐，纯文本版)
            # 假设文件在 models/clip/bpe_simple_vocab_16e6.txt
            os.path.join(current_dir, "bpe_simple_vocab_16e6.txt"),
            
            # 优先级 C: Kaggle 环境 / 本地相对路径 (压缩版)
            # 假设文件在 models/clip/bpe_simple_vocab_16e6.txt.gz
            os.path.join(current_dir, "bpe_simple_vocab_16e6.txt.gz"),

            # 优先级 D: 服务器 1 的绝对路径 (压缩版)
            '/3240608030/bpe_simple_vocab_16e6.txt.gz',
            
            # 优先级 E: 服务器 1 的绝对路径 (纯文本版，防备万一)
            '/3240608030/bpe_simple_vocab_16e6.txt',
            
            # 优先级 F: Kaggle 特定的硬编码绝对路径
            "/kaggle/working/LipFD-main/LipFD-main-base/models/clip/bpe_simple_vocab_16e6.txt"
        ]

        merges = None
        loaded_path = None

        # 3. 循环尝试每一个路径
        for path in candidate_paths:
            # 只有当路径不为空且文件确实存在时，才尝试读取
            if path and os.path.exists(path):
                try:
                    # [关键逻辑] 根据后缀名决定怎么读
                    if path.endswith(".gz"):
                        # 针对服务器 1 的 .gz 文件：使用 gzip 打开，'rt'表示文本模式读取
                        print(f"[SimpleTokenizer] Found .gz file at {path}, unzipping...")
                        with gzip.open(path, 'rt', encoding='utf-8') as f:
                            merges = f.read().split('\n')
                    else:
                        # 针对 Kaggle 的 .txt 文件：使用普通 open 打开
                        print(f"[SimpleTokenizer] Found text file at {path}, reading...")
                        with open(path, "r", encoding="utf-8") as f:
                            merges = f.read().split('\n')
                    
                    loaded_path = path
                    break # 成功读取，跳出循环
                except Exception as e:
                    print(f"[Warning] Found file at {path} but failed to read: {e}")
                    continue

        # 4. 如果找遍了所有地方都没找到，抛出致命错误
        if merges is None:
            error_msg = "\n[Error] Critical: BPE vocab file not found!\nChecked the following paths:\n"
            error_msg += "\n".join([str(p) for p in candidate_paths])
            error_msg += "\nPlease ensure 'bpe_simple_vocab_16e6.txt' (or .gz) is inside 'models/clip/' directory."
            raise FileNotFoundError(error_msg)

        # ==================================================================
        # [修改结束] 下面是原始的数据处理逻辑，保持不变
        # ==================================================================

        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
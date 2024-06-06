from pathlib import Path


tokenizer_path = Path("/public/nijiahui/Meta-Llama-3-8B-Instruct-torch/tokenizer.model")

print(f"File name: {tokenizer_path.name}")
print(f"Parent directory:{tokenizer_path.parent}")

if tokenizer_path.exists():
    print("the file exists:")
else:
    print("The file does not exist:")
    
try:
    content = tokenizer_path.read_text()
    #print(content)
except FileExistsError:
    print("file nor found.")
    
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt

# 加载分词器模型路径
tokenizer_path = "/public/nijiahui/Meta-Llama-3-8B-Instruct-torch/tokenizer.model"
special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

# 测试分词器编码和解码功能
print(tokenizer.decode(tokenizer.encode("hello world!")))
print((tokenizer.encode("hello world!")))

'''
special_token_ids={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}
for token, token_id in special_token_ids.items():
    print(f"Special token: '{token}' -> ID: {token_id}")
'''
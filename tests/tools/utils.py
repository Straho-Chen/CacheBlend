import json
import collections
import string
import re
from rouge_score import rouge_scorer
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional


def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def get_repo_root(start_path: Optional[Path] = None) -> Path:
    """Find repository root by searching upward for common project markers.

    The function looks for these files/directories at or above the start_path:
    - .git (directory)
    - pyproject.toml
    - setup.py
    - requirements.txt

    If none are found it falls back to the workspace root (two parents up from this file)
    or the filesystem root.
    """
    if start_path is None:
        start_path = Path(__file__).resolve()

    cur = start_path if start_path.is_dir() else start_path.parent
    markers = {".git", "pyproject.toml", "setup.py", "requirements.txt", "README.md"}
    for p in [cur] + list(cur.parents):
        for m in markers:
            if (p / m).exists():
                return p

    # Fallback: choose repository top-level based on known layout (two parents up from tests/)
    # If that doesn't exist, return current working directory's root.
    probable_root = Path(__file__).resolve().parents[2]
    return probable_root if probable_root.exists() else Path.cwd()


# Export a module-level constant that other tests/tools can import.
REPO_ROOT: Path = get_repo_root()

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def parse_generation(s):
    s = s.lstrip('\n').split('\n')[0]
    if(len(s.split()) == 0):
        return s
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# For Mistral.
def build_qa_prompt(example, query_prompt):

    q = normalize_question(example["question"])
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    #ex_prompt = f"{docs_text}\n\nBased on these texts, answer the question:\nQ: {q}\nA:"
    #q_prompt = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: {q}\nAnswer:"
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt

# For deepseek
def build_qa_prompt_deepseek(example, query_prompt, think_marker=True):
    q = example["question"]
    doc_prompts = [f"<|User|>{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    if think_marker:
        print("think marker enabled")
        q_prompt = f"{query_prompt}{q}\nAnswer:<|Assistant|><think>\n"
    else:
        print("think marker disabled")
        q_prompt = f"{query_prompt}{q}\nAnswer:<|Assistant|></think>\n"
    return doc_prompts, q_prompt

def build_fewshot_prompt(example):
    q = "\n\n"+example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt

def build_fewshot_prompt_deepseek(example, think_marker=True):
    q = "\n\n"+example["question"]
    doc_prompts = [f"<|User|>{ctx['text']}\n\n" for ctx in example["ctxs"]]
    if think_marker:
        print("think marker enabled")
        q_prompt = f"{q}\nAnswer:<|Assistant|><think>\n"
    else:
        print("think marker disabled")
        q_prompt = f"{q}\nAnswer:<|Assistant|></think>\n"
    return doc_prompts, q_prompt

def compute_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    #gold_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_gold))])).tokens[4:-4]
    #pred_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_pred))])).tokens[4:-4]
    #pdb.set_trace()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_rl(pred, gold):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = scorer.score(gold, pred)['rougeL'].fmeasure
    return rougeL

def extract_after_think(text):
    marker = '</think>\n\n'
    index = text.find(marker)
    if index != -1:
        return text[index + len(marker):]
    else:
        return ''

# For qwen
def build_qa_prompt_qwen(example, query_prompt):
    q = example["question"]
    doc_prompts = [f"<|im_start|>user\n{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    q_prompt = f"{query_prompt}{q}\nAnswer:<|im_end|>\n<|im_start|>assistant\n"
    return doc_prompts, q_prompt

def build_fewshot_prompt_qwen(example):
    q = "\n\n"+example["question"]
    doc_prompts = [f"<|im_start|>user\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    q_prompt = f"{q}\nAnswer:<|im_end|>\n<|im_start|>assistant\n"
    return doc_prompts, q_prompt

# for normal LLMs
def gen_surrounding_tokens(model):
    # feel free to add you prompting templates here =)
    if model == "mistral":
        start="[INST]"
        end="[/INST]"
    elif model == "deepseek":
        start="<|User|>"
        end="<|Assistant|><think>\n"
    elif model == "deepseek-nothink":
        start="<|User|>"
        end="<|Assistant|><think>Okay, I think I have finished thinking.\n</think>\n"
    elif model == "qwen":
        start="<|im_start|>user\n"
        end="<|im_end|>\n<|im_start|>assistant\n"
    else:
        start=""
        end=""
    print(f"Using surrounding tokens: {start} ... {end}")
    return start, end

def build_qa_prompt_normal(model, prefix, example, query):
    q = normalize_question(example["question"])
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    q_prompt = f"{query}{q}\nAnswer:"
    start, end = gen_surrounding_tokens(model)
    p_prompt = f"{start}{prefix}"
    q_prompt = f"{q_prompt}{end}"
    return p_prompt, doc_prompts, q_prompt
    
def build_fewshot_prompt_normal(model, prefix, example):
    q="\n\n"+example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    start, end = gen_surrounding_tokens(model)
    p_prompt = f"{start}{prefix}"
    q_prompt = f"{q}{end}"
    return p_prompt, doc_prompts, q_prompt

def export_attention_matrices(cache_fuse_metadata, export_dir="./attn_exports", name_prefix=""):
    os.makedirs(export_dir, exist_ok=True)
    for layer_to_inspect in list(cache_fuse_metadata.get("hack_q", {}).keys()):
        print(f"Inspecting layer {layer_to_inspect}:")
        hq_list = cache_fuse_metadata["hack_q"].get(layer_to_inspect, [])
        hk_list = cache_fuse_metadata["hack_k"].get(layer_to_inspect, [])
        if len(hq_list) == 0 or len(hk_list) == 0:
            print(f"  no data for layer {layer_to_inspect}, skipping")
            continue

        # Concatenate collected rows. Each element was appended as a per-token
        # slice with shape [num_heads, head_size] (for query) or
        # [num_kv_heads, head_size] (for key). After cat we expect:
        #   hq_cat.shape == (num_tokens * num_heads, head_size)
        #   hk_cat.shape == (num_tokens_kv * num_kv_heads, head_size)
        hq_cat = torch.cat(hq_list, dim=0)
        hk_cat = torch.cat(hk_list, dim=0)

        # Read shape metadata written by xformers impl
        h_dim = cache_fuse_metadata.get("h_dim")
        meta_num_tokens = cache_fuse_metadata.get("num_tokens")
        num_kv_heads = cache_fuse_metadata.get("num_kv_heads")
        num_queries_per_kv = cache_fuse_metadata.get("num_queries_per_kv")
        scaling = cache_fuse_metadata.get("scaling")
        print(f"  metadata: h_dim={h_dim}, num_tokens={meta_num_tokens}, num_kv_heads={num_kv_heads}, num_queries_per_kv={num_queries_per_kv}, scaling={scaling}")

        if None in (h_dim, meta_num_tokens, num_kv_heads, num_queries_per_kv, scaling):
            print(f"  missing metadata for layer {layer_to_inspect}, skipping export")
            continue

        num_heads = int(num_kv_heads * num_queries_per_kv)

        # Infer token counts if cat sizes don't match metadata exactly
        # assume keys/queries correspond to same sequence length
        expected_hq_rows = int(meta_num_tokens * num_heads)
        expected_hk_rows = int(meta_num_tokens * num_kv_heads)
        assert hq_cat.shape[0] == expected_hq_rows, f"hq_cat shape {hq_cat.shape[0]} != expected {expected_hq_rows}"
        assert hk_cat.shape[0] == expected_hk_rows, f"hk_cat shape {hk_cat.shape[0]} != expected {expected_hk_rows}"

        q = hq_cat.view(meta_num_tokens, num_kv_heads, num_queries_per_kv, h_dim)
        k = hk_cat.view(meta_num_tokens, num_kv_heads, h_dim)

        # Now compute attention logits per head. We align GQA to per-head layout by
        # expanding k to match each query head when num_kv_heads != num_heads.
        Mq = q.shape[0]
        Mk = k.shape[0]

        # Save to file (CPU tensors)
        ts = np.datetime64("now").astype(str).replace(":", "-")
        fname = f"{name_prefix}attn_layer{layer_to_inspect}_{Mq}x{Mk}_{ts}.pt"
        out_path = os.path.join(export_dir, fname)
        torch.save({
            "meta": {
                "h_dim": h_dim,
                "num_tokens": meta_num_tokens,
                "num_kv_heads": num_kv_heads,
                "num_queries_per_kv": num_queries_per_kv,
                "num_heads": num_heads,
                "scaling": scaling,
                "Mq": Mq,
                "Mk": Mk,
            },
            "q": q,
            "k": k,
        }, out_path)
        print(f"  exported attention for layer {layer_to_inspect} to {out_path}")


def export_imp_indices(cache_fuse_metadata, export_dir="./imp_indices_exports", name_prefix=""):
    os.makedirs(export_dir, exist_ok=True)
    topk_num = cache_fuse_metadata.get("topk_num", None)
    imp_indices = cache_fuse_metadata.get("imp_indices", None)
    if imp_indices is None or topk_num is None:
        print("No important indices found, skipping export")
        return
    ts = np.datetime64("now").astype(str).replace(":", "-")
    fname = f"{name_prefix}imp_indices_{ts}.pt"
    out_path = os.path.join(export_dir, fname)
    print(f"topk_num: {topk_num}")
    print(f"imp_indices: {imp_indices}")
    torch.save({
        "topk_num": topk_num,
        "imp_indices": imp_indices,
    }, out_path)
    print(f"Exported important indices to {out_path}")
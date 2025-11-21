from vllm import LLM, SamplingParams
import torch
import numpy as np
from transformers import AutoTokenizer
from utils.utils import REPO_ROOT, load_dataset, build_qa_prompt_normal, compute_f1, extract_after_think, export_attention_matrices
from itertools import chain
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run cache-fuse blending test for musique dataset")
parser.add_argument("--model-size", dest="model_size", type=str, default="7B")
parser.add_argument("--enable-think", dest="enable_think", action="store_true", help="Whether to enable think marker in DeepSeek")
parser.add_argument("--export-dir", dest="export_dir", type=str, default=None, help="Directory to export attention matrices (default: ./attn_exports_deepseek{model_size}_musique)")
args = parser.parse_args()

eval_dataset = load_dataset(f"{REPO_ROOT}/inputs/musique_s.json")

test_model_7B="/workspaces/modelscope-yrcache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
test_model_14B="/workspaces/modelscope-yrcache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

if args.model_size == "7B":
    print("Using 7B model, think mode:", args.enable_think)
    test_model = test_model_7B
else:
    print("Using 14B model, think mode:", args.enable_think)
    test_model = test_model_14B

# Set default export directory if not provided
if args.export_dir is None:
    args.export_dir = f"./attn_exports_deepseek{args.model_size}_musique"

llm = LLM(model=test_model, gpu_memory_utilization=0.95, dtype=torch.bfloat16, max_model_len=20000,
          #tokenizer=tokenizer,
          )
tokenizer = AutoTokenizer.from_pretrained(test_model)
llm.set_tokenizer(tokenizer)

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

# Create export directory
os.makedirs(args.export_dir, exist_ok=True)
print(f"Exporting attention matrices to: {args.export_dir}")

sample = 0

for ex in eval_dataset:
    sample += 1
    answers = ex["answers"]
    if args.enable_think:
        p_promt, doc_prompts, q_prompt = build_qa_prompt_normal("deepseek", prefix_prompt, ex, query_prompt)
    else:
        p_promt, doc_prompts, q_prompt = build_qa_prompt_normal("deepseek-nothink", prefix_prompt, ex, query_prompt)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]
    p_ids = tokenizer.encode(p_promt)[1:]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Create an tokenizer and LLM.
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata

    s_start_len = len(p_ids) + 1

    s_start = []
    s_start_1_len = len(s_start) + 1

    s_start_prefix = [151646]

    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [p_ids] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids]

    # export chunk attention matrix
    last_len = len(q_ids)

    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['attn_bias'] = None
    cache_fuse_metadata["hack_start"] = True
    chunk_past_key_values = []
    shift = 0
    # Concatenate old KVs
    print("len of doc_chunk_ids:", len(doc_chunk_ids))
    for i in range(len(doc_chunk_ids)):
        cache_fuse_metadata["hack_q"] = {}
        cache_fuse_metadata["hack_k"] = {}
        doc_chunk_ids_full = s_start_prefix + doc_chunk_ids[i]
        llm.generate(None, sampling_params, prompt_token_ids=[doc_chunk_ids_full])
        shift += len(doc_chunk_ids[i])
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        num_layer = len(llm_layers)
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            if i == 0:
                temp_k = past_key_values[0][:s_start_len].clone() # do not chage with s_start_1
                temp_v = past_key_values[1][:s_start_len].clone()
            else:
                temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
                temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()    

            if i == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=0)
            llm_layers[j].self_attn.hack_kv = None
        export_attention_matrices(cache_fuse_metadata, name_prefix=f"chunk{i}_", export_dir=args.export_dir)
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values

    # Export prefill attention matrix
    input_ids = []

    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = s_start_prefix+ doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
        
    input_prompt = tokenizer.decode(input_ids)

    # for full prefill
    print("Running full prefill to export attention matrices...")
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["hack_start"] = True
    cache_fuse_metadata["hack_q"] = {}
    cache_fuse_metadata["hack_k"] = {}
    output = llm.generate([input_prompt], sampling_params)
    export_attention_matrices(cache_fuse_metadata, name_prefix=f"prefill_", export_dir=args.export_dir)
    print("------------")
    
    # Only process first sample for attention matrix export
    break

print("---------------Attention Matrix Export Complete---------------------")
print(f"Attention matrices exported to: {args.export_dir}")


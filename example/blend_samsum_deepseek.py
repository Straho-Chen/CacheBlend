from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, normalize_question, build_fewshot_prompt_deepseek, compute_rl, extract_after_think
from pathlib import Path
from itertools import chain
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run cache-fuse blending test for samsum dataset")
parser.add_argument("--model-size", dest="model_size", type=str, default="7B")
parser.add_argument("--enable-think", dest="enable_think", action="store_true", help="Whether to enable think marker in DeepSeek")
args = parser.parse_args()

eval_dataset = load_dataset("inputs/samsum.json")

test_model_7B="/mnt/nvme0n1/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
test_model_14B="/mnt/nvme0n1/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

if args.model_size == "7B":
    print("Using 7B model, think mode:", args.enable_think)
    test_model = test_model_7B
else:
    print("Using 14B model, think mode:", args.enable_think)
    test_model = test_model_14B

llm = LLM(model=test_model, gpu_memory_utilization=0.95, dtype=torch.bfloat16, max_model_len=20000,
          #tokenizer=tokenizer,
          )
tokenizer = AutoTokenizer.from_pretrained(test_model)
llm.set_tokenizer(tokenizer)

prefix_prompt = "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"

ttft_blend = []
ttft_full_reuse = []
ttft_full_prefill = []
rl_blend = []
rl_full_reuse = []
rl_full_prefill = []

max_ctx_len = 3400
#TODO (Jiayi): fix filler tokens at the begining or pass in tokenizer
for sample_idx, ex in enumerate(eval_dataset):
    answers = ex["answers"]
    if args.enable_think:
        doc_prompts, q_prompt = build_fewshot_prompt_deepseek(ex, True)
    else:
        doc_prompts, q_prompt = build_fewshot_prompt_deepseek(ex, False)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    
    # drop last few-shot examples if exceeding max_ctx_len
    while len(list(chain.from_iterable(doc_chunk_ids))) > max_ctx_len:
        del_idx = int(len(doc_chunk_ids)/2)
        del doc_chunk_ids[del_idx]
    
    # skip if all ctxs are dropped
    if len(doc_chunk_ids)==0:
        continue
                
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Create an tokenizer and LLM.
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['check'] = False
    cache_fuse_metadata['attn_bias'] = None

    s_start_full = tokenizer.encode(prefix_prompt)[1:]
    s_start_len = len(s_start_full) + 1

    s_start = []
    s_start_1_len = len(s_start) + 1

    s_start_prefix = [151646]

    s_end = []
    s_end_len = len(s_end)

    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids+s_end]

    last_len = len(q_ids+s_end)

    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False
    chunk_past_key_values = []
    shift = 0
    # Concatenate old KVs
    for i in range(len(doc_chunk_ids)):
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
                #pdb.set_trace()
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=0)
            llm_layers[j].self_attn.hack_kv = None
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
    
    input_ids = []

    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = s_start_prefix+ doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
        
    input_prompt = tokenizer.decode(input_ids)

    # for blend
    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['recomp_ratio'] = 0.2
    cache_fuse_metadata['fast_attention'] = True
    cache_fuse_metadata['suffix_len'] = last_len
    output = llm.generate(None, sampling_params, prompt_token_ids=[input_ids])
    res = output[0].outputs[0].text
    res = extract_after_think(res)
    # TODO(Jiayi): please move this to utils
    res = res.lstrip('\n').split('\n')[0]
    print(f"cache generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"sample: {sample_idx}, TTFT: {ttft}")
    ttft_blend.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers])
    rl_blend.append(rl)

    # for full reuse
    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['recomp_ratio'] = 0.0
    cache_fuse_metadata['fast_attention'] = True
    cache_fuse_metadata['suffix_len'] = last_len
    output = llm.generate(None, sampling_params, prompt_token_ids=[input_ids])
    res = output[0].outputs[0].text
    res = extract_after_think(res)
    # TODO(Jiayi): please move this to utils
    res = res.lstrip('\n').split('\n')[0]
    print(f"full reuse generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"sample: {sample_idx}, TTFT: {ttft}")
    ttft_full_reuse.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers])
    rl_full_reuse.append(rl)
    
    # for full prefill
    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params)
    res = output[0].outputs[0].text
    res = extract_after_think(res)
    res = res.lstrip('\n').split('\n')[0]
    print(f"full prefill generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"sample: {sample_idx}, TTFT: {ttft}")
    ttft_full_prefill.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers])
    rl_full_prefill.append(rl)
    print("------------")
    

print("---------------Result Summary---------------------")
print(f"Avg TTFT with cache: {np.mean(ttft_blend)}")
print(f"Avg TTFT with full reuse: {np.mean(ttft_full_reuse)}")
print(f"Avg TTFT with full prefill: {np.mean(ttft_full_prefill)}")
print(f"Avg F1 with cache: {np.mean(rl_blend)}")
print(f"Avg F1 with full reuse: {np.mean(rl_full_reuse)}")
print(f"Avg F1 with full prefill: {np.mean(rl_full_prefill)}")

# print(f"TTFT with cache: {np.mean(ttft_blend)}")
# print(f"TTFT with full prefill: {np.mean(ttft_full)}")
# print(f"rl with cache: {np.mean(rl_blend)}")
# print(f"rl with full prefill: {np.mean(rl_full)}")
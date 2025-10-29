from vllm import LLM, SamplingParams
import torch
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, build_qa_prompt_normal, compute_f1, extract_after_think
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run cache-fuse blending test for musique dataset")
parser.add_argument("--model-size", dest="model_size", type=str, default="7B")
parser.add_argument("--enable-think", dest="enable_think", action="store_true", help="Whether to enable think marker in DeepSeek")
args = parser.parse_args()

eval_dataset = load_dataset("inputs/musique_s.json")

test_model_7B="/workspaces/modelscope-yrcache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
test_model_14B="/workspaces/modelscope-yrcache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

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

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

ttft_blend = []
ttft_full_reuse = []
ttft_full_prefill = []
f1_blend = []
f1_full_reuse = []
f1_full_prefill = []

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

    #import pdb
    #pdb.set_trace()
    
    #while len(list(chain.from_iterable(doc_chunk_ids))) > max_ctx_len:
    #    del_idx = len(doc_chunk_ids)-1
    #    del doc_chunk_ids[del_idx]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Create an tokenizer and LLM.
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['check'] = False

    s_start_len = len(p_ids) + 1

    #s_start = [518, 25580, 29962]
    s_start = []
    s_start_1_len = len(s_start) + 1

    s_start_prefix = [151646]

    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [p_ids] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids]

    last_len = len(q_ids)

    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False
    chunk_past_key_values = []
    
    # Concatenate old KVs
    for i in range(len(doc_chunk_ids)):
        doc_chunk_ids_full = s_start_prefix + doc_chunk_ids[i]
        llm.generate(None, sampling_params, prompt_token_ids=[doc_chunk_ids_full])

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

    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
        
    input_ids = []

    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = s_start_prefix+doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
        
    # print(len(input_ids))
        
    input_prompt = tokenizer.decode(input_ids)

    # for blend
    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['suffix_len'] = last_len
    cache_fuse_metadata['recomp_ratio'] = 0.2
    output = llm.generate(None, sampling_params, prompt_token_ids=[input_ids])
    res = output[0].outputs[0].text
    # print(f"Raw generation: {res}")
    # res = extract_after_think(res)
    print(f"blend generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"sample: {sample}, TTFT: {ttft}")
    ttft_blend.append(ttft)
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    f1_blend.append(f1)

    # for full reuse
    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['suffix_len'] = last_len
    cache_fuse_metadata['recomp_ratio'] = 0.0
    output = llm.generate(None, sampling_params, prompt_token_ids=[input_ids])
    res = output[0].outputs[0].text
    # print(f"Raw generation: {res}")
    # res = extract_after_think(res)
    print(f"full reuse generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"sample: {sample}, TTFT: {ttft}")
    ttft_full_reuse.append(ttft)
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    f1_full_reuse.append(f1)

    
    # for full prefill
    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = False
    output = llm.generate(None, sampling_params, prompt_token_ids=[input_ids])
    res = output[0].outputs[0].text
    # print(f"Raw generation: {res}")
    # res = extract_after_think(res)
    print(f"full prefill generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"sample: {sample}, TTFT: {ttft}")
    ttft_full_prefill.append(ttft)
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    f1_full_prefill.append(f1)
    print("------------")

print("---------------Result Summary---------------------")
print(f"Avg TTFT with cache: {np.mean(ttft_blend)}")
print(f"Avg TTFT with full reuse: {np.mean(ttft_full_reuse)}")
print(f"Avg TTFT with full prefill: {np.mean(ttft_full_prefill)}")
print(f"Avg F1 with cache: {np.mean(f1_blend)}")
print(f"Avg F1 with full reuse: {np.mean(f1_full_reuse)}")
print(f"Avg F1 with full prefill: {np.mean(f1_full_prefill)}")
from vllm import LLM, SamplingParams
import torch
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, build_fewshot_prompt_normal, compute_rl
from itertools import chain

eval_dataset = load_dataset("inputs/samsum.json")

test_model="/mnt/nvme0n1/modelscope/Qwen/Qwen2.5-3B-Instruct"

llm = LLM(model=test_model, gpu_memory_utilization=0.95,
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
    p_prompt, doc_prompts, q_prompt = build_fewshot_prompt_normal("qwen", prefix_prompt, ex)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]
    p_ids = tokenizer.encode(p_prompt)[1:]
    
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

    s_start_len = len(p_ids)

    s_start = []
    s_start_1_len = len(s_start)

    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [p_ids] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids]

    last_len = len(q_ids)

    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False
    chunk_past_key_values = []
    shift = 0
    # Concatenate old KVs
    for i in range(len(doc_chunk_ids)):
        prompts = [tokenizer.decode(doc_chunk_ids[i])]
        llm.generate(prompts, sampling_params)
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
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len:]
        input_ids += temp_ids
        
    input_prompt = tokenizer.decode(input_ids)

    # for blend
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['suffix_len'] = last_len
    cache_fuse_metadata['recomp_ratio'] = 0.2
    output = llm.generate([input_prompt], sampling_params)
    res = output[0].outputs[0].text
    res = res.lstrip('\n').split('\n')[0]
    print(f"blend generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"sample: {sample_idx}, TTFT: {ttft}")
    ttft_blend.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers])
    rl_blend.append(rl)

    # for full reuse
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['suffix_len'] = last_len
    cache_fuse_metadata['recomp_ratio'] = 0.0
    output = llm.generate([input_prompt], sampling_params)
    res = output[0].outputs[0].text
    res = res.lstrip('\n').split('\n')[0]
    print(f"full reuse generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"sample: {sample_idx}, TTFT: {ttft}")
    ttft_full_reuse.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers])
    rl_full_reuse.append(rl)

    # for full prefill
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params)
    res = output[0].outputs[0].text
    res = res.lstrip('\n').split('\n')[0]
    print(f"full prefill generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"sample: {sample_idx}, TTFT: {ttft}")
    ttft_full_prefill.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers])
    rl_full_prefill.append(rl)
    print("------------")
    

print("---------------Result Summary---------------------")
# print(f"TTFT with cache: {np.mean(ttft_blend)}")
# print(f"TTFT with full prefill: {np.mean(ttft_full)}")
# print(f"rl with cache: {np.mean(rl_blend)}")
# print(f"rl with full prefill: {np.mean(rl_full)}")
print(f"Avg TTFT with cache: {np.mean(ttft_blend)}")
print(f"Avg TTFT with full reuse: {np.mean(ttft_full_reuse)}")
print(f"Avg TTFT with full prefill: {np.mean(ttft_full_prefill)}")
print(f"Avg F1 with cache: {np.mean(rl_blend)}")
print(f"Avg F1 with full reuse: {np.mean(rl_full_reuse)}")
print(f"Avg F1 with full prefill: {np.mean(rl_full_prefill)}")
from transformers import AutoTokenizer

# test_model_7B="/mnt/nvme0n1/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# test_model_14B="/mnt/nvme0n1/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# test_model = "/mnt/nvme0n1/modelscope/Qwen/Qwen2.5-3B-Instruct"
test_model = "/mnt/nvme0n1/modelscope/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(test_model)

e=tokenizer.encode("[INST]")
print(e)

e=tokenizer.encode("[/INST]")
print(e)
# d=tokenizer.decode(e)
# print(d)

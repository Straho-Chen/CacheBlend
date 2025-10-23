from transformers import AutoTokenizer

test_model_7B="/mnt/nvme0n1/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
test_model_14B="/mnt/nvme0n1/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

test_model = test_model_7B

tokenizer = AutoTokenizer.from_pretrained(test_model)

print(tokenizer.encode("<"))
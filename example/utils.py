import json
import collections
import string
import re
from rouge_score import rouge_scorer

def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)

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
    elif model == "qwen":
        start="<|im_start|>user\n"
        end="<|im_end|>\n<|im_start|>assistant\n"
    else:
        start=""
        end=""
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
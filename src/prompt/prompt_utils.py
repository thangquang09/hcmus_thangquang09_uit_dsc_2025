import random, json
from collections import defaultdict
from src.utils.constants import LABELS


def sample_fewshots(fewshot_data, k=5, seed=None):
    if k < len(LABELS):
        raise ValueError(f"k={k} phải >= {len(LABELS)}")
    by_label = defaultdict(list)
    for ex in fewshot_data:
        lab = ex.get("label")
        if lab in LABELS: by_label[lab].append(ex)
    rng = random.Random(seed)
    selected = [rng.choice(by_label[l]) for l in LABELS]
    pool = [ex for ex in fewshot_data if id(ex) not in map(id, selected)]
    rng.shuffle(pool)
    selected += pool[:(k-len(LABELS))]
    rng.shuffle(selected)
    return selected


def normalize_label(s):
    s = (s or "").strip().lower()
    for lab in LABELS:
        if s.startswith(lab) or lab in s: return lab
    return "no"


def build_few_shots(fewshot_data):
    FEWSHOT = "Context: {context}\n\nPrompt: {prompt}\n\nResponse: {response}\n\nLabel: {label}\n\nExplanation: {explanation}\n\n"
    return "".join([
        FEWSHOT.format(
            context=d["context"], prompt=d["prompt"],
            response=d["generated_response"], label=d["label"],
            explanation=d["explanation"]
        ) for d in fewshot_data
    ])


def build_user_msg_train(context, prompt, response, fewshot_data):
    INSTRUCTION = """You are a hallucination detection classifier for Vietnamese language models. 
Your task is to classify the RESPONSE into exactly ONE label from {no, intrinsic, extrinsic}, 
based ONLY on the given CONTEXT and PROMPT. 
You must NEVER use knowledge outside the provided CONTEXT.

Label Definitions:
- no: RESPONSE is fully supported by CONTEXT, with no added or fabricated content. 
       Allowed to reject false assumptions in PROMPT if CONTEXT shows they are wrong.
- intrinsic: RESPONSE contradicts, reverses, or distorts facts from CONTEXT. 
             This includes repeating false assumptions from PROMPT that conflict with CONTEXT.
- extrinsic: RESPONSE adds new information not grounded in CONTEXT and not directly verifiable from it, 
             without explicit contradiction.

Classification Rules:
1) If RESPONSE both contradicts CONTEXT AND adds unsupported info → intrinsic (contradiction takes priority).
2) Match at semantic level; ignore minor spelling or grammatical errors.
3) If PROMPT contains false assumptions and RESPONSE accepts/repeats them against CONTEXT → intrinsic.
4) If RESPONSE only says “insufficient / not enough information” (without fabricating) → no.
5) Output must be EXACTLY one word: no | intrinsic | extrinsic

Evaluation Order:
1. First check for contradictions with CONTEXT → intrinsic
2. If no contradiction, check for unsupported additions → extrinsic
3. If fully supported with no addition → no"""  # giữ nguyên phần gốc
    FEWSHOT = "EXAMPLE CLASSIFICATION:\n\n\n" + build_few_shots(fewshot_data)
    return (
        INSTRUCTION + "\n\n" + FEWSHOT +
        f"\n\nPlease classify the following:\n\nContext: {context}\n\nPrompt: {prompt}\n\nResponse: {response}\nLabel:"
    )


def build_assistant_msg_train(label):
    return normalize_label(label)

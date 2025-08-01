"""
Call this script after the initial model call with prompt PAQ_prompt_paraphrase_search.txt
beginning of string: "Paraphrases:\n"
separators: \n
"""
from typing import List, Tuple
import json

log_path = "/home/tmp/"
beginning_tokens = []
separator_tokens = [198, 715, 2303, 5872, 271, 4710, 18611]

def get_logprobs(prompt_token_ids: List[int]) -> List[Tuple[int, float]]:
    tokens_hash = hash(tuple(prompt_token_ids))
    file_path = f"{log_path}{tokens_hash}.logprobs"
    logprobs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            key, value = line.strip().split("\t")
            logprobs.append((int(key), float(value)))
    return logprobs

def get_output_token_ids(prompt_token_ids: List[int]) -> List[int]:
    log_filename = hash(tuple(prompt_token_ids))
    file_path = f"{log_path}{log_filename}.output_token_ids"
    return json.load(open(file_path, "r", encoding="utf-8"))


def get_paraphrase_scores(output_token_ids: List[int], output_token_logprobs: List[Tuple[int, float]])\
        -> Tuple[List[List[int]], List[float]]:
    if output_token_ids[:len(output_token_ids)] == output_token_ids:
        output_token_ids = output_token_ids[len(output_token_ids):]

    assert len(output_token_ids) == len(output_token_logprobs)

    scores = []
    paraphrases_output_tokens = []
    score = 0
    paraphrase = []
    for idx, token in enumerate(output_token_ids):
        if token not in separator_tokens:
            assert token == output_token_logprobs[idx][0]
            score += output_token_logprobs[idx][1]
            paraphrase.append(token)
        else:
            scores.append(score)
            paraphrases_output_tokens.append(paraphrase)
            score = 0

    return paraphrases_output_tokens, scores

def get_best_paraphrase(prompt_token_ids):
    output_token_logprobs = get_logprobs(prompt_token_ids)
    output_token_ids = get_output_token_ids(prompt_token_ids)
    paraphrases_output_tokens, scores = get_paraphrase_scores(output_token_ids, output_token_logprobs)
    argmax_paraphrase = max(range(len(scores)), key=lambda i: scores[i])
    return paraphrases_output_tokens[argmax_paraphrase]

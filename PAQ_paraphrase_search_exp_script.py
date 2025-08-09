import requests
import json
import argparse
from typing import List
from transformers import AutoTokenizer
from itertools import islice

log_path = "/home/tmp/"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

def parse_line(line):
    line_id, text = tuple(line.strip().split('\t'))
    question, answer = tuple(text.split(' Answer: '))
    return line_id, question, answer

def get_chat_answer(question, prompt, URL):
    data = {"question": question,
            "prompt": prompt,
            "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
    # this should be run with FM index disabled
    response = requests.post(f"{URL}/chat", json=data)
    response_jsn = json.loads(response.text)
    system_answer = response_jsn['answer']
    return system_answer

def get_tokens_hash(question, prompt, URL):
    data = {"question": question, "prompt": prompt}
    response = requests.post(f"{URL}/chat_get_prompt_token_ids", json=data)
    response_jsn = json.loads(response.text)
    prompt_token_ids = response_jsn['prompt_token_ids']
    tokens_hash = hash(tuple(prompt_token_ids))
    return tokens_hash

def get_chat_output(question, prompt, URL):
    data = {"question": question,
            "prompt": prompt,
            "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
    response = requests.post(f"{URL}/chat_get_output", json=data)
    response_jsn = json.loads(response.text)['output']
    return response_jsn

def get_beginnings(question, prompt, URL):
    # recover input_token_ids to get hash
    tokens_hash = get_tokens_hash(question, prompt, URL)
    with open(f"{log_path}{tokens_hash}.beginnings", 'r') as beginnings_fh:
        beginnings = beginnings_fh.read().strip().split('\n')
    return beginnings

def get_logprobs(tokens_hash) -> List[float]:
    file_path = f"{log_path}{tokens_hash}.logprobs"
    logprobs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            key, value = line.strip().split("\t")
            logprobs.append(float(value))
    return logprobs

def read_in_batches(filename, batch_size):
    with open(filename, 'r') as file:
        while True:
            batch = [parse_line(line) for line in islice(file, batch_size)]
            if not batch:
                break
            yield batch

def run_stage_1(FILE_I, URL):
    with open("PAQ_prompt_paraphrase_search.txt", 'r') as fh:
        prompt = fh.read().strip()
    with open(FILE_I, newline='', encoding='utf-8') as in_file:
        for line in in_file:
            line_id, question, answer = parse_line(line)
            system_answer = get_chat_answer(question, prompt, URL)
            tokens_hash = get_tokens_hash(question, prompt, URL)
            with open(f"{log_path}{tokens_hash}.beginnings", 'w') as out_file:
                out_file.write(system_answer)

def run_stage_2(FILE_I, URL):
    with open("PAQ_prompt_paraphrase_search.txt", 'r') as fh:
        paraphrase_prompt = fh.read().strip()

    with open(FILE_I, newline='', encoding='utf-8') as in_file:
        for line in in_file:
            line_id, question, answer = parse_line(line)
            beginnings = get_beginnings(question, paraphrase_prompt, URL)

            # this should be run with FM index enabled
            for beginning in beginnings:
                prompt = f'Paraphrase this sentence in lowercase starting with "{beginning}":'
                response_jsn = get_chat_output(question, prompt, URL)
                beginning_tokens_hash = hash(tuple(response_jsn['prompt_token_ids']))
                with open(f"{log_path}{beginning_tokens_hash}.output_token_ids", 'w') as beginning_tokens_fh:
                    json.dump(response_jsn['output_token_ids'], beginning_tokens_fh)

def run_stage_3(FILE_I, FILE_O, URL):
    with open("PAQ_prompt_paraphrase_search.txt", 'r') as fh:
        paraphrase_prompt = fh.read().strip()

    with open("PAQ_prompt_repeat.txt", 'r') as fh:
        repeat_prompt = fh.read().strip()

    with open(FILE_I, 'r', newline='', encoding='utf-8') as in_file:
        with open(FILE_O, 'w', newline='', encoding='utf-8') as out_file:
            for line in in_file:
                line_id, question, answer = parse_line(line)
                beginnings = get_beginnings(question, paraphrase_prompt, URL)

                beginning_tokens_lst = []
                beginning_logprobs_lst = []
                for beginning in beginnings:
                    prompt = f'Paraphrase this sentence starting with "{beginning}":'
                    beginning_tokens_hash = get_tokens_hash(question, prompt, URL)

                    with open(f"{log_path}{beginning_tokens_hash}.output_token_ids", 'r') as beginning_tokens_fh:
                        beginning_tokens = json.load(beginning_tokens_fh)
                        beginning_tokens_lst.append(beginning_tokens)

                    beginning_logprobs = get_logprobs(beginning_tokens_hash)
                    beginning_logprobs_lst.append(beginning_logprobs)

                beginning_scores = [sum(beginning_logprobs) for beginning_logprobs in beginning_logprobs_lst]
                argmax_paraphrase = max(range(len(beginning_scores)), key=lambda i: beginning_scores[i])
                best_paraphrase = beginning_tokens_lst[argmax_paraphrase]

                # find the most successful paraphrase
                # better_question_tokens = get_best_paraphrase(prompt_token_ids)
                # better_question = tokenizer.decode(better_question_tokens)
                better_question = tokenizer.decode(best_paraphrase)

                # this should be run with FM index enabled
                # repeat the best paraphrase and answer the question
                system_answer = get_chat_answer(better_question, repeat_prompt, URL)
                out_file.write(line_id + '\t' + system_answer + '\n')

def run_stage_1_batch(FILE_I, URL, batch_size):
    with open("PAQ_prompt_paraphrase_search.txt", 'r') as fh:
        prompt = fh.read().strip()

    for batch in read_in_batches(FILE_I, batch_size):
        questions = [question for _, question, _ in batch]
        data = {"questions": questions,
                "prompt": prompt,
                "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
        # this should be run with FM index disabled
        response = requests.post(f"{URL}/chat_batch", json=data)
        response_jsn = json.loads(response.text)
        system_answers = response_jsn['answers']
        for ind, system_answer in enumerate(system_answers):
            # system_answer = system_answer.replace('\n', '')
            # if '</think>' in system_answer:
            #     system_answer = system_answer.split('</think>')[-1]

            tokens_hash = get_tokens_hash(questions[ind], prompt, URL)
            with open(f"{log_path}{tokens_hash}.beginnings", 'w') as out_file:
                out_file.write(system_answer)

def run_stage_2_batch(FILE_I, URL, batch_size):
    with open("PAQ_prompt_paraphrase_search.txt", 'r') as fh:
        paraphrase_prompt = fh.read().strip()

    with open(FILE_I, newline='', encoding='utf-8') as in_file:
        for line in in_file:
            line_id, question, answer = parse_line(line)
            beginnings = get_beginnings(question, paraphrase_prompt, URL)

            # this should be run with FM index enabled
            for beginning in beginnings:
                prompt = f'Paraphrase this sentence starting with "{beginning}":'
                response_jsn = get_chat_output(question, prompt, URL)
                beginning_tokens_hash = hash(tuple(response_jsn['prompt_token_ids']))
                with open(f"{log_path}{beginning_tokens_hash}.output_token_ids", 'w') as beginning_tokens_fh:
                    json.dump(response_jsn['output_token_ids'], beginning_tokens_fh)


def run_experiment(FILE_I, FILE_O, stage, URL):
    if stage == 1:
        run_stage_1(FILE_I, URL)
    elif stage == 2:
        run_stage_2(FILE_I, URL)
    elif stage == 3:
        run_stage_3(FILE_I, FILE_O, URL)

def run_experiment_batch(FILE_I, FILE_O, stage, URL, batch_size):
    if stage == 1:
        run_stage_1_batch(FILE_I, URL, batch_size)
    elif stage == 2:
        # run_stage_2_batch(FILE_I, URL, batch_size)
        run_stage_2(FILE_I, URL, batch_size)
    elif stage == 3:
        # run_stage_3_batch(FILE_I, FILE_O, URL, batch_size)
        run_stage_3(FILE_I, FILE_O, URL, batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PAQ experiment")
    parser.add_argument('--input', '-i', required=True, help='Path to input TSV file')
    parser.add_argument('--output', '-o', required=True, help='Path to output TSV file')
    parser.add_argument('--stage', '-s', required=True, help='Experiment stage')
    parser.add_argument('--url', '-u', default='http://127.0.0.1:8001/', help='URL for the API endpoint')
    parser.add_argument('--batch', '-b', default=1, help='Batch size')

    args = parser.parse_args()
    args.batch = int(args.batch)

    assert args.batch > 0
    if args.batch == 1:
        run_experiment(args.input, args.output, int(args.stage), args.url)
    else:
        run_experiment_batch(args.input, args.output, int(args.stage), args.url, args.batch)



# modify llm.py in vllm to add a function that returns input_token_ids
# add get_output method to model api
# add get_chat_input_token_ids to model api



# (venv) root@pminervini-lm-eval-x8gfg-qmlzl:/home/fm-index-constrained-decoding# cat /home/tmp/8640584250548683661.output_token_ids
# [15191, 572, 4767, 518, 279, 882, 279, 1156, 5139, 72752, 19724, 13316, 57085, 30, 151645]
# [15191, 572, 4767, 518, 279, 882, 279, 1156, 5139, 72752, 19724, 13316, 57085, 30, 151645]

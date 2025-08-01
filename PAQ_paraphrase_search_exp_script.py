import requests
import json
import argparse
from paraphrase_search import get_best_paraphrase
from transformers import AutoTokenizer

log_path = "/home/tmp/"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

def run_stage_1(FILE_I, URL):
    with open("PAQ_prompt_paraphrase_search.txt", 'r') as fh:
        prompt = fh.read().strip()
    with open(FILE_I, newline='', encoding='utf-8') as in_file:
        for line in in_file:
            line_id, _, text = tuple(line.strip().split('\t'))
            question, answer = tuple(text.split(' Answer: '))
            data = {"question": question,
                    "prompt": prompt,
                    "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
            response = requests.post(f"{URL}/chat_get_output", json=data)
            response_jsn = json.loads(response.text)
            # TODO: match this to output structure
            tokens_hash = hash(tuple(response_jsn['prompt_token_ids']))
            with open(f"{log_path}{tokens_hash}.output_token_ids", 'w') as out_file:
                json.dump(response_jsn['output_token_ids'], out_file)

def run_stage_2(FILE_I, FILE_O, URL):
    with open("PAQ_prompt_paraphrase_search.txt", 'r') as fh:
        paraphrase_prompt = fh.read().strip()

    with open("PAQ_prompt_repeat.txt", 'r') as fh:
        repeat_prompt = fh.read().strip()

    with open(FILE_I, newline='', encoding='utf-8') as in_file:
        with open(FILE_O, 'w', newline='', encoding='utf-8') as out_file:
            for line in in_file:
                line_id, _, text = tuple(line.strip().split('\t'))
                question, answer = tuple(text.split(' Answer: '))

                # recover input_token_ids to get hash
                data = {"question": question, "prompt": paraphrase_prompt}
                response = requests.post(f"{URL}/get_chat_input_token_ids", json=data)
                response_jsn = json.loads(response.text)
                prompt_token_ids = response_jsn['prompt_token_ids']

                # find the most successful paraphrase
                better_question_tokens = get_best_paraphrase(prompt_token_ids)
                better_question = tokenizer.decode(better_question_tokens)

                # repeat the best paraphrase and answer the question
                data = {"question": better_question,
                        "prompt": repeat_prompt,
                        "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
                response = requests.post(f"{URL}/chat", json=data)
                response_jsn = json.loads(response.text)
                system_answer = response_jsn['answer']
                out_file.write(line_id + '\t' + system_answer + '\n')

def run_experiment(FILE_I, FILE_O, stage, URL):
    if stage == 1:
        run_stage_1(FILE_I, URL)
    elif stage == 2:
        run_stage_2(FILE_I, FILE_O, URL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PAQ experiment")
    parser.add_argument('--input', '-i', required=True, help='Path to input TSV file')
    parser.add_argument('--output', '-o', required=True, help='Path to output TSV file')
    parser.add_argument('--stage', '-s', required=True, options=[1,2], help='Experiment stage')
    parser.add_argument('--url', '-u', default='http://127.0.0.1:8001/', help='URL for the API endpoint')

    args = parser.parse_args()

    run_experiment(args.input, args.output, args.stage, args.url)

# modify llm.py in vllm to add a function that returns input_token_ids
# add get_output method to model api
# add get_chat_input_token_ids to model api

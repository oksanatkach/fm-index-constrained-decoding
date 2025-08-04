import requests
import json
import argparse
from itertools import islice

def read_in_batches(filename, batch_size):
    with open(filename, 'r') as file:
        while True:
            batch = [parse_line(line) for line in islice(file, batch_size)]
            if not batch:
                break
            yield batch

def parse_line(line):
    line_id, _, text = tuple(line.strip().split('\t'))
    question, answer = tuple(text.split(' Answer: '))
    return line_id, question, answer

def run_experiment(FILE_I, FILE_O, prompt_file_path, URL):
    with open(prompt_file_path, 'r') as fh:
        prompt = fh.read().strip()
    with open(FILE_I, newline='', encoding='utf-8') as in_file:
        with open(FILE_O, 'w', newline='', encoding='utf-8') as out_file:
            for line in in_file:
                line_id, _, text = tuple(line.strip().split('\t'))
                question, answer = tuple(text.split(' Answer: '))
                data = {"question": question,
                        "prompt": prompt,
                        "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
                response = requests.post(f"{URL}/chat", json=data)
                response_jsn = json.loads(response.text)
                system_answer = response_jsn['answer']
                out_file.write(line_id + '\t' + system_answer + '\n')

def run_experiment_batch(FILE_I, FILE_O, prompt_file_path, URL):
    with open(prompt_file_path, 'r') as fh:
        prompt = fh.read().strip()
        with open(FILE_O, 'w', newline='', encoding='utf-8') as out_file:
            for batch in read_in_batches(FILE_I, 100):

                questions = [question for _, question, _ in batch]

                data = {"questions": questions,
                        "prompt": prompt,
                        "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
                response = requests.post(f"{URL}/chat_batch", json=data)
                response_jsn = json.loads(response.text)
                system_answers = response_jsn['answers']
                for ind, system_answer in enumerate(system_answers):
                    out_file.write(batch[ind][0] + '\t' + system_answer + '\n')

if __name__ == '__main__':
    # URL = "http://127.0.0.1:8001/chat"
    # FILE_I = 'data/PAQ/PAQ.tsv'
    # FILE_O = 'test_1.tsv'

    parser = argparse.ArgumentParser(description="Run PAQ experiment")
    parser.add_argument('--input', '-i', required=True, help='Path to input TSV file')
    parser.add_argument('--output', '-o', required=True, help='Path to output TSV file')
    parser.add_argument('--prompt', '-p', required=True, help='Path to prompt file')
    parser.add_argument('--url', '-u', default='http://127.0.0.1:8001', help='URL for the API endpoint')
    parser.add_argument('--batch', '-b', default=False, help='Batch the requests?')

    args = parser.parse_args()

    if args.batch:
        run_experiment_batch(args.input, args.output, args.prompt, args.url)
    else:
        run_experiment(args.input, args.output, args.prompt, args.url)


# list of what to test

# read PAQ tsv
# iterate over questions
# send request for each question
# save to a new tsv with the original ID



# question = "In which part of England was Dame Judi Dench born?"
# data = {"question": question, "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
#
# response = requests.post(URL, json=data)
# response_jsn = json.loads(response.text)
#
# print("Status Code:", response.status_code)
# print("Response:", response_jsn['answer'])



import json
import requests
import csv
from itertools import islice
import argparse

nq_open_path = 'data/nq_open/NQ-open.dev.jsonl'

def read_in_batches(filename, batch_size):
    with open(filename, 'r') as file:
        while True:
            batch = [json.loads(row)['question'] + '?' for row in islice(file, batch_size)]
            if not batch:
                break
            yield batch

def run_test(result_path, prompt_file_path, URL):
    with open(prompt_file_path, 'r') as fh:
        prompt = fh.read().strip()

    with open(nq_open_path, 'r') as nq_open:
        with open(result_path, 'w') as out_file:
            writer = csv.writer(out_file, delimiter='\t')

            ind = 0
            for row in nq_open:
                question = json.loads(row)['question']
                data = {"question": question,
                        "prompt": prompt,
                        "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
                response = requests.post(f"{URL}/chat", json=data)
                response_jsn = json.loads(response.text)
                system_answer = response_jsn['answer']

                writer.writerow([ind, system_answer])
                ind += 1


def run_test_batch(result_path, prompt_file_path, URL, batch_size):
    with open(prompt_file_path, 'r') as fh:
        prompt = fh.read().strip()

    with open(result_path, 'w') as out_file:
        writer = csv.writer(out_file, delimiter='\t')

        ind = 0
        for questions in read_in_batches(nq_open_path, batch_size):

            data = {"questions": questions,
                    "prompt": prompt,
                    "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}

            response = requests.post(f"{URL}/chat_batch", json=data)
            response_jsn = json.loads(response.text)
            system_answers = response_jsn['answers']
            for system_answer in system_answers:
                writer.writerow([ind, system_answer])
                ind += 1

def run_test_batch_beam_search(result_path, prompt_file_path, URL, batch_size, beam_width):
    with open(prompt_file_path, 'r') as fh:
        prompt = fh.read().strip()

    with open(result_path, 'w') as out_file:
        writer = csv.writer(out_file, delimiter='\t')

        ind = 0
        for questions in read_in_batches(nq_open_path, batch_size):

            data = {"questions": questions,
                    "prompt": prompt,
                    "beam_width": beam_width,
                    "temperature": 0.0
                    }

            response = requests.post(f"{URL}/beam_search_chat_batch", json=data)
            response_jsn = json.loads(response.text)
            system_answers = response_jsn['answers']

            for system_answer in system_answers:
                writer.writerow([ind, system_answer])
                ind += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test NQ Open")
    parser.add_argument('--output', '-o', required=True, help='Path to output TSV file')
    parser.add_argument('--prompt', '-p', required=True, help='Path to prompt file')
    parser.add_argument('--url', '-u', default='http://127.0.0.1:8001', help='URL for the API endpoint')
    parser.add_argument('--batch', '-b', default=1, help='Batch size')
    parser.add_argument('--beam', '-bs', default=1, help='Beam width')

    args = parser.parse_args()
    args.batch = int(args.batch)
    args.beam = int(args.beam)

    assert args.batch > 0
    assert args.beam > 0

    if args.beam == 1:
        if args.batch == 1:
            run_test(args.output, args.prompt, args.url)
        else:
            run_test_batch(args.output, args.prompt, args.url, args.batch)

    else:
        run_test_batch_beam_search(args.output, args.prompt, args.url, args.batch, args.beam)

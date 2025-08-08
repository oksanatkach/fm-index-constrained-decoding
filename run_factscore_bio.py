import csv
import requests
import json
import argparse
from itertools import islice


def read_in_batches(reader, batch_size):
    while True:
        batch = [row[1] for row in islice(reader, batch_size)]
        if not batch:
            break
        yield batch


def main(I_FILE, O_FILE, URL):
    reader = csv.reader(open(I_FILE, 'r'))
    writer = csv.writer(open(O_FILE, 'w'))
    writer.writerow(["index", "response"])

    i = 0
    for _, question in reader:
        data = {"question": question,
                "prompt": "",
                "min_tokens": 100, "max_tokens": 2000,
                "temperature": 0.0, "n": 1, "top_n": 1.0}
        response = requests.post(f"{URL}/chat", json=data)
        response_jsn = json.loads(response.text)
        system_answer = response_jsn['answer']
        system_answer = system_answer.replace('\n', '')
        if '</think>' in system_answer:
            system_answer = system_answer.split('</think>')[-1]
        writer.writerow([i, system_answer])

        i += 1

def main_batch(I_FILE, O_FILE, URL, batch_size):
    reader = csv.reader(open(I_FILE, 'r'))
    batches = read_in_batches(reader, batch_size)

    writer = csv.writer(open(O_FILE, 'w'))
    writer.writerow(["index", "response"])

    i = 0
    for questions in batches:
        data = {"questions": questions,
                "prompt": "",
                "min_tokens": 100, "max_tokens": 2000,
                "temperature": 0.0, "n": 1, "top_n": 1.0}
        response = requests.post(f"{URL}/chat_batch", json=data)
        response_jsn = json.loads(response.text)
        system_answers = response_jsn['answers']
        for system_answer in system_answers:
            system_answer = system_answer.replace('\n', '')
            if '</think>' in system_answer:
                system_answer = system_answer.split('</think>')[-1]
            writer.writerow([i, system_answer])

            i += 1


if __name__ == "__main__":
    # "factscore-bio-data/factscore-bio-input.csv"
    # "factscore-bio-data/output_free.csv"

    parser = argparse.ArgumentParser(description="Run PAQ experiment")
    parser.add_argument('--input', '-i', required=True, help='Path to input TSV file')
    parser.add_argument('--output', '-o', required=True, help='Path to output TSV file')
    parser.add_argument('--url', '-u', default='http://127.0.0.1:8001', help='URL for the API endpoint')
    parser.add_argument('--batch', '-b', default=1, help='Batch size')

    args = parser.parse_args()
    args.batch = int(args.batch)

    assert args.batch > 0
    if args.batch == 1:
        main(args.input, args.output, args.url)
    else:
        main_batch(args.input, args.output, args.url, args.batch)

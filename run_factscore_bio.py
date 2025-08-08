import csv
import requests
import json
import argparse


def main(I_FILE, O_FILE, URL):
    reader = csv.reader(open(I_FILE, 'r'))
    writer = csv.writer(open(O_FILE, 'w'))
    writer.writerow(["index", "response"])

    i = 0
    for _, question in reader:
        writer.writerow([i, question])
        print(question)
        data = {"question": question,
                "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
        response = requests.post(f"{URL}/chat", json=data)
        response_jsn = json.loads(response.text)
        system_answer = response_jsn['answer']
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

    args = parser.parse_args()
    main(args.input, args.output, args.url)

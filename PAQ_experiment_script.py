import requests
import json
import argparse


def run_experiment(FILE_I, FILE_O, URL, paraphrase):
    with open(FILE_I, newline='', encoding='utf-8') as in_file:
        with open(FILE_O, 'w', newline='', encoding='utf-8') as out_file:
            for line in in_file:
                line_id, _, text = tuple(line.strip().split('\t'))
                question, answer = tuple(text.split(' Answer: '))

                if paraphrase:
                    data = {"question": f"Paraphrase this question: {question}", "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
                    response = requests.post("http://127.0.0.1:8001/ask_custom", json=data)
                    response_jsn = json.loads(response.text)
                    question = response_jsn['answer']

                data = {"question": question, "temperature": 0.0, "min_tokens": 10, "n": 1, "top_n": 1.0}
                response = requests.post(URL, json=data)
                response_jsn = json.loads(response.text)
                system_answer = response_jsn['answer']
                out_file.write(line_id + '\t' + system_answer + '\n')


if __name__ == '__main__':
    # URL = "http://127.0.0.1:8001/chat"
    # FILE_I = 'data/PAQ/PAQ.tsv'
    # FILE_O = 'test_1.tsv'

    parser = argparse.ArgumentParser(description="Run PAQ experiment")
    parser.add_argument('--input', '-i', required=True, help='Path to input TSV file')
    parser.add_argument('--output', '-o', required=True, help='Path to output TSV file')
    parser.add_argument('--url', '-u', default='http://127.0.0.1:8001/chat', help='URL for the API endpoint')
    parser.add_argument('--paraphrase', '-p', default=True, help='Paraphrase flag')

    args = parser.parse_args()

    run_experiment(args.input, args.output, args.url, args.paraphrase)


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

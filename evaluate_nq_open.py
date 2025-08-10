import argparse
import pandas as pd

nq_open_path = 'data/nq_open/NQ-open.dev.jsonl'

def process_string(string):
    string = string.strip().lower()
    if '</think>' in string:
        string = string.split('</think>')[-1]
    string = string.replace('\\n', '')
    return string

def process_answers(answers):
    return [process_string(answer) for answer in answers]

def run_evaluation(test_file):
    df = pd.read_json(nq_open_path, lines=True)
    test_df = pd.read_csv(test_file, delimiter='\t', header=None)
    test_df.columns = ['index', 'test_answer']
    df = df.join(test_df['test_answer'])

    df['answer'] = df['answer'].apply(process_answers)
    df['test_answer'] = df['test_answer'].apply(process_string)

    df["match"] = df.apply(
        lambda row: any(row["test_answer"] in s or s in row["test_answer"] for s in row["answer"]),
        axis=1
    )
    return df["match"].mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open QA experiment evaluation")
    parser.add_argument('--experiment', '-e', required=True, help='Path to experiment results TSV file')

    args = parser.parse_args()
    print(run_evaluation(args.experiment))

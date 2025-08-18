import argparse
import re
import pandas as pd
from collections import Counter

nq_open_path = 'data/nq_open/NQ-open.dev.jsonl'


def calculate_f1_score(ground_truth_lst: list, predicted: str) -> float:
    """
    Calculate F1 score between ground truth and predicted answer.
    Treats answers as bags of words (tokens).
    """
    f1 = 0.0
    # Handle NaN or None values
    if not ground_truth_lst or not predicted:
        return f1

    for ground_truth in ground_truth_lst:
        # Tokenize (split on whitespace and punctuation)
        gt_tokens = re.findall(r'\b\w+\b', ground_truth)
        pred_tokens = re.findall(r'\b\w+\b', predicted)

        # If both are empty, return 1.0 (perfect match)
        if not gt_tokens and not pred_tokens:
            return 1.0

        # If one is empty but other isn't, return 0.0
        if not gt_tokens or not pred_tokens:
            return 0.0

        # Calculate precision, recall, and F1
        gt_counter = Counter(gt_tokens)
        pred_counter = Counter(pred_tokens)

        # Find common tokens
        common_tokens = gt_counter & pred_counter
        overlap = sum(common_tokens.values())

        # Calculate precision and recall
        precision = overlap / sum(pred_counter.values()) if sum(pred_counter.values()) > 0 else 0
        recall = overlap / sum(gt_counter.values()) if sum(gt_counter.values()) > 0 else 0

        # Calculate F1 score
        if precision + recall == 0:
            return 0.0

        this_f1 = 2 * (precision * recall) / (precision + recall)
        if this_f1 > f1:
            f1 = this_f1

    return f1

def calculate_exact_match_f1(ground_truth, predicted):
    """
    Calculate F1 score with exact string matching (case-insensitive).
    Returns 1.0 for exact match, 0.0 otherwise.
    """
    if pd.isna(ground_truth) or pd.isna(predicted):
        return 0.0

    gt_normalized = str(ground_truth).lower().strip()
    pred_normalized = str(predicted).lower().strip()

    return 1.0 if gt_normalized == pred_normalized else 0.0

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

    df["soft_match"] = df.apply(
        lambda row: any(row["test_answer"] in s or s in row["test_answer"] for s in row["answer"]),
        axis=1
    )
    df['f1'] = df.apply(
        lambda x: calculate_f1_score(x.answer, x.test_answer),
        axis=1
    )
    print(df['f1'].mean())
    print(df["soft_match"].mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open QA experiment evaluation")
    parser.add_argument('--experiment', '-e', required=True, help='Path to experiment results TSV file')

    args = parser.parse_args()
    print(run_evaluation(args.experiment))

import argparse
import re

def parse_exp_results_line(line):
    line_id, text = tuple(line.strip().split('\t'))
    question, answer = tuple(re.split(r'(?i)answer:', text))
    return line_id, question.strip(), answer.strip()

def parse_line(line):
    line_id, text = tuple(line.strip().split('\t'))
    question, answer = tuple(text.split(' Answer: '))
    return line_id, question, answer

def line_generator(filename):
    with open(filename, "r") as fh:
        for line in fh:
            yield line

def main(test_data_path, experiment_results_path):
    test_set = line_generator(test_data_path)
    exp_results = line_generator(experiment_results_path)
    n_exact_matches = 0
    total_count = 0
    f1_sum = 0
    for exp_line in exp_results:
        try:
            exp_line_id, _, exp_answer = parse_exp_results_line(exp_line)
            testset_line = next(test_set)
            testset_line_id, _, testset_answer = parse_line(testset_line)

            while exp_line_id != testset_line_id:
                testset_line = next(test_set)
                testset_line_id, _, testset_answer = parse_line(testset_line)

            exp_answer = exp_answer.strip().lower().replace('  ', ' ')
            testset_answer = testset_answer.strip().lower().replace('  ', ' ')

            if (exp_answer in testset_answer) or (testset_answer in exp_answer):
                n_exact_matches += 1

            # calc F1
            prediction_tokens = set(exp_answer.split())
            golden_tokens = set(testset_answer.split())
            precision = len(prediction_tokens.intersection(golden_tokens)) / len(prediction_tokens)
            recall = len(prediction_tokens.intersection(golden_tokens)) / len(golden_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_sum += f1
            total_count += 1

        except:
            pass

    print("Experiment:", experiment_results_path)
    print("Exact match accuracy:", n_exact_matches / total_count)
    print("macro F1:", f1_sum / total_count)

if __name__ == '__main__':
    # test_data_path = "/"
    # experiment_results_path = "test_data/results/exp1.tsv"

    parser = argparse.ArgumentParser(description="Run PAQ experiment evaluation")
    parser.add_argument('--testset', '-t', required=True, help='Path to test set TSV file')
    parser.add_argument('--experiment', '-e', required=True, help='Path to experiment results TSV file')

    args = parser.parse_args()
    main(args.testset, args.experiment)

    # main("test_data/PAQ_testset.tsv", "test_data/results/exp1.tsv")

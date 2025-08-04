import argparse

def parse_exp_results_line(line):
    line_id, text = tuple(line.strip().split('\t'))
    question, answer = tuple(text.split('Answer:'))
    return line_id, question.strip(), answer.strip()

def parse_line(line):
    line_id, _, text = tuple(line.strip().split('\t'))
    question, answer = tuple(text.split(' Answer: '))
    return line_id, question, answer

def line_generator(filename, proc_func):
    with open(filename, "r") as fh:
        for line in fh:
            yield proc_func(line)

def main(test_data_path, experiment_results_path):
    test_set = line_generator(test_data_path, parse_line)
    exp_results = line_generator(experiment_results_path, parse_exp_results_line)
    n_exact_matches = 0
    for exp_line_id, _, exp_answer in exp_results:
        testset_line_id, _, testset_answer = next(test_set)
        assert exp_line_id == testset_line_id
        if exp_answer == testset_answer:
            n_exact_matches += 1

    print("Experiment:", experiment_results_path)
    print("Exact match accuracy:", n_exact_matches)

if __name__ == '__main__':
    # test_data_path = "test_data/PAQ_testset.tsv"
    # experiment_results_path = "test_data/results/exp1.tsv"

    parser = argparse.ArgumentParser(description="Run PAQ experiment evaluation")
    parser.add_argument('--testset', '-t', required=True, help='Path to test set TSV file')
    parser.add_argument('--experiment', '-e', required=True, help='Path to experiment results TSV file')

    args = parser.parse_args()
    main(args.testset, args.experiment)

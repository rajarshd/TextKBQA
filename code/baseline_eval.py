import json

def augment_with_baseline_answers(baseline_answer_file, input_file, output_file):
    out = open(output_file, 'w')
    with open(baseline_answer_file) as input, open(input_file) as data_file:
        for baseline_answer_line, line in zip(input, data_file):
            baseline_answer_line = baseline_answer_line.strip()
            sentence, correct_answer, predicted_answer = baseline_answer_line.split('\t')
            correct = 1 if correct_answer == predicted_answer else 0
            data = json.loads(line)
            data['baseline_answer'] = predicted_answer
            data['is_correct'] =correct
            out.write(json.dumps(data)+'\n')


def get_baseline_accuracy(input_file, min_num_mem, max_num_mem):
    num_correct = 0
    num_data = 0
    with open(input_file) as input:
        for line in input:
            line = line.strip()
            data = json.loads(line)
            num_facts = data['num_facts']
            if num_facts < min_num_mem or num_facts > max_num_mem:
                continue
            num_data += 1
            num_correct += data['is_correct']

    print('Num data {0:10d}, Num correct {1:10d}, %correct {2:10.4f}'.format(num_data, num_correct, 1.0*num_correct/num_data))


if __name__ == '__main__':

    baseline_answer_file = "/home/rajarshi/canvas/data/TextKBQA/dev_answers.txt"
    input_file = "/home/rajarshi/canvas/data/TextKBQA/dev_with_facts.json"
    output_file = "/home/rajarshi/canvas/data/TextKBQA/dev_with_baseline_answers.json"

    # augment_with_baseline_answers(baseline_answer_file, input_file, output_file)
    get_baseline_accuracy(output_file, 0, 25000)
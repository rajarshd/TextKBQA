import numpy as np
import argparse
from tqdm import tqdm
import json


class QualAnalysis(object):
    def __init__(self):

        self.kb_facts = self.read_kb_facts(kb_file) if use_kb else None
        self.text_kb_facts = self.read_text_kb_facts(text_kb_file) if use_text else None
        self.questions = self.read_questions(input_test_file)
        # print('Reading mid to word map')
        # self.mid_to_word_map = self.mid_to_word()

    def read_kb_facts(self, input_file):
        facts = []
        print('Reading kb file at {}'.format(input_file))
        with open(input_file) as fb:
            for line in tqdm(fb):
                line = line.strip()
                line = line[1:-1]
                e1, r1, r2, e2 = [a.strip('"') for a in [x.strip() for x in line.split(',')]]
                r = r1 + '_' + r2
                facts.append({'e1': e1, 'r': r, 'e2': e2})
        return facts

    def read_text_kb_facts(self, input_file):
        facts = []
        print('Reading text kb file at {}'.format(input_file))
        with open(input_file) as fin:
            for counter, line in tqdm(enumerate(fin)):
                kb_instance = json.loads(line)
                facts.append(kb_instance)
        return facts

    def read_questions(self, input_file):
        questions = []
        print('Reading file at {}'.format(input_file))
        with open(input_file) as f_in:
            for counter, line in tqdm(enumerate(f_in)):
                question = json.loads(line)
                questions.append(question)
        return questions

    def get_relevant_memory(self, question_index, mem_index, use_kb=True):
        """
        get the relevant memory either kb or text. Note one of use_kb and use_text can be true
        at a given time, if both are true this needs to be called twice with each value of use_kb (true, false)
        and the returned value needs to be handled appropriately.
        :param question_index:
        :param mem_index:
        :param use_kb:
        :return:
        """
        question = self.questions[question_index]
        mem_index += 1  # convert from 0 index
        start_index_key = 'start_indices' if use_kb else 'text_kb_start_indices'
        length_key = 'lengths' if use_kb else 'text_kb_lengths'
        memory = self.kb_facts if use_kb else self.text_kb_facts
        start_indices = question[start_index_key]
        lengths = question[length_key]
        q_start_indices = np.asarray(start_indices)
        q_fact_lengths = np.asarray(lengths)
        sorted_index = np.argsort(q_fact_lengths)
        q_fact_lengths = q_fact_lengths[sorted_index]
        q_start_indices = q_start_indices[sorted_index]
        cum_num_mem_slots = 0
        counter = 0
        for fact_len in q_fact_lengths:
            if cum_num_mem_slots + fact_len >= mem_index:  # the mem lies in the next partition
                # calculate the off set
                offset = mem_index - cum_num_mem_slots - 1  # -1 because converting to zero index
                return memory[q_start_indices[counter] + offset]
            else:
                cum_num_mem_slots += fact_len
                counter += 1

    def read_attn_wts_file(self, input_file, input_predicted_answer_file):

        f_out = open(output_dir+'/attn_memory.txt','a')
        f_out_correct = open(output_dir + '/attn_memory.txt.correct', 'a')
        print('Loading the attn wights...')
        attn_wts = np.load(input_file)
        print('Loading predicted answer file')
        num_questions = len(self.questions)
        answers = np.fromfile(input_predicted_answer_file)
        answers = answers.reshape(num_questions, -1)
        assert attn_wts.ndim == 2
        num_data, max_mem_slots = attn_wts.shape
        # get the index
        print('Sorting....')
        sorted_index = np.argsort(attn_wts, axis=1)
        sorted_wts = np.sort(attn_wts, axis=1)
        print('done...')
        # get the slice we are interested in
        start_index = max_mem_slots - topk
        sorted_index = sorted_index[:, start_index:]
        sorted_wts = sorted_wts[:, start_index:]
        for data_index in range(num_data): # refactor the double loop
            sentence = self.questions[data_index]['sentence']
            split_sentence = sentence.split(' ')
            entities = self.questions[data_index]['entities']
            for entity in entities:
                split_sentence[entity['index']] = entity['entity']
            sentence_with_entities = ' '.join(split_sentence)
            is_correct = (answers[data_index][1] == answers[data_index][0])
            f_out.write('Sentence: {}\n'.format(sentence))
            f_out.write('Sentence with entities: {}\n'.format(sentence_with_entities))
            f_out.write('Correct Answer: {}\n'.format(rev_entity_vocab[int(answers[data_index][1])]))
            f_out.write('Predicted Answer: {}\n'.format(rev_entity_vocab[int(answers[data_index][0])]))
            f_out.write('Memories\n')
            if is_correct:
                f_out_correct.write('Sentence: {}\n'.format(sentence))
                f_out_correct.write('Sentence with entities: {}\n'.format(sentence_with_entities))
                f_out_correct.write('Correct Answer: {}\n'.format(rev_entity_vocab[int(answers[data_index][1])]))
                f_out_correct.write('Predicted Answer: {}\n'.format(rev_entity_vocab[int(answers[data_index][0])]))
                f_out_correct.write('Memories\n')
            for index in reversed(range(topk)):
                mem_index = self.get_relevant_memory(data_index, sorted_index[data_index][index], use_kb=use_kb)
                f_out.write('Attn wt: {0:10.4f}\n'.format(sorted_wts[data_index][index]))
                f_out.write('Memory: {}\n'.format(mem_index))
                if is_correct:
                    f_out_correct.write('Attn wt: {0:10.4f}\n'.format(sorted_wts[data_index][index]))
                    f_out_correct.write('Memory: {}\n'.format(mem_index))
                    if mem_index is not None:
                        mem_index['value'] = self.mid_to_word_map[mem_index['value']] if mem_index['value'] in self.mid_to_word_map else mem_index['value']
                        key = [self.mid_to_word_map[word] if word in self.mid_to_word_map else word for word in mem_index['key']]
                        mem_index['key'] = key
                        mem_index['entity'] = self.mid_to_word_map[mem_index['entity']] if mem_index['entity'] in self.mid_to_word_map else mem_index['entity']
                        f_out_correct.write('Memory in words: {}\n'.format(mem_index))
                        f_out.write('Memory in words: {}\n'.format(mem_index))

            f_out.write("=============\n")
            if is_correct:
                f_out_correct.write("=============\n")

    def get_siva_output(self, input_predicted_answer_file):

        num_questions = len(self.questions)
        outputs = np.fromfile(input_predicted_answer_file) # manzil had changed the output structure storing (sentence, prediction). Also he changed np.save to np.tofile so reading would change
        # outputs = np.load(input_predicted_answer_file)
        outputs = outputs.reshape(num_questions, -1)
        num_questions, sequence_length = outputs.shape
        predicted_answers = outputs[:,sequence_length-1] #last column
        # predicted_answers = outputs[:, 0]  # last column
        correct_counter = 0
        for counter, question in enumerate(self.questions):
            print(question['sentence']+'\t'+ '[\"'+question['answerSubset'][0]+'\"]'+'\t'+'[\"'+rev_entity_vocab[predicted_answers[counter]]+'\"]')
            if question['answerSubset'][0] == rev_entity_vocab[predicted_answers[counter]]:
                correct_counter += 1
        print('Accuracy: {}'.format(correct_counter*1.0/num_questions))
            # print(question['answerSubset'])
            # print(rev_entity_vocab[predicted_answers[counter]])



    def __call__(self, *args, **kwargs):

        self.read_attn_wts_file(input_attn_file, input_predicted_answer_file)
        # self.get_siva_output(input_predicted_answer_file)
        print('Done')

    def mid_to_word(self):
        word_to_mid = {}
        mid_to_word = {}
        with open('/iesl/canvas/pat/data/freebase/freebase_names', 'r') as f:
            for line in tqdm(f):
                mid, word, _ = line.split('\t')
                word_to_mid[word] = 'm.' + mid[2:]
                mid_to_word['m.' + mid[2:]] = word
        return mid_to_word



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_kb", default=1, type=int)
    parser.add_argument("--use_text", default=0, type=int)
    parser.add_argument("--kb_file", required=True)
    parser.add_argument("--text_kb_file", required=True)
    parser.add_argument("--attn_file", required=True)
    parser.add_argument("--answer_file", required=True)
    parser.add_argument("--input_test_file", required=True)
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    kb_file = args.kb_file
    text_kb_file = args.text_kb_file
    use_kb = (args.use_kb == 1)
    use_text = (args.use_text == 1)
    input_attn_file = args.attn_file
    input_predicted_answer_file = args.answer_file
    input_test_file = args.input_test_file
    topk = args.k
    output_dir = args.output_dir
    vocab_dir = "/home/rajarshi/research/joint-text-and-kb-inference-semantic-parsing/vocab"
    print('Reading entity vocab')
    entity_vocab = json.load(open(vocab_dir + '/entity_vocab.json'))
    rev_entity_vocab = {}
    for k,v in entity_vocab.iteritems():
        rev_entity_vocab[v] = k
    qual_analysis = QualAnalysis()
    qual_analysis()
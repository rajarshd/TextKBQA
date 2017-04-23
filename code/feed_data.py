from data_utils import KB, Text, TextKb
import numpy as np
from tqdm import tqdm


class Batcher(object):
    def __init__(self, input_file, kb_file, text_kb_file, batch_size, vocab_dir, return_one_epoch=False, shuffle=True,
                 min_num_mem_slots=100,
                 max_num_mem_slots=500,
                 min_num_text_mem_slots=0,
                 max_num_text_mem_slots=1000,
                 use_kb_mem=True,
                 use_text_mem=False):
        self.batch_size = batch_size
        self.input_file = input_file
        self.kb_file = kb_file
        self.text_kb_file = text_kb_file
        self.shuffle = shuffle
        self.max_num_mem_slots = max_num_mem_slots
        self.min_num_mem_slots = min_num_mem_slots
        self.max_num_text_mem_slots = max_num_text_mem_slots
        self.min_num_text_mem_slots = min_num_text_mem_slots
        self.vocab_dir = vocab_dir
        self.return_one_epoch = return_one_epoch
        self.use_kb_mem = use_kb_mem
        self.use_text_mem = use_text_mem
        self.questions, self.q_lengths, self.answers, \
        self.kb_memory_slots, self.kb_num_memories, \
        self.text_key_mem, self.text_key_len, \
        self.text_val_mem, self.num_text_mems = self.read_files()
        self.max_key_len = None

        if self.use_text_mem and self.use_kb_mem:
            assert self.text_key_mem is not None and self.kb_memory_slots is not None
        elif self.use_kb_mem:
            assert self.text_key_mem is None and self.kb_memory_slots is not None
        else:
            assert self.text_key_mem is not None and self.kb_memory_slots is  None

        self.num_questions = len(self.questions)
        print('Num questions {}'.format(self.num_questions))
        self.start_index = 0
        if self.shuffle:
            self.shuffle_data()

    def get_next_batch(self):
        """
        returns the next batch
        TODO(rajarshd): move the if-check outside the loop, so that conditioned is not checked every damn time. the conditions are suppose to be immutable.
        """
        while True:
            if self.start_index >= self.num_questions:
                if self.return_one_epoch:
                    return  # stop after returning one epoch
                self.start_index = 0
                if self.shuffle:
                    self.shuffle_data()
            else:
                num_data_returned = min(self.batch_size, self.num_questions - self.start_index)
                assert num_data_returned > 0
                end_index = self.start_index + num_data_returned
                if self.use_kb_mem and self.use_text_mem:
                    yield self.questions[self.start_index:end_index], self.q_lengths[self.start_index:end_index], \
                      self.answers[self.start_index:end_index], self.kb_memory_slots[self.start_index:end_index], \
                      self.kb_num_memories[self.start_index:end_index], self.text_key_mem[self.start_index:end_index], \
                          self.text_key_len[self.start_index:end_index], self.text_val_mem[self.start_index:end_index], \
                          self.num_text_mems[self.start_index:end_index]
                elif self.use_kb_mem:
                    yield self.questions[self.start_index:end_index], self.q_lengths[self.start_index:end_index], \
                          self.answers[self.start_index:end_index], self.kb_memory_slots[self.start_index:end_index], \
                          self.kb_num_memories[self.start_index:end_index]
                else:
                    yield self.questions[self.start_index:end_index], self.q_lengths[self.start_index:end_index], \
                          self.answers[self.start_index:end_index], self.text_key_mem[self.start_index:end_index], \
                          self.text_key_len[self.start_index:end_index], self.text_val_mem[self.start_index:end_index], \
                          self.num_text_mems[self.start_index:end_index]
                self.start_index = end_index

    def shuffle_data(self):
        """
        Shuffles maintaining the same order.
        """
        perm = np.random.permutation(self.num_questions)  # perm of index in range(0, num_questions)
        assert len(perm) == self.num_questions
        if self.use_kb_mem and self.use_text_mem:
            self.questions, self.q_lengths, self.answers, self.kb_memory_slots, self.kb_num_memories, self.text_key_mem,\
            self.text_key_len, self.text_val_mem, self.num_text_mems = \
                self.questions[perm], self.q_lengths[perm], self.answers[perm], self.kb_memory_slots[perm], \
                self.kb_num_memories[perm], self.text_key_mem[perm], self.text_key_len[perm], self.text_val_mem[perm], self.num_text_mems[perm]
        elif self.use_kb_mem:
            self.questions, self.q_lengths, self.answers, self.kb_memory_slots, self.kb_num_memories = \
                self.questions[perm], self.q_lengths[perm], self.answers[perm], self.kb_memory_slots[perm], \
                self.kb_num_memories[perm]
        else:
            self.questions, self.q_lengths, self.answers, self.text_key_mem, self.text_key_len, self.text_val_mem,\
            self.num_text_mems = self.questions[perm], self.q_lengths[perm], self.answers[perm], self.text_key_mem[perm],\
                                 self.text_key_len[perm], self.text_val_mem[perm], self.num_text_mems[perm]
    def reset(self):
        self.start_index = 0

    def read_files(self):
        """reads the kb and text files and creates the numpy arrays after padding"""
        # read the KB file
        kb = KB(self.kb_file, vocab_dir=self.vocab_dir) if self.use_kb_mem else None
        # read text kb file
        text_kb = TextKb(self.text_kb_file, vocab_dir=self.vocab_dir) if self.use_text_mem else None
        self.max_key_len = text_kb.max_key_length if self.use_text_mem else None
        # Question file
        questions = Text(self.input_file,
                         max_num_facts=self.max_num_mem_slots,
                         min_num_facts=self.min_num_mem_slots,
                         min_num_text_facts=self.min_num_text_mem_slots,
                         max_num_text_facts=self.max_num_text_mem_slots)
        max_q_length, max_num_kb_facts, max_num_text_kb_facts, question_list = questions.max_q_length, \
                                                                               questions.max_num_kb_facts, \
                                                                               questions.max_num_text_kb_facts, \
                                                                               questions.question_list
        entity_vocab = kb.entity_vocab if self.use_kb_mem else text_kb.entity_vocab
        relation_vocab = kb.relation_vocab if self.use_kb_mem else text_kb.relation_vocab
        num_questions = len(question_list)
        question_lengths = np.ones([num_questions]) * -1
        questions = np.ones([num_questions, max_q_length]) * entity_vocab['PAD']
        answers = np.ones_like(question_lengths) * entity_vocab['UNK']
        all_kb_memories = None
        num_kb_memories = None
        text_key_memories = None
        text_key_lengths = None
        text_val_memories = None
        num_text_memories = None

        if self.use_kb_mem:
            print('Make data tensors for kb')
            all_kb_memories = np.ones([num_questions, max_num_kb_facts, 3])
            all_kb_memories[:, :, 0].fill(entity_vocab['DUMMY_MEM'])
            all_kb_memories[:, :, 2].fill(entity_vocab['DUMMY_MEM'])
            all_kb_memories[:, :, 1].fill(relation_vocab['DUMMY_MEM'])
            num_kb_memories = np.ones_like(question_lengths) * -1
            for q_counter, q in enumerate(tqdm(question_list)):
                question_str = q.parsed_question['question']
                question_entities = q.parsed_question['entities']
                question_indices = q.parsed_question['indices']
                q_answers = q.parsed_question['answers']
                # num_kb_memories.append(q.parsed_question['num_facts'])
                num_kb_memories[q_counter] = q.parsed_question['num_facts']
                q_start_indices = np.asarray(q.parsed_question['start_indices'])
                q_fact_lengths = np.asarray(
                    q.parsed_question['fact_lengths'])  # for each entity in question retrieve the fact
                sorted_index = np.argsort(q_fact_lengths)
                q_fact_lengths = q_fact_lengths[sorted_index]
                q_start_indices = q_start_indices[sorted_index]
                question_words_list = question_str.split(' ')
                for counter, index in enumerate(question_indices):  # replace the entities with their ids
                    question_words_list[index] = question_entities[counter]
                question_int = [entity_vocab[w_q] if w_q.strip() in entity_vocab else entity_vocab['UNK'] for w_q in
                                question_words_list]
                question_len = len(question_int)
                questions[q_counter, 0:question_len] = question_int
                question_lengths[q_counter] = question_len
                answer_int = [entity_vocab[a] if a in entity_vocab else entity_vocab['UNK'] for a in q_answers]
                answers[q_counter] = answer_int[0]

                # memories
                kb_facts = kb.facts
                mem_counter = 0
                for counter, start_index in enumerate(q_start_indices):
                    num_facts = q_fact_lengths[counter]
                    if mem_counter < self.max_num_mem_slots:
                        for mem_index in xrange(start_index, start_index + num_facts):
                            mem = kb_facts[mem_index]
                            e1_int = entity_vocab[mem['e1']] if mem['e1'] in entity_vocab else entity_vocab['UNK']
                            e2_int = entity_vocab[mem['e2']] if mem['e2'] in entity_vocab else entity_vocab['UNK']
                            r_int = relation_vocab[mem['r']] if mem['r'] in relation_vocab else relation_vocab['UNK']
                            all_kb_memories[q_counter][mem_counter][0] = e1_int
                            all_kb_memories[q_counter][mem_counter][1] = r_int
                            all_kb_memories[q_counter][mem_counter][2] = e2_int
                            mem_counter += 1
                            if mem_counter == self.max_num_mem_slots:  # will use the first max_num_mem_slots slots
                                break
        if self.use_text_mem:

            print('Make data tensors for text kb')
            max_key_len = text_kb.max_key_length
            text_key_memories = np.ones([num_questions, max_num_text_kb_facts, max_key_len]) * entity_vocab['DUMMY_MEM']
            text_key_lengths = np.zeros([num_questions, max_num_text_kb_facts])
            text_val_memories = np.ones([num_questions, max_num_text_kb_facts]) * entity_vocab['DUMMY_MEM']
            num_text_memories = np.ones_like(question_lengths) * -1
            for q_counter, q in enumerate(tqdm(question_list)):
                # TODO (rajarshd): Move the repeated piece of code in a method.
                question_str = q.parsed_question['question']
                question_entities = q.parsed_question['entities']
                question_indices = q.parsed_question['indices']
                q_answers = q.parsed_question['answers']
                question_words_list = question_str.split(' ')
                for counter, index in enumerate(question_indices):  # replace the entities with their ids
                    question_words_list[index] = question_entities[counter]
                question_int = [entity_vocab[w_q] if w_q.strip() in entity_vocab else entity_vocab['UNK'] for w_q in
                                question_words_list]
                question_len = len(question_int)
                questions[q_counter, 0:question_len] = question_int
                question_lengths[q_counter] = question_len
                answer_int = [entity_vocab[a] if a in entity_vocab else entity_vocab['UNK'] for a in q_answers]
                answers[q_counter] = answer_int[0]

                # memories
                num_q_text_memories = q.parsed_question['text_kb_num_facts']
                # in the training set, account for the discarded memories
                if 'black_lists' in q.parsed_question:
                    num_discarded = 0
                    for black_list in q.parsed_question['black_lists']:
                        num_discarded += len(black_list)
                    num_q_text_memories -= num_discarded
                num_text_memories[q_counter] = num_q_text_memories
                q_start_indices = np.asarray(q.parsed_question['text_kb_start_indices'])
                q_fact_lengths = np.asarray(
                    q.parsed_question['text_kb_lengths'])  # for each entity in question retrieve the fact
                q_black_lists = np.asarray(
                    q.parsed_question['black_lists']) if 'black_lists' in q.parsed_question else None
                sorted_index = np.argsort(q_fact_lengths)
                q_fact_lengths = q_fact_lengths[sorted_index]
                q_start_indices = q_start_indices[sorted_index]
                q_black_lists = q_black_lists[sorted_index] if q_black_lists is not None else None
                text_kb_facts = text_kb.facts_list
                mem_counter = 0
                for counter, start_index in enumerate(q_start_indices):
                    num_facts = q_fact_lengths[counter]
                    black_list_entity = set(q_black_lists[counter]) if q_black_lists is not None else None
                    if mem_counter < self.max_num_text_mem_slots:
                        for mem_entity_counter, mem_index in enumerate(xrange(start_index, start_index + num_facts)):
                            if black_list_entity is not None and mem_entity_counter in black_list_entity:
                                continue
                            mem = text_kb_facts[mem_index]
                            key = mem['key']
                            key_int = [entity_vocab[k] if k in entity_vocab else entity_vocab['UNK'] for k in key]
                            val = mem['value']
                            val_int = entity_vocab[val] if val in entity_vocab else entity_vocab['UNK']
                            key_len = int(mem['key_length'])
                            text_key_memories[q_counter][mem_counter][0:key_len] = key_int
                            text_val_memories[q_counter][mem_counter] = val_int
                            text_key_lengths[q_counter][mem_counter] = key_len
                            mem_counter += 1
                            if mem_counter == self.max_num_text_mem_slots:  # will use the first max_num_mem_slots slots
                                break

        return questions, question_lengths, answers, all_kb_memories, num_kb_memories, \
               text_key_memories, text_key_lengths, text_val_memories, num_text_memories

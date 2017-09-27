from __future__ import print_function
from feed_data import Batcher
from KBQA import KBQA, TextQA, TextKBQA
import tensorflow as tf
import argparse
import time
import numpy as np
import cPickle as pickle
from tqdm import tqdm
import pdb


class Trainer(object):
    def __init__(self):
        with tf.Session() as sess:
            print('Blake hack for acquiring gpu')

        # pretraining
        entity_lookup_table = None
        if load_pretrained_vectors:
            print('Loading pretrained word embeddings...')
            with open(pretrained_vector_path, 'rb') as f:
                entity_lookup_table = pickle.load(f)
            if verbose:
                print("Loaded pretrained vectors of size: ", entity_lookup_table.shape)
                print("Entity vocab size: ", entity_vocab_size)

        # data
        self.batcher = Batcher(train_file, kb_file, text_kb_file, batch_size, vocab_dir,
                               min_num_mem_slots=min_facts, max_num_mem_slots=max_facts, use_kb_mem=use_kb,
                               use_text_mem=use_text, max_num_text_mem_slots=max_text_facts,
                               min_num_text_mem_slots=min_facts)

        self.dev_batcher = Batcher(dev_file, kb_file, text_kb_file, dev_batch_size, vocab_dir,
                                   min_num_mem_slots=min_facts, max_num_mem_slots=dev_max_facts,
                                   return_one_epoch=True, shuffle=False, use_kb_mem=use_kb, use_text_mem=use_text,
                                   max_num_text_mem_slots=dev_max_text_facts, min_num_text_mem_slots=min_facts)

        # define network
        if use_kb and use_text:
            self.model = TextKBQA(entity_vocab_size=entity_vocab_size, relation_vocab_size=relation_vocab_size,
                                  embedding_size=embedding_size, hops=hops, load_pretrained_model=load_model,
                                  load_pretrained_vectors=load_pretrained_vectors, join=combine_text_kb_answer,
                                  pretrained_entity_vectors=entity_lookup_table, verbose=verbose,
                                  separate_key_lstm=separate_key_lstm)
        elif use_kb:
            self.model = KBQA(entity_vocab_size=entity_vocab_size, relation_vocab_size=relation_vocab_size,
                              embedding_size=embedding_size, hops=hops, load_pretrained_model=load_model,
                              load_pretrained_vectors=load_pretrained_vectors,
                              pretrained_entity_vectors=entity_lookup_table, verbose=verbose)
        elif use_text:
            self.model = TextQA(entity_vocab_size=entity_vocab_size, embedding_size=embedding_size, hops=hops, load_pretrained_model=load_model,
                                load_pretrained_vectors=load_pretrained_vectors,
                                pretrained_entity_vectors=entity_lookup_table, verbose=verbose,
                                separate_key_lstm=separate_key_lstm)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(lr)

        self.max_dev_acc = -1.0

    def bp(self, cost):
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        return train_op

    def initialize(self):
        #### inputs ####
        self.question = tf.placeholder(tf.int32, [None, None], name="question")
        self.question_lengths = tf.placeholder(tf.int32, [None], name="question_lengths")
        self.answer = tf.placeholder(tf.int32, [None], name="answer")
        if use_kb and use_text:
            self.memory = tf.placeholder(tf.int32, [None, None, 3], name="memory")
            self.text_key_mem = tf.placeholder(tf.int32, [None, None, None], name="key_mem")
            self.text_key_len = tf.placeholder(tf.int32, [None, None], name="key_len")
            self.text_val_mem = tf.placeholder(tf.int32, [None, None], name="val_mem")
            # network output
            self.output = self.model(self.memory, self.text_key_mem, self.text_key_len, self.text_val_mem,
                                self.question, self.question_lengths)
        elif use_kb:
            self.memory = tf.placeholder(tf.int32, [None, None, 3], name="memory")
            # network output
            self.output = self.model(self.memory, self.question, self.question_lengths)
        elif use_text:
            self.text_key_mem = tf.placeholder(tf.int32, [None, None, None], name="key_mem")
            self.text_key_len = tf.placeholder(tf.int32, [None, None], name="key_len")
            self.text_val_mem = tf.placeholder(tf.int32, [None, None], name="val_mem")
            # network output
            self.output = self.model(self.text_key_mem, self.text_key_len, self.text_val_mem, self.question,
                                self.question_lengths)

        # predict
        self.probs = tf.nn.softmax(self.output)
        self.predict_op = tf.argmax(self.output, 1, name="predict_op")
        self.rank_op = tf.nn.top_k(self.output, 50)

        # loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.output, self.answer)
        self.loss = tf.reduce_mean(cross_entropy, name="loss_op") 

        if use_kb and use_text:
            # Graph created now load/save op for it
            # load the parameters for the kb only model
            #var_list = [v for v in tf.trainable_variables() if v.name.startswith('BiRNN/')]
            #var_list += [self.model.entity_lookup_table, self.model.relation_lookup_table, self.model.W, self.model.b,
            #             self.model.W1, self.model.b1, self.model.R[0]]
            #self.saver = tf.train.Saver(var_list=var_list)
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.Saver()

        # Add to the Graph the Ops that calculate and apply gradients.
        self.train_op = self.bp(self.loss)

        # return the variable initializer Op.
        init_op = tf.initialize_all_variables()

        return init_op

    def dev_eval(self, sess):
        print('Evaluating on dev set...')
        dev_start_time = time.time()
        num_dev_data = 0
        dev_loss = 0.0
        dev_acc = 0.0

        attn_weight = None
        preds = []
        SRR = 0.0
        for data in tqdm(self.dev_batcher.get_next_batch()):

            if use_kb and use_text:
                dev_batch_question, dev_batch_q_lengths, dev_batch_answer, dev_batch_memory, dev_batch_num_memories, \
                dev_batch_text_key_mem, dev_batch_text_key_len, dev_batch_text_val_mem, dev_batch_num_text_mems = data
                feed_dict_dev = {self.question: dev_batch_question,
                                 self.question_lengths: dev_batch_q_lengths,
                                 self.answer: dev_batch_answer,
                                 self.memory: dev_batch_memory,
                                 self.text_key_mem: dev_batch_text_key_mem,
                                 self.text_key_len: dev_batch_text_key_len,
                                 self.text_val_mem: dev_batch_text_val_mem}
            elif use_kb:
                dev_batch_question, dev_batch_q_lengths, dev_batch_answer, dev_batch_memory, dev_batch_num_memories = data
                feed_dict_dev = {self.question: dev_batch_question,
                                 self.question_lengths: dev_batch_q_lengths,
                                 self.answer: dev_batch_answer,
                                 self.memory: dev_batch_memory}
            elif use_text:
                dev_batch_question, dev_batch_q_lengths, dev_batch_answer, dev_batch_text_key_mem, dev_batch_text_key_len, \
                dev_batch_text_val_mem, dev_batch_num_text_mems = data
                feed_dict_dev = {self.question: dev_batch_question,
                                 self.question_lengths: dev_batch_q_lengths,
                                 self.answer: dev_batch_answer,
                                 self.text_key_mem: dev_batch_text_key_mem,
                                 self.text_key_len: dev_batch_text_key_len,
                                 self.text_val_mem: dev_batch_text_val_mem}

            dev_batch_loss_value, dev_prediction, batch_attn_weight, topk = sess.run(
                [self.loss, self.predict_op, self.model.attn_weights_all_hops, self.rank_op],
                feed_dict=feed_dict_dev)

            for j,v in enumerate(topk.indices):
                for i,w in enumerate(v):
                    if w == dev_batch_answer[j]:
                       SRR += 1.0/(i+1)

            dev_loss += dev_batch_loss_value
            num_dev_data += dev_batch_question.shape[0]
            dev_acc += np.sum(dev_prediction == dev_batch_answer)
            attn_weight = batch_attn_weight[0] if attn_weight is None \
                else np.vstack((attn_weight, batch_attn_weight[0]))
            # store predictions
            dev_prediction = np.expand_dims(dev_prediction, axis=1)
            dev_batch_answer = np.expand_dims(dev_batch_answer, axis=1)
            if dev_prediction is not None:
                concat = np.concatenate((dev_prediction, dev_batch_answer), axis=1)
                preds.append(concat)

        print('MRR: ', SRR/num_dev_data)
        dev_acc = (1.0 * dev_acc / num_dev_data)
        dev_loss = (1.0 * dev_loss / num_dev_data)
        if print_attention_weights:
            f_out = open(output_dir + "/attn_wts.npy", 'w')
            np.save(f_out, attn_weight)
            print('Wrote attention weights...')

        self.dev_batcher.reset()
        if dev_acc >= 0.3 or mode == 'test':
            f_out = open(output_dir + "/out_txt." + str(dev_acc), 'w')
            print('Writing to {}'.format("out_txt." + str(dev_acc)))

            preds = np.vstack(preds)
            preds.tofile(f_out)
            if mode == 'test':
                f_out1 = open(output_dir + "/out.txt", 'w')
                preds.tofile(f_out1)
                f_out1.close()

            f_out.close()
        print(
            'It took {0:10.4f}s to evaluate on dev set of size: {3:10d} with dev loss: {1:10.4f} and dev acc: {2:10.4f}'.format(
                time.time() - dev_start_time, dev_loss, dev_acc, num_dev_data))

        return dev_acc, dev_loss

    def fit(self):

        train_loss = 0.0
        batch_counter = 0
        train_acc = 0.0
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(self.initialize())

            if load_model:
                print('Loading retrained model from {}'.format(model_path))
                self.saver.restore(sess, model_path)

                if mode == 'test':
                    self.dev_eval(sess)
                    # print(sess.run(self.model.b))
            # self.dev_eval(sess)
            if mode == 'train':
                self.start_time = time.time()
                print('Starting to train')
                for data in self.batcher.get_next_batch():
                    batch_counter += 1
                    if use_kb and use_text:
                        batch_question, batch_q_lengths, batch_answer, batch_memory, batch_num_memories, \
                        batch_text_key_mem, batch_text_key_len, batch_text_val_mem, batch_num_text_mems = data
                        feed_dict = {self.question: batch_question,
                                     self.question_lengths: batch_q_lengths,
                                     self.answer: batch_answer,
                                     self.memory: batch_memory,
                                     self.text_key_mem: batch_text_key_mem,
                                     self.text_key_len: batch_text_key_len,
                                     self.text_val_mem: batch_text_val_mem}
                    elif use_kb:
                        batch_question, batch_q_lengths, batch_answer, batch_memory, batch_num_memories = data
                        feed_dict = {self.question: batch_question,
                                     self.question_lengths: batch_q_lengths,
                                     self.answer: batch_answer,
                                     self.memory: batch_memory}
                    elif use_text:
                        batch_question, batch_q_lengths, batch_answer, batch_text_key_mem, batch_text_key_len, \
                        batch_text_val_mem, batch_num_text_mems = data
                        feed_dict = {self.question: batch_question,
                                     self.question_lengths: batch_q_lengths,
                                     self.answer: batch_answer,
                                     self.text_key_mem: batch_text_key_mem,
                                     self.text_key_len: batch_text_key_len,
                                     self.text_val_mem: batch_text_val_mem}

                    # train
                    batch_loss_value, _, prediction = sess.run([self.loss, self.train_op, self.predict_op],
                                                               feed_dict=feed_dict)
                    batch_train_acc = (1.0 * np.sum(prediction == batch_answer) / (batch_question.shape[0]))

                    train_loss = 0.98 * train_loss + 0.02 * batch_loss_value
                    train_acc = 0.98 * train_acc + 0.02 * batch_train_acc
                    print('\t at iter {0:10d} at time {1:10.4f}s train loss: {2:10.4f}, train_acc: {3:10.4f} '.format(
                        batch_counter,
                        time.time() - self.start_time,
                        train_loss, train_acc))
                    if batch_counter != 0 and batch_counter % dev_eval_counter == 0:  # predict on dev
                        dev_acc, dev_loss = self.dev_eval(sess)
                        print('\t at iter {0:10d} at time {1:10.4f}s dev loss: {2:10.4f} dev_acc: {3:10.4f} '.format(
                            batch_counter, time.time() - self.start_time, dev_loss, dev_acc))
                        if dev_acc > self.max_dev_acc:
                            self.max_dev_acc = dev_acc
                            # save this model
                            save_path = self.saver.save(sess, output_dir + "/max_dev_out.ckpt")
                            if use_kb and use_text:
                                save_path = self.saver.save(sess, output_dir + "/full_max_dev_out.ckpt")
                            with open(output_dir + "/dev_accuracies.txt", mode='a') as out:
                                out.write(
                                    'Dev accuracy while writing max_dev_out.ckpt {0:10.4f}\n'.format(self.max_dev_acc))
                            print("Saved model")
                        if batch_counter % save_counter == 0:
                            save_path = self.saver.save(sess, output_dir + "/out.ckpt")
                            print("Saved model")
                            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", required=True)
    parser.add_argument("--dev_file", required=True)
    parser.add_argument("-k", "--kb_file", required=True)
    parser.add_argument("--text_kb_file", required=True)
    parser.add_argument("-v", "--vocab_dir", required=True)
    parser.add_argument("-b", "--batch_size", default=32)
    parser.add_argument("--dev_batch_size", default=200)
    parser.add_argument("-M", "--max_facts", required=True)
    parser.add_argument("--max_text_facts", required=True)
    parser.add_argument("-m", "--min_facts", required=True)
    parser.add_argument("-i", "--hops", default=3)
    parser.add_argument("-d", "--embedding_dim", default=50)
    parser.add_argument("--entity_vocab_size", required=True)
    parser.add_argument("--relation_vocab_size", required=True)
    parser.add_argument("--learning_rate", required=True)
    parser.add_argument("--grad_clip_norm", required=True)
    parser.add_argument("--verbose", default=0)
    parser.add_argument("--dev_eval_counter", default=200)
    parser.add_argument("--save_counter", default=200)
    parser.add_argument("--dev_max_facts", default=15000)
    parser.add_argument("--dev_max_text_facts", default=15000)
    parser.add_argument("--output_dir", default='.')
    parser.add_argument("--load_model", default=0)
    parser.add_argument("--model_path", default='')
    parser.add_argument("--load_pretrained_vectors", default=0)
    parser.add_argument("--pretrained_vector_path", default='')
    parser.add_argument("--use_kb", default=1, type=int)
    parser.add_argument("--use_text", default=0, type=int)
    parser.add_argument("--print_attention_weights", default=0, type=int)
    parser.add_argument("--mode", default='train')
    parser.add_argument("--combine_text_kb_answer", default='concat2')
    parser.add_argument("--separate_key_lstm", default=0, type=int)

    args = parser.parse_args()
    entity_vocab_size = int(args.entity_vocab_size)
    relation_vocab_size = int(args.relation_vocab_size)
    train_file = args.train_file
    dev_file = args.dev_file
    kb_file = args.kb_file
    text_kb_file = args.text_kb_file
    vocab_dir = args.vocab_dir
    embedding_size = int(args.embedding_dim)
    batch_size = int(args.batch_size)
    dev_batch_size = int(args.dev_batch_size)
    min_facts = int(args.min_facts)
    max_facts = int(args.max_facts)
    max_text_facts = int(args.max_text_facts)
    lr = float(args.learning_rate)
    grad_clip_norm = int(args.grad_clip_norm)
    verbose = (int(args.verbose) == 1)
    hops = int(args.hops)
    dev_eval_counter = int(args.dev_eval_counter)
    save_counter = int(args.save_counter)
    dev_max_facts = int(args.dev_max_facts)
    dev_max_text_facts = int(args.dev_max_text_facts)
    output_dir = args.output_dir
    load_model = (int(args.load_model) == 1)
    model_path = args.model_path
    use_kb = (args.use_kb == 1)
    use_text = (args.use_text == 1)
    if load_model:
        assert len(model_path) != 0 or model_path is not None
    load_pretrained_vectors = (int(args.load_pretrained_vectors) == 1)
    pretrained_vector_path = args.pretrained_vector_path
    print_attention_weights = (args.print_attention_weights == 1)
    mode = args.mode
    combine_text_kb_answer = args.combine_text_kb_answer
    separate_key_lstm = (args.separate_key_lstm == 1)

    t = Trainer()
    t.fit()

from __future__ import division
import tensorflow as tf
import abc
import util


class QAbase(object):
    """
    Base class for Question Ansering
    """

    def __init__(self, entity_vocab_size, embedding_size, hops=3,
                 question_encoder='lstm', use_peepholes=True, load_pretrained_model=False,
                 load_pretrained_vectors=False, pretrained_entity_vectors=None, verbose=False):

        self.entity_vocab_size = entity_vocab_size
        self.embedding_size = embedding_size
        self.lstm_hidden_size = embedding_size
        self.question_encoder = question_encoder
        self.use_peepholes = use_peepholes
        self.hops = hops

        """Common Network parameters"""
        # projection
        self.W = tf.get_variable("W", shape=[self.embedding_size, 2 * self.embedding_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.Variable(tf.zeros([2 * self.embedding_size]), name="b")

        self.W1 = tf.get_variable("W1", shape=[2 * self.embedding_size, self.embedding_size],
                                  initializer=tf.contrib.layers.xavier_initializer())

        self.b1 = tf.Variable(tf.zeros([self.embedding_size]), name="b1")
        # weights for each hop of the memory network
        self.R = [tf.get_variable('R{}'.format(h), shape=[2 * self.embedding_size, 2 * self.embedding_size],
                                  initializer=tf.contrib.layers.xavier_initializer()) for h in range(self.hops)]
        self.attn_weights_all_hops = []
        # with tf.device('/cpu:0'):
        # embedding layer
        initializer_op = None
        trainable = False
        if load_pretrained_model:
            if verbose:
                print(
                    'Load pretrained model is set to {0} and hence entity_lookup_table trainable is set to {0}'.format(
                        load_pretrained_model))
                trainable = True
        if load_pretrained_vectors:
            if verbose:
                print('pretrained entity & word embeddings available. Initializing with them.')
            assert (pretrained_entity_vectors is not None)
            initializer_op = tf.constant_initializer(pretrained_entity_vectors)
        else:
            if verbose:
                print('No pretrained entity & word embeddings available. Learning entity embeddings from scratch')
                trainable = True
            initializer_op = tf.contrib.layers.xavier_initializer()

        self.entity_lookup_table = tf.get_variable("entity_lookup_table",
                                                   shape=[self.entity_vocab_size - 1, self.embedding_size],
                                                   dtype=tf.float32,
                                                   initializer=initializer_op, trainable=trainable)

        # dummy memory is set to -inf, so that during softmax for attention weight, we correctly
        # assign these slots 0 weight.
        self.entity_dummy_mem = tf.constant(0.0, shape=[1, self.embedding_size], dtype='float32')

        self.entity_lookup_table_extended = tf.concat(0, [self.entity_lookup_table, self.entity_dummy_mem])

        # for encoding question
        # with tf.variable_scope('q_forward'):
        self.q_fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size, use_peepholes=self.use_peepholes,
                                                 state_is_tuple=True)
        # with tf.variable_scope('q_backward'):
        self.q_bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size, use_peepholes=self.use_peepholes,
                                                 state_is_tuple=True)

    def get_question_embedding(self, question, question_lengths):
        """ encodes the question. Current implementation is encoding with biLSTM."""
        # question_word_embedding: [B, max_question_length, embedding_dim]
        question_word_embedding = tf.nn.embedding_lookup(self.entity_lookup_table_extended, question)
        question_word_embedding_shape = tf.shape(question_word_embedding)
        if self.question_encoder == 'lstm':
            scope_name = tf.get_variable_scope()
            with tf.variable_scope(scope_name, reuse=True):
                lstm_outputs, lstm_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.q_fw_cell,
                                                                                   cell_bw=self.q_bw_cell,
                                                                                   dtype=tf.float32,
                                                                                   inputs=question_word_embedding,
                                                                                   parallel_iterations=32,
                                                                                   sequence_length=question_lengths)
            # fwd_out, bwd_out: [batch_size, embedding_dim]
            fwd_out_all, bwd_out_all = lstm_outputs
            last_fwd = util.last_relevant(fwd_out_all, question_lengths)
            last_bwd = bwd_out_all[:, 0, :]
            # question_embedding: [B,2D]
            question_embedding = tf.concat(1, [last_fwd, last_bwd])
        else:
            raise NotImplementedError
        return question_embedding

    def get_key_embedding(self, *args, **kwargs):
        raise NotImplementedError

    def get_value_embedding(self, val_mem):
        # each is [B, max_num_slots, D]
        val_embedding = tf.nn.embedding_lookup(self.entity_lookup_table_extended, val_mem, name="val_embedding")
        return val_embedding

    def seek_attention(self, question_embedding, key, value, C, mask):
        """ Iterative attention. """
        for h in range(self.hops):
            expanded_question_embedding = tf.expand_dims(question_embedding, 1)
            # self.key*expanded_question_embedding [B, M, 2D]; self.attn_weights: [B,M]
            attn_logits = tf.reduce_sum(key * expanded_question_embedding, 2)
            attn_logits = tf.select(mask, attn_logits, C)
            self.attn_weights = tf.nn.softmax(attn_logits)
            self.attn_weights_all_hops.append(self.attn_weights)
            # self.p = tf.Print(attn_weights, [attn_weights], message='At hop {}'.format(h), summarize=10)
            # attn_weights_reshape: [B, M, 1]
            attn_weights_reshape = tf.expand_dims(self.attn_weights, -1)
            # self.value * attn_weights_reshape:[B, M, D]; self.attn_value:[B, D]
            attn_value = tf.reduce_sum(value * attn_weights_reshape, 1)
            # attn_value_proj : [B, 2D]
            # attn_value_proj = tf.nn.relu(tf.add(tf.matmul(attn_value, self.W), self.b))
            attn_value_proj = tf.add(tf.matmul(attn_value, self.W), self.b)
            sum = question_embedding + attn_value_proj
            # question_embedding: [B, 2D]
            question_embedding = tf.matmul(sum, self.R[h])
        return question_embedding

    # def seek_attention(self, question_embedding, key, value, C, mask):
    #    """ Iterative attention. """
    #    for h in range(self.hops):
    #        attn_logits = tf.einsum('ijk,ik->ij', key, question_embedding) # self.attn_weights: [B,M] 
    #        attn_logits = tf.select(mask, attn_logits, C)
    #        attn_weights = tf.nn.softmax(attn_logits)
    #        attn_value = tf.einsum('ijk,ij->ik',value,attn_weights) # self.attn_value:[B, D] 
    #        attn_value_proj = tf.add(tf.matmul(attn_value, self.W), self.b) # attn_value_proj : [B, 2D]
    #        total_emb = question_embedding + attn_value_proj
    #        question_embedding = tf.matmul(total_emb, self.R[h]) # question_embedding: [B, 2D]
    #    return question_embedding


    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class KBQA(QAbase):
    """
    Class for KB Question Answering
    TODO(rajarshd): describe input/output behaviour
    """

    def __init__(self, relation_vocab_size,
                 key_encoder='concat', **kwargs):
        super(KBQA, self).__init__(**kwargs)
        self.key_encoder = key_encoder
        self.relation_vocab_size = relation_vocab_size

        """Specialized Network parameters"""
        self.relation_lookup_table = tf.get_variable("relation_lookup_table", shape=[self.relation_vocab_size - 1,
                                                                                     self.embedding_size],
                                                     initializer=tf.contrib.layers.xavier_initializer())

        self.relation_dummy_mem = tf.constant(0.0, shape=[1, self.embedding_size], dtype='float32')

        self.relation_lookup_table = tf.concat(0, [self.relation_lookup_table, self.relation_dummy_mem])

    def get_key_embedding(self, entity, relation):
        """TODO(rajarshd): describe various options"""
        # each is [B, max_num_slots, D]
        e1_embedding = tf.nn.embedding_lookup(self.entity_lookup_table_extended, entity, name="e1_embedding")
        r_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, relation, name="r_embedding")

        # key shape is [B, max_num_slots, 2D]
        if self.key_encoder == 'concat':
            key = tf.concat(2, [e1_embedding, r_embedding])
        else:
            raise NotImplementedError
        return key

    def __call__(self, memory, question, question_lengths):
        # split memory and get corresponding embeddings
        e1, r, e2 = tf.unpack(memory, axis=2)
        C = tf.ones_like(e1, dtype='float32') * -1000
        mask = tf.not_equal(e1, self.entity_vocab_size - 1)
        key = self.get_key_embedding(e1, r)
        value = self.get_value_embedding(e2)
        ques = self.get_question_embedding(question, question_lengths)

        # get attention on retrived informations based on the question
        attn_ques = self.seek_attention(ques, key, value, C, mask)

        # output embeddings - share with entity lookup table
        # B = tf.slice(self.entity_lookup_table, [0, 0], [1789936, -1])
        B = self.entity_lookup_table_extended
        # project down
        model_answer = tf.add(tf.matmul(attn_ques, self.W1), self.b1)  # model_answer: [B, D]
        logits = tf.matmul(model_answer, B, transpose_b=True, name='ent_mul_manzil')  # scores: [B, num_entities]
        return logits


class TextQA(QAbase):
    """
    Class for QA with Text only
    TODO(rajarshd): describe input/output behaviour
    """

    def __init__(self, key_encoder='lstm',
                 separate_key_lstm=False, **kwargs):
        super(TextQA, self).__init__(**kwargs)
        self.key_encoder = key_encoder
        self.separate_key_lstm = separate_key_lstm

        """Specialized Network parameters"""
        # for encoding key
        if self.separate_key_lstm:
            with tf.variable_scope('k_forward'):
                self.k_fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size, use_peepholes=self.use_peepholes,
                                                         state_is_tuple=True)
            with tf.variable_scope('k_backward'):
                self.k_bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size, use_peepholes=self.use_peepholes,
                                                         state_is_tuple=True)

    def get_key_embedding(self, key_mem, key_lens):
        """TODO(rajarshd): describe various options"""
        # each is [B, max_num_slots, max_key_len, D]
        key_embedding = tf.nn.embedding_lookup(self.entity_lookup_table_extended, key_mem, name="key_embedding")
        # reshape the data to [(B, max_num_slots), max_key_len, D]
        dims = tf.shape(key_embedding)
        key_embedding_reshaped = tf.reshape(key_embedding, [-1, dims[2], self.embedding_size])
        key_len_reshaped = tf.reshape(key_lens, [-1])
        if self.key_encoder == 'lstm':
            scope_name = tf.get_variable_scope()
            with tf.variable_scope(scope_name, reuse=None):
                if self.separate_key_lstm:
                    lstm_key_outputs, lstm_key_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.k_fw_cell,
                                                                                               cell_bw=self.k_bw_cell,
                                                                                               dtype=tf.float32,
                                                                                               inputs=key_embedding_reshaped,
                                                                                               parallel_iterations=32,
                                                                                               sequence_length=key_len_reshaped)
                else:
                    lstm_key_outputs, lstm_key_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.q_fw_cell,
                                                                                               cell_bw=self.q_bw_cell,
                                                                                               dtype=tf.float32,
                                                                                               inputs=key_embedding_reshaped,
                                                                                               parallel_iterations=32,
                                                                                               sequence_length=key_len_reshaped)

            lstm_key_output_states_fw, lstm_key_output_states_bw = lstm_key_output_states
            fw_c, fw_h = lstm_key_output_states_fw
            bw_c, bw_h = lstm_key_output_states_bw
            # [(B, max_num_slots), 2D]
            key = tf.reshape(tf.concat(1, [fw_h, bw_h]), [-1, dims[1], 2 * self.embedding_size])
        else:
            raise NotImplementedError
        return key

    def __call__(self, key_mem, key_len, val_mem, question, question_lengths):
        # key_mem is [B, max_num_mem, max_key_len]
        # key_len is [B, max_num_mem]
        # val_mem is [B, max_num_mem]

        C = tf.ones_like(key_len, dtype='float32') * -1000
        mask = tf.not_equal(key_len, 0)
        key = self.get_key_embedding(key_mem, key_len)
        value = self.get_value_embedding(val_mem)

        ques = self.get_question_embedding(question, question_lengths)

        # get attention on retrived informations based on the question
        attn_ques = self.seek_attention(ques, key, value, C, mask)

        # output embeddings - share with entity lookup table
        B = self.entity_lookup_table_extended
        # project down
        model_answer = tf.add(tf.matmul(attn_ques, self.W1), self.b1)  # model_answer: [B, D]
        logits = tf.matmul(model_answer, B, transpose_b=True, name='ent_mul_manzil')  # scores: [B, num_entities]
        return logits


class TextKBQA(QAbase):
    """
    Class for QA with Text+KB
    """

    def __init__(self, relation_vocab_size,
                 kb_key_encoder='concat',
                 text_key_encoder='lstm',
                 join='concat2',
                 separate_key_lstm=False, **kwargs):
        super(TextKBQA, self).__init__(**kwargs)
        self.join = join
        self.kb_key_encoder = kb_key_encoder
        self.text_key_encoder = text_key_encoder
        self.separate_key_lstm = separate_key_lstm
        self.relation_vocab_size = relation_vocab_size

        """Specialized Network parameters"""
        # projection
        self.relation_lookup_table = tf.get_variable("relation_lookup_table", shape=[self.relation_vocab_size - 1,
                                                                                     self.embedding_size],
                                                     initializer=tf.contrib.layers.xavier_initializer())

        self.relation_dummy_mem = tf.constant(0.0, shape=[1, self.embedding_size], dtype='float32')

        self.relation_lookup_table_extended = tf.concat(0, [self.relation_lookup_table, self.relation_dummy_mem])

        # for encoding key
        if self.separate_key_lstm:
            with tf.variable_scope('k_forward'):
                self.k_fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size, use_peepholes=self.use_peepholes,
                                                         state_is_tuple=True)
            with tf.variable_scope('k_backward'):
                self.k_bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size, use_peepholes=self.use_peepholes,
                                                         state_is_tuple=True)

    def get_key_embedding(self, entity, relation, key_mem, key_lens):
        # each is [B, max_num_slots, D]
        e1_embedding = tf.nn.embedding_lookup(self.entity_lookup_table_extended, entity, name="e1_embedding")
        r_embedding = tf.nn.embedding_lookup(self.relation_lookup_table_extended, relation, name="r_embedding")

        # key shape is [B, max_num_slots, 2D]
        if self.kb_key_encoder == 'concat':
            kb_key = tf.concat(2, [e1_embedding, r_embedding])
        else:
            raise NotImplementedError

        # each is [B, max_num_slots, max_key_len, D]
        key_embedding = tf.nn.embedding_lookup(self.entity_lookup_table_extended, key_mem, name="key_embedding")
        # reshape the data to [(B, max_num_slots), max_key_len, D]
        dims = tf.shape(key_embedding)
        key_embedding_reshaped = tf.reshape(key_embedding, [-1, dims[2], self.embedding_size])
        key_len_reshaped = tf.reshape(key_lens, [-1])
        if self.text_key_encoder == 'lstm':
            scope_name = tf.get_variable_scope()
            with tf.variable_scope(scope_name, reuse=None):
                if self.separate_key_lstm:
                    lstm_key_outputs, lstm_key_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.k_fw_cell,
                                                                                               cell_bw=self.k_bw_cell,
                                                                                               dtype=tf.float32,
                                                                                               inputs=key_embedding_reshaped,
                                                                                               parallel_iterations=32,
                                                                                               sequence_length=key_len_reshaped)
                else:
                    lstm_key_outputs, lstm_key_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.q_fw_cell,
                                                                                               cell_bw=self.q_bw_cell,
                                                                                               dtype=tf.float32,
                                                                                               inputs=key_embedding_reshaped,
                                                                                               parallel_iterations=32,
                                                                                               sequence_length=key_len_reshaped)

            lstm_key_output_states_fw, lstm_key_output_states_bw = lstm_key_output_states
            fw_c, fw_h = lstm_key_output_states_fw
            bw_c, bw_h = lstm_key_output_states_bw
            # [(B, max_num_slots), 2D]
            text_key = tf.reshape(tf.concat(1, [fw_h, bw_h]), [-1, dims[1], 2 * self.embedding_size])
        else:
            raise NotImplementedError
        return kb_key, text_key

    def __call__(self, memory, key_mem, key_len, val_mem, question, question_lengths):
        # split memory and get corresponding embeddings
        e1, r, e2 = tf.unpack(memory, axis=2)
        kb_C = tf.ones_like(e1, dtype='float32') * -1000
        kb_mask = tf.not_equal(e1, self.entity_vocab_size - 1)
        kb_value = self.get_value_embedding(e2)

        # key_mem is [B, max_num_mem, max_key_len]
        # key_len is [B, max_num_mem]
        # val_mem is [B, max_num_mem]
        text_C = tf.ones_like(key_len, dtype='float32') * -1000
        text_mask = tf.not_equal(key_len, 0)
        text_value = self.get_value_embedding(val_mem)

        kb_key, text_key = self.get_key_embedding(e1, r, key_mem, key_len)
        ques = self.get_question_embedding(question, question_lengths)

        # get attention on retrived informations based on the question
        kb_attn_ques = self.seek_attention(ques, kb_key, kb_value, kb_C, kb_mask)  # [B, 2D]
        text_attn_ques = self.seek_attention(ques, text_key, text_value, text_C, text_mask)  # [B, 2D]

        if self.join == 'batch_norm':
            mean_kb_key, var_kb_key = tf.nn.moments(kb_key, axes=[0,1])
            mean_kb_value, var_kb_value = tf.nn.moments(kb_value, axes=[0,1])
            mean_text_key, var_text_key = tf.nn.moments(kb_key, axes=[0,1])
            mean_text_value, var_text_value = tf.nn.moments(kb_value, axes=[0,1])
            text_key = tf.nn.batch_normalization(text_key, mean_text_key, var_text_key, mean_kb_key, var_kb_key, 1e-8)
            text_value = tf.nn.batch_normalization(text_value, mean_text_value, var_text_value, mean_kb_value, var_kb_value, 1e-8)

            merged_key = tf.concat(1, [kb_key, text_key])
            merged_value = tf.concat(1, [kb_value, text_value])
            merged_C = tf.concat(1, [kb_C, text_C])
            merged_mask = tf.concat(1, [kb_mask, text_mask])

            # get attention on retrived informations based on the question
            attn_ques = self.seek_attention(ques, merged_key, merged_value, merged_C, merged_mask)  # [B, 2D]
            model_answer = tf.add(tf.matmul(attn_ques, self.W1), self.b1)  # model_answer: [B, D]

        # output embeddings - share with entity lookup table
        B = self.entity_lookup_table_extended
        logits = tf.matmul(model_answer, B, transpose_b=True, name='ent_mul_manzil')  # scores: [B, num_entities]
        return logits

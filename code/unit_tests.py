from feed_data import Batcher
import sys


def test_batcher():

    train_file = "/iesl/canvas/rajarshi/data/TextKBQA/small_train_with_facts.json"
    kb_file = "/iesl/canvas/rajarshi/data/TextKBQA/freebase.spades.txt"
    batch_size = 32
    vocab_dir = "/home/rajarshi/research/joint-text-and-kb-inference-semantic-parsing/vocab/"
    min_num_mem_slots = 100
    max_num_mem_slots = 500
    batcher = Batcher(train_file, kb_file, batch_size, vocab_dir,
                               min_num_mem_slots=min_num_mem_slots, max_num_mem_slots=max_num_mem_slots,
                               return_one_epoch=True, shuffle=False)
    batch_counter = 0
    for data in batcher.get_next_batch():
        batch_counter += 1
        batch_question, batch_q_lengths, batch_answer, batch_memory, batch_num_memories = data

    print("####### Test1: Checking number of batches returned#########")
    assert batch_counter == 1
    print("Test passed!")

    batch_size = 19
    batcher.batch_size = batch_size
    batcher.reset()
    batch_counter = 0
    for data in batcher.get_next_batch():
        batch_counter += 1
        batch_question, batch_q_lengths, batch_answer, batch_memory, batch_num_memories = data

    print("####### Test2: Checking number of batches returned with different batch size #########")
    print(batch_counter)
    assert batch_counter == 2
    print("Test passed!")

    batch_size = 20
    batcher.batch_size = batch_size
    batcher.reset()
    for data in batcher.get_next_batch():
        batch_counter += 1
        batch_question, batch_q_lengths, batch_answer, batch_memory, batch_num_memories = data
        print(batch_question[0])
        print(batch_answer[0])
        sys.exit(1)



if __name__ == '__main__':
    test_batcher()









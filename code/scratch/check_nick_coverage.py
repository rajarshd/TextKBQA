import json
vocab_file="/iesl/canvas/nmonath/data/wikipedia/20160305/en/enwiki-20160305-lstm-vocab.tsv"
entity_to_freebase_mapping="/iesl/canvas/nmonath/data/freebase/20160513/en-freebase_wiki_cat_title_map.txt"


#Read the entity_to_freebase_mapping file
print('Read the entity_to_freebase_mapping file')
name_to_mid = {}
with open(entity_to_freebase_mapping) as input_file:
    for line in input_file:
        line = line.strip()
        name, mid = line.split('\t')
        name_to_mid[name] = mid

#read the vocab file
# print('read nick\'s vocab file')
# vocab_set = set()
# with open(vocab_file) as vocab_in:
#     for line in vocab_in:
#         line = line.strip()
#         _, name = line.split(' ')
#         if name in name_to_mid:
#             vocab_set.add(name_to_mid[name].replace('/', '.'))
#         else:
#             vocab_set.add(name)

#read the embedding file and read only the words (1st column)
embeddings_file_from_nick = "/iesl/canvas/nmonath/data/wikipedia/20160305/en/embeddings/aabt/context/target.tsv"
vocab_set = set()
with open(embeddings_file_from_nick) as embedding_file:
    for line in embedding_file:
        split = line.split('\t')
        name = split[0]
        if name in name_to_mid:
            vocab_set.add(name_to_mid[name].replace('/', '.'))
        else:
            name = name.replace('W_SLUG_', '')
            name = name.replace('_lang_EN', '')
            vocab_set.add(name.lower())


#read entity vocab and check coverage
print('Reading entity vocab')
entity_vocab_file = "/home/rajarshi/research/joint-text-and-kb-inference-semantic-parsing/vocab/entity_vocab.json"
entity_vocab = {}
with open(entity_vocab_file) as entity_vocab_in:
    entity_vocab = json.load(entity_vocab_in)

counter = 0
mid_counter = 0
for k, _ in entity_vocab.iteritems():
    k = k.lower()
    if k in vocab_set:
        if k.startswith('m.'):
            mid_counter += 1
        counter += 1

print('Total overlap is {}'.format(counter))
print('Overlap of entities is {}'.format(mid_counter))
import json
import os

def write_to_file(sentence_list, fout):
    for sentence in sentence_list:
        fout.write(sentence+'\n')

dir_name = "/iesl/local/rajarshi/clueweb_siva/"
clueweb_files = ['spadesClueWeb09_1.1', 'spadesClueWeb09_1.2',
                 'spadesClueWeb09_1.3', 'spadesClueWeb09_1.wiki']
LIST_MAX = 10000
counter = 0
output_file = dir_name+'all_sentences'
if os.path.exists(output_file):
    os.remove(output_file)

fout = open(output_file, 'a')
for file_name in clueweb_files:
    file_path = dir_name + file_name
    print('Reading file {}'.format(file_path))
    sentence_list = []
    with open(file_path) as f_in:
        for line in f_in:
            line = line.strip()
            json_line = json.loads(line)
            entities = json_line['entities']
            indices = [entity['index'] for entity in entities]
            mids = [entity['entity'] for entity in entities]
            words = json_line['words']
            s_words = [word['word'] for word in words]
            for c, index in enumerate(indices):
                s_words[index] = mids[c]
            sentence = ' '.join(s_words)
            sentence = sentence.strip()
            sentence_list.append(sentence)
            if len(sentence_list) > LIST_MAX:
                write_to_file(sentence_list, fout)
                sentence_list = []
                counter = counter+1
                print('Wrote {} lines'.format(LIST_MAX*counter))
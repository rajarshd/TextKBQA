import json
from collections import defaultdict

def get_coverage(input_file):
    entities_in_file = set()
    with open(input_file) as f_in:
        for line in f_in:
            line = line.strip()
            json_line = json.loads(line)
            if 'entities' not in json_line:
                continue
            entity_list = json_line['entities']
            for entity in entity_list:
                if 'entity' not in entity:
                    continue
                entities_in_file.add(entity['entity'])

    print('Total number of entities on file are {}'.format(len(entities_in_file)))
    hit = 0
    total = 0
    for entity in entities_in_file:
        if entity in entity_count_map:
            hit += 1
        total += 1

    print('Total {}, hit {}'.format(total, hit))


print('Gathering entities in clueweb file...')
dir_name = "/iesl/local/rajarshi/clueweb_siva/"
clueweb_files = ['spadesClueWeb09_1.1', 'spadesClueWeb09_1.2', 'spadesClueWeb09_1.3', 'spadesClueWeb09_1.wiki']

entity_count_map = defaultdict(int)
for file_name in clueweb_files:
    file_path = dir_name + file_name
    print('Reading file {}'.format(file_path))
    with open(file_path) as f_in:
        for line in f_in:
            line = line.strip()
            json_line = json.loads(line)
            if 'entities' not in json_line:
                continue
            entity_list = json_line['entities']
            for entity in entity_list:
                if 'entity' not in entity:
                    continue
                entity_count_map[entity['entity']] += 1

print('Total Num entities {}'.format(len(entity_count_map)))

project_dir = '/home/rajarshi/canvas/data/TextKBQA/'
train_file_name = 'train_with_facts.json'
dev_file_name = 'dev_with_facts.json'
test_file_name = 'test.json'
print('Checking coverage in train file')
get_coverage(project_dir + train_file_name)

print('Checking coverage in dev file')
get_coverage(project_dir + dev_file_name)

print('Checking coverage in test file')
get_coverage(project_dir + test_file_name)

def parse_line(line):
    line = line.strip().split()
    sub = line[0]
    rel = line[1]
    obj = line[2]
    val = [1]
    if len(line) > 3:
        if line[3] == '-1':
            val = [-1]
    return sub, obj, rel, val


def load_triples_from_txt(filename, entity_indexes=None, relation_indexes=None, parse_line=parse_line):
    if entity_indexes is None:
        entity_indexes = {}
        next_entity = 0
    else:
        next_entity = max(entity_indexes.values()) + 1

    if relation_indexes is None:
        relation_indexes = {}
        next_relation = 0
    else:
        next_relation = max(relation_indexes.values()) + 1

    data = dict()

    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        sub, obj, rel, val = parse_line(line)

        if sub not in entity_indexes:
            entity_indexes[sub] = next_entity
            next_entity += 1
        sub_ind = entity_indexes[sub]

        if obj not in entity_indexes:
            entity_indexes[obj] = next_entity
            next_entity += 1
        obj_ind = entity_indexes[obj]

        if rel not in relation_indexes:
            relation_indexes[rel] = next_relation
            next_relation += 1
        rel_ind = relation_indexes[rel]

        data[(sub_ind, rel_ind, obj_ind)] = val

    return data, entity_indexes, relation_indexes


def build_data(name='WN18RR', path='./data'):
    folder = path + '/' + name + '/'


    train_triples, entity_indexes, relation_indexes = load_triples_from_txt(
        folder + 'train.txt', entity_indexes=None, relation_indexes=None, parse_line=parse_line
    )


    valid_triples, entity_indexes, relation_indexes = load_triples_from_txt(
        folder + 'valid.txt', entity_indexes=entity_indexes, relation_indexes=relation_indexes, parse_line=parse_line
    )


    test_triples, entity_indexes, relation_indexes = load_triples_from_txt(
        folder + 'test.txt', entity_indexes=entity_indexes, relation_indexes=relation_indexes, parse_line=parse_line
    )

    num_entities = len(entity_indexes)
    num_relations = len(relation_indexes)


    return (
        train_triples, valid_triples, test_triples,
        entity_indexes,
        relation_indexes,
    )


def read_to_dict(file_path):
        result = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.rsplit(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    try:

                        result[key] = int(value)
                    except ValueError:
                        print(f"error")
        return result


def build_DB15K(name='DB15K', path='./data'):
    folder = path + '/' + name + '/'
    train_path = folder + 'train2id.txt'
    valid_path = folder + 'valid2id.txt'
    test_path = folder + 'test2id.txt'
    entity_path = folder + 'entity2id.txt'
    relation_path = folder + 'relation2id.txt'


    def read_triples(file_path):
        data = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:

                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        sub_ind = int(parts[0])
                        rel_ind = int(parts[1])
                        obj_ind = int(parts[2])
                        data[(sub_ind, rel_ind, obj_ind)] = 1
                    except ValueError:
                        print(f"error")
                else:
                    print(f"error")
        return data
    train = read_triples(train_path)
    valid = read_triples(valid_path)
    test = read_triples(test_path)

    entity_indexes = read_to_dict(entity_path)
    relation_indexes = read_to_dict(relation_path)

    return train, valid, test, entity_indexes, relation_indexes

def build_MKG(name='MKG', path='./data'):
    folder = path + '/' + name + '/'
    train_path = folder + 'train2id.txt'
    valid_path = folder + 'valid2id.txt'
    test_path = folder + 'test2id.txt'

    entity_path = folder + 'entity2id.txt'
    relation_path = folder + 'relation2id.txt'
    def read_triples(file_path):
        data = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        sub_ind = int(parts[0])
                        rel_ind = int(parts[2])
                        obj_ind = int(parts[1])
                        data[(sub_ind, rel_ind, obj_ind)] = 1
                    except ValueError:
                        print(f"error")
        return data

    train = read_triples(train_path)
    valid = read_triples(valid_path)
    test = read_triples(test_path)



    entity_indexes = read_to_dict(entity_path)
    relation_indexes = read_to_dict(relation_path)

    return train, valid, test, entity_indexes, relation_indexes



def build_KT(name='Kuai16K', path='./data'):
    folder = path + '/' + name + '/'
    train_path = folder + 'train2id.txt'
    valid_path = folder + 'valid2id.txt'
    test_path = folder + 'test2id.txt'

    entity_path = folder + 'entity2id.txt'
    relation_path = folder + 'relation2id.txt'

    def read_triples(file_path):
        data = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        sub_ind = int(parts[0])
                        rel_ind = int(parts[2])
                        obj_ind = int(parts[1])
                        data[(sub_ind, rel_ind, obj_ind)] = 1
                    except ValueError:
                        print(f"error")
        return data

    train = read_triples(train_path)
    valid = read_triples(valid_path)
    test = read_triples(test_path)



    entity_indexes = read_to_dict(entity_path)
    relation_indexes = read_to_dict(relation_path)

    return train, valid, test, entity_indexes, relation_indexes



def get_doubles(train, valid, test, relation_indexes):
    num_relations = len(relation_indexes)  # 237
    new_relation_indexes = {}
    for rel_name, rel_id in relation_indexes.items():
        new_relation_indexes[rel_name] = rel_id
        reverse_rel_name = rel_name + "_reverse"
        reverse_rel_id = rel_id + num_relations  # 237 ~ 473
        new_relation_indexes[reverse_rel_name] = reverse_rel_id

    train_doubles = []
    count=0
    for (h, r, t) in train.keys():
        # 原始三元组
        if count<=69319:#Y:21310    W:34130  DB15K:69319
            train_doubles.append((h, r, t, 0))
            train_doubles.append((t, r + num_relations, h,0))
        else:
            train_doubles.append((h, r, t, 1))
            train_doubles.append((t, r + num_relations, h, 1))
        count +=1
    print(len(train_doubles))
    valid_doubles = []
    for (h, r, t) in valid.keys():
        valid_doubles.append((h, r, t,0))
        valid_doubles.append((t, r + num_relations, h,0))

    test_doubles = []
    for (h, r, t) in test.keys():
        test_doubles.append((h, r, t,0))
        test_doubles.append((t, r + num_relations, h,0))
    all_entities = [h for h, r, t,_ in train_doubles] + [t for h, r, t,_ in train_doubles]
    all_relations = [r for h, r, t,_ in train_doubles]

    max_ent = max(all_entities)
    max_rel = max(all_relations)
    return train_doubles, valid_doubles, test_doubles, new_relation_indexes



if __name__ == '__main__':
    data_name = "DB15K"
    if data_name == "FB15k-237" or data_name == "fb15k-237":
        train, valid, test, entity_indexes, relation_indexes = build_data(name=data_name)
    elif data_name == "WN18RR" or data_name == "wn18rr":
        train, valid, test, entity_indexes, relation_indexes = build_data(name=data_name)
    elif data_name == "DB15K" or data_name == "db15k":
        train, valid, test, entity_indexes, relation_indexes = build_DB15K(name=data_name)
    elif data_name == "DB15K-2":
        train, valid, test, entity_indexes, relation_indexes = build_DB15K(name=data_name)
    elif data_name == "TIVA" or data_name == "Kuai16K":
        train, valid, test, entity_indexes, relation_indexes = build_KT(name=data_name)
    elif data_name == "MKG-W" or data_name == "MKG-Y":
        train, valid, test, entity_indexes, relation_indexes = build_MKG(name=data_name)
    else:
        print("No such dataset")

    train_doubles, valid_doubles, test_doubles, relation_indexes = get_doubles(train, valid, test,
                                                                               relation_indexes)
    save_file = "./data/" + data_name + "/train_doubles2id.txt"
    doubles = train_doubles

    def save_my_file(save_file, doubles):
        with open(save_file, 'w', encoding='utf-8') as f:
            for triple in doubles:
                line = ' '.join(map(str, triple)) + '\n'
                f.write(line)

    save_my_file(save_file, doubles)



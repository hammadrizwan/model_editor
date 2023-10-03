import random, itertools, json




def create_dataset_from_file(file_path,dsplits):
    data = []

    # Open the JSONL file and read it line by line.
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON object in each line.
            try:
                json_object = json.loads(line.strip())
                data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {str(e)}")

    train_x=[]
    paraphrase_x=[]
    local_neutral_x_train=[]
    local_neutral_x_test=[]

    increament_factor=1/dsplits
    for index in range(dsplits):
        tx,px,lnxtr,lnxte=process_dataset(data[:50])#int(increament_factor*len(data))])
        train_x.append(tx)
        paraphrase_x.append(px)
        local_neutral_x_train.append(lnxtr)
        local_neutral_x_test.append(lnxte)
        break
    return train_x, paraphrase_x, local_neutral_x_train, local_neutral_x_test

def process_dataset(data,limit=4):
    train_x={}
    paraphrase_x={}
    local_neutral_x_train={}
    local_neutral_x_test={}
    control=0
    record_number=0
    for row in data:
        if(control==0):#construct train and paraphrase dataset
            random.shuffle(row["entity1"])
            random.shuffle(row["entity2"])
            random.shuffle(row["relation"])

            split_point_entity_1 = len(row["entity1"]) // 2
            split_point_entity_2 = len(row["entity2"]) // 2
            split_point_relation = len(row["relation"]) // 2

            entity1_train = row["entity1"][:split_point_entity_1]
            entity1_paraphrase = row["entity1"][split_point_entity_1:]

            entity2_train = row["entity2"][:split_point_entity_2]
            entity2_paraphrase = row["entity2"][split_point_entity_2:]

            train_relation = row["relation"][:split_point_relation]
            paraphrase_relation = row["relation"][split_point_relation:]

            template = "{} has relation \'{}\' to {}."

            # Generate all possible combinations of elements from the two lists
            combinations_train = list(itertools.product(entity1_train, entity2_train,train_relation))
            combinations_paraphrase = list(itertools.product(entity1_paraphrase, entity2_paraphrase,paraphrase_relation))


            # Fill the template with each combination
            train_x[record_number]=[template.format(item1, item2, item3) for item1, item2,item3 in combinations_train]
            random.shuffle(train_x[record_number])
            train_x[record_number]=train_x[record_number][:limit]
            paraphrase_x[record_number]=[template.format(item1, item2, item3) for item1, item2,item3 in combinations_paraphrase]
            random.shuffle(paraphrase_x[record_number])
            paraphrase_x[record_number]=paraphrase_x[record_number][:limit]
            control+=1

        elif(control==1):
            random.shuffle(row["entity1"])
            random.shuffle(row["entity2"])
            random.shuffle(row["relation"])
            combinations_local_neutral = list(itertools.product(row["entity1"], row["entity2"],row["relation"]))
            local_neutral_x_train[record_number]= [template.format(item1, item2, item3) for item1, item2,item3 in combinations_local_neutral]
            random.shuffle(local_neutral_x_train[record_number])
            local_neutral_x_test[record_number]=local_neutral_x_train[record_number][:limit]
            local_neutral_x_train[record_number]=local_neutral_x_train[record_number][limit:limit*2]
            local_neutral_x_test
            control-=1
            record_number+=1

    return train_x, paraphrase_x, local_neutral_x_train, local_neutral_x_test



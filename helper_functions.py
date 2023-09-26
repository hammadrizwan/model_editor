import random
from itertools import product,permutations
from shared_imports import tqdm, os, torch

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


def get_first_val_unique_groups(groups):
  unique_first_values = set([item[0] for item in groups])

  # Step 2: Randomly select an entry for each unique first value
  random_values = []
  for first_value in unique_first_values:
      entries_with_first_value = [item for item in groups if item[0] == first_value]
      random_entry = random.choice(entries_with_first_value)
      random_values.append(random_entry)
  return random_values



def create_siames_dataset(x_pos,x_neg):
  dataset=[]
  for k, v in tqdm(x_pos.items()):
    incremented_list_pos = [i for i in range(0, len(v))]
    groups_pos=[list(val) for val in list(permutations(incremented_list_pos, 2))]
    positives_random=get_first_val_unique_groups(groups_pos)

    positives=[]
    for pair in positives_random:
      positives.append([v[pair[0]],v[pair[1]],torch.tensor(1)])

    incremented_list_neg = [i for i in range(0, len(x_neg[k]))]
    groups_neg=[list(val) for val in list(product(incremented_list_neg, incremented_list_pos))]
    negative_random=get_first_val_unique_groups(groups_neg)
    negatives=[]
    random.shuffle(negative_random)
    for pair in negative_random:#[:len(positives)]:#reduce if too many negative samples
      negatives.append([x_neg[k][pair[0]],v[pair[1]],torch.tensor(0)])

    dataset.extend(positives)#positive samples combinations
    dataset.extend(negatives)#positive samples combinations
  return dataset

def embedding_dict_flatten(embedding_dict):
  embedding_list=[]
  for key in embedding_dict.keys():
      embedding_list.extend(embedding_dict[key])
  return embedding_list
from shared_imports import pd



def read_from_file(file_path):
    dataset_df = pd.read_json(path_or_buf=file_path)


def convert_format(dataset_df):
    requested_rewrites={}
    paraphrase_prompts={}
    neighborhood_prompts_ln={}
    target_old={}
    target_new={}
    for index,val in enumerate(jsonObj["requested_rewrite"].values.tolist()):
        requested_rewrites[index]=val["prompt"].format(val['subject'])
        target_old[index]=val["target_true"]["str"]
        target_new[index]=val["target_new"]["str"]

    for index,val in enumerate(jsonObj["paraphrase_prompts"].values.tolist()):
        if(index not in paraphrase_prompts):
            paraphrase_prompts[index]=[]
        paraphrase_prompts[index].extend(val)

    for index,val in enumerate(jsonObj["neighborhood_prompts"].values.tolist()):
        if(index not in neighborhood_prompts_ln):
            neighborhood_prompts_ln[index]=[]
        neighborhood_prompts_ln[index].extend(val)

    requested_paraphrase={}
    for key in requested_rewrites.keys():
        requested_paraphrase[key]=[requested_rewrites[key]]+paraphrase_prompts[key]
    
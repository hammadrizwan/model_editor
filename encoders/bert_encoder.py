
from transformers import BertTokenizer, BertModel
from shared_imports import torch, tqdm, np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'bert-base-uncased'  # You can use a different BERT model if needed
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name,output_hidden_states=True).to(device)


def get_embeddings(data,layer=6,token_number=1,flatten=True):
  if(flatten):
    embeddings=[]
  else:
    embeddings={}
  for key, samples in tqdm(data.items(), desc="Processing"):
    # Tokenize the input text
    for sample in samples:
      encoded_input = tokenizer(sample, return_tensors='pt', padding=True, truncation=True).to(device)
      with torch.no_grad():
        outputs = bert_model(**encoded_input)
        # print(len(outputs.hidden_states))#[11][0][1])
      # Extract embeddings from block 6 (index 5, as indexing starts from 0)
      if(flatten):
        matrix=outputs.hidden_states[layer][:, token_number, :].detach().cpu().numpy()
        list_of_arrays=[np.array(row) for row in matrix]
        # outputs.hidden_states[6][0][1].detach().cpu().numpy()
        embeddings.extend(list_of_arrays)#outputs.last_hidden_state[0][5].detach().cpu().numpy())
      else:
        if(key not in embeddings):
          embeddings[key]=[]
        matrix=outputs.hidden_states[layer][:, token_number, :].detach().cpu()#.numpy()
        list_of_tensors = torch.split(matrix, 1, dim=0)
        list_of_tensors = [tensor.squeeze() for tensor in list_of_tensors]
        # list_of_arrays=[np.array(row) for row in matrix]
        embeddings[key].extend(list_of_tensors)
  return embeddings
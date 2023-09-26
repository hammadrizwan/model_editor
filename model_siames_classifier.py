
from shared_imports import *

class SiameseClassificationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1=256, hidden_size2=128, num_classes=2):
        super(SiameseClassificationNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(hidden_size2, num_classes)
        self.sm = nn.Softmax()

    def forward_sequential(self, x):
        return self.fc(x)

    def forward(self, input1, input2):
        output1 = self.forward_sequential(input1)
        output2 = self.forward_sequential(input2)
        output3 = self.sm(self.fc1(output1))

        return output1, output2, output3
    

class CosineSimilarityLoss_SentenceTransformers(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss_SentenceTransformers, self).__init__()
        self.loss_fct = torch.nn.MSELoss()
        self.cos_score_transformation=nn.Identity()

    def forward(self, output1, output2, target):
        score= torch.cosine_similarity(output1.to(dtype=torch.float), output2.to(dtype=torch.float))
        score = self.cos_score_transformation(score)
        loss=self.loss_fct(score, target.to(dtype=torch.float).view(-1))
        return loss


class Siames_Classifier_Dataset(Dataset):
    def __init__(self, dataset):
        self.input_embeddings1 = [val[0] for val in dataset]
        self.input_embeddings2 = [val[1] for val in dataset]
        self.labels = [val[2] for val in dataset]

    def __len__(self):
        return len(self.input_embeddings1)

    def __getitem__(self, index):
        emb1 = self.input_embeddings1[index]
        emb2 = self.input_embeddings2[index]

        label = self.labels[index]

        return emb1, emb2, label
    

def train_model_combined(model,optimizer,data_loader,criterion1,criterion2,criterion1_weight=0.8,num_epochs = 70):
  
  lowest_error = float('inf')
  best_model_weights = None
  for epoch in range(num_epochs):
      for batch in data_loader:
          embs1,embs2, labels = batch
          optimizer.zero_grad()
          output1, output2, output3 = model(embs1,embs2)
          loss_classification = criterion1(output3, labels)
          loss_semantic_sim = criterion2(output1,output2, labels)
          combined_loss = (criterion1_weight * loss_classification) + ((1-criterion1_weight) * loss_semantic_sim)
          combined_loss.backward()
          optimizer.step()
          if combined_loss.item() < lowest_error:
            lowest_error = combined_loss.item()
            best_model_weights = model.state_dict()
      print(f'Epoch [{epoch + 1}/{num_epochs}] - Combined loss: {combined_loss.item():.4f} - Classification loss: {loss_classification.item():.4f} - Semantic Sim loss: {loss_semantic_sim.item():.4f}')
  torch.save(best_model_weights, './model_wieghts/best_model_weights_siames_classifier_test.pth')
  print('Training finished.')


def get_predictions_combind(model,dataset):
  model.eval()
  predicted_labels=[]
  with torch.no_grad():
    for index,data in enumerate(dataset):
      emb1,emb2,label=data
      outputs = model(emb1.view(1, -1),emb2.view(1, -1))
      out, inds = torch.max(outputs[2],dim=1)
      pred=inds.cpu().numpy()
      predicted_labels.extend(pred)
  return predicted_labels

def create_dataloader_siames(dataset, batch_size=32):
  scotus_bert_embedded_dataset = Siames_Classifier_Dataset(dataset)
  return DataLoader(scotus_bert_embedded_dataset, batch_size=batch_size, shuffle=True)
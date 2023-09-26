
from shared_imports import *


class FeedForwardClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2, num_classes):
        super(FeedForwardClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sm(x)
        return x
    
class FeedForwardDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.input_embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.input_embeddings)

    def __getitem__(self, index):
        sentence_embedding = self.input_embeddings[index]
        label=self.labels[index]

        return sentence_embedding, label

def create_dataloader_classification(x,y, batch_size):
  scotus_bert_embedded_dataset = FeedForwardDataset(x, y)
  return DataLoader(scotus_bert_embedded_dataset, batch_size=batch_size, shuffle=True)
   
def train_model(model,data_loader,optimizer,criterion,num_epochs = 70):
    lowest_error = float('inf')
    best_model_weights = None
    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if loss.item() < lowest_error:
                lowest_error = loss.item()
                best_model_weights = model.state_dict()
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.4f}')
    torch.save(best_model_weights, './model_wieghts/best_model_weights_feed_forward_classifier_test.pth')
    print('Training finished.')


def get_predictions_clf(model,embeddings_train_numpy):
    model.eval()
    predicted_labels=[]
    with torch.no_grad():
        for x in embeddings_train_numpy:
            outputs = model(x.view(1, -1))
            out, inds = torch.max(outputs,dim=1)
            pred=inds.cpu().numpy()
            predicted_labels.extend(pred)
    return predicted_labels


# def save_classification_report_clf(file_name,y_test,y_predicted):
#   pd.DataFrame(classification_report(y_test, y_predicted, target_names=["flag","Not_Flag"], output_dict=True)).T.to_csv(file_name, index= True)

# def save_heat_map_classification(file_name,y_test,y_predicted):
#   conf_matrix = confusion_matrix(y_test,y_predicted)
#   conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
#   plt.figure(figsize=(6, 4))
#   sns.heatmap(conf_matrix_norm, annot=True, cmap="Blues", fmt=".2f", annot_kws={"size": 12})

#   plt.title("Confusion Matrix Heatmap")
#   plt.xlabel("Predicted Labels")
#   plt.ylabel("True Labels")

#   plt.savefig(file_name)
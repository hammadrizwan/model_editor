import argparse
import Wikidata5m as wiki
import model_feedforward_classifier as ff
import model_siames_classifier as msc
import helper_functions as helper

from shared_imports import nn, optim, np, tqdm, metrics, os, torch
import sys

argParser = argparse.ArgumentParser()

argParser.add_argument("-d", "--datasets", nargs='+', help="name of datasets space seperated(need one minimum) wikidata5m, counterfact", choices={"wikidata5m", "counterfact"})
argParser.add_argument("-dsplit", "--dataset_splits", type=int, help="divide dataset into x number of ratios for testing")

args = argParser.parse_args()

from bert_encoder import get_embeddings
dataset_paths={
    "wikidata5m": "D:/PhD_Dalhousie/Model_Editing_Siames_Adapter/datasets/filtered_wikidata5m_transductive_train.jsonl",
    "counterfact": "path"
}
if __name__ == '__main__':
    print("datasets",vars(args)["datasets"])
    print("dataset_splits",vars(args)["dataset_splits"])
    if(vars(args)["datasets"]==None or vars(args)["dataset_splits"]==None):
        print("Missing arguments")
        sys.exit(0)
    

    for dataset in vars(args)["datasets"]:
            if(dataset=="counterfact"):
                loaded_dataset=[]
            else:
                train_x, paraphrase_x, local_neutral_x_train, local_neutral_x_test=wiki.create_dataset_from_file(dataset_paths[dataset],vars(args)["dataset_splits"])
                for index in tqdm(range(len(train_x))):
                    # For Feed Forward Model
                    #Get embeddings in dictionary style for use in both zsRE dataset and counterfact dataset. Test out Counterfact+
                    print(local_neutral_x_train[index].keys())
                    train_x_embeddings_dict=get_embeddings(train_x[index],flatten=False)
                    local_neutral_x_train_embeddings_dict=get_embeddings(local_neutral_x_train[index],flatten=False)
                    test_x_embeddings_dict=get_embeddings(paraphrase_x[index],flatten=False)
                    local_neutral_x_test_embeddings_dict=get_embeddings(local_neutral_x_test[index],flatten=False)
                    #Convert to list for use in feed forward classifier
                    train_x_embeddings_list=helper.embedding_dict_flatten(train_x_embeddings_dict)
                    local_neutral_x_train_embeddings_list=helper.embedding_dict_flatten(local_neutral_x_train_embeddings_dict)
                    test_x_embeddings_list=helper.embedding_dict_flatten(test_x_embeddings_dict)
                    local_neutral_x_test_embeddings_list=helper.embedding_dict_flatten(local_neutral_x_test_embeddings_dict)
                    
                    #Dataset construction
                    training_data=train_x_embeddings_list + local_neutral_x_train_embeddings_list
                    y_train=list(np.ones(len(train_x_embeddings_list),dtype=np.int64)) + list(np.zeros(len(local_neutral_x_train_embeddings_list),dtype=np.int64))
                    data_loader_train= ff.create_dataloader_classification(training_data,y_train,32)
                    
                    #model declaration
                    model = ff.FeedForwardClassifier(input_size=768,hidden_size1=384,hidden_size2=192, num_classes=2)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.0001,)

                    ff.train_model(model,data_loader_train,optimizer,criterion)

                    #test the model
                    testing_data=  test_x_embeddings_list + local_neutral_x_test_embeddings_list
                    y_test=list(np.ones(len(test_x_embeddings_list),dtype=np.int64)) + list(np.zeros(len(local_neutral_x_test_embeddings_list),dtype=np.int64))
                    predicted_labels=ff.get_predictions_clf(model,testing_data)
                    # Calculate metrics
                    accuracy = metrics.accuracy_score(y_test, predicted_labels)
                    precision = metrics.precision_score(y_test, predicted_labels)
                    recall = metrics.recall_score(y_test, predicted_labels)
                    f1 = metrics.f1_score(y_test, predicted_labels)
                    
                    formatted_metrics = f"Metrics for #samples in train {len(training_data)} and in test {len(testing_data)}, \nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}"

                    # Write the formatted metrics to a file
                    folder_path_feedforward="./feed_forward_classifier_results/"
                    helper.create_folder_if_not_exists(folder_path_feedforward)
                    file_path_ff_write = folder_path_feedforward+"metrics_feedforward.txt"
                    mode = 'a' if os.path.exists(file_path_ff_write) else 'w'#check if file exists
                    with open(file_path_ff_write,mode) as file:
                        file.write(formatted_metrics)

                    # For Siames Classifier
                    #dataset construction
                    dataset_train_siames= helper.create_siames_dataset(train_x_embeddings_dict,local_neutral_x_train_embeddings_dict)
                    # print(dataset_train_siames[0])
                    data_loader_train = msc.create_dataloader_siames(dataset_train_siames)

                    #model declaration
                    model_siames = msc.SiameseClassificationNetwork(input_size=768,hidden_size1=384,hidden_size2=192, num_classes=2)
                    criterion1 = nn.CrossEntropyLoss()
                    criterion2 = msc.CosineSimilarityLoss_SentenceTransformers()
                    optimizer = optim.Adam(model_siames.parameters(), lr=0.0001,)

                    msc.train_model_combined(model_siames,optimizer,data_loader_train,criterion1,criterion2)

                    dataset_test_siames= helper.create_siames_dataset(test_x_embeddings_dict,local_neutral_x_test_embeddings_dict)
                    # data_loader_test = msc.create_dataloader_siames(dataset_test_siames)
                    
                    y_test=[val[2] for val in dataset_test_siames]
                    predicted_labels= msc.get_predictions_combind(model_siames,dataset_test_siames)

                    # Calculate metrics
                    accuracy = metrics.accuracy_score(y_test, predicted_labels)
                    precision = metrics.precision_score(y_test, predicted_labels)
                    recall = metrics.recall_score(y_test, predicted_labels)
                    f1 = metrics.f1_score(y_test, predicted_labels)

                    formatted_metrics = f"Metrics for #samples in train {len(training_data)} and in test {len(testing_data)}, \nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}"

                    # Write the formatted metrics to a file
                    folder_path_feedforward="./siames_classifier_results/"
                    helper.create_folder_if_not_exists(folder_path_feedforward)
                    file_path_ff_write = folder_path_feedforward+"metrics_siames.txt"
                    mode = 'a' if os.path.exists(file_path_ff_write) else 'w'#check if file exists
                    with open(file_path_ff_write,mode) as file:
                        file.write(formatted_metrics)
                    
                    # For Feed Siames model
# predicted_labels=get_predictions(model,embeddings_train_numpy)
# save_classification_report(path_to_save+name_sub+".csv",Y_train_processed,predicted_labels)
# save_heat_map(path_to_save+name_sub+".png",Y_train_processed,predicted_labels)




        
    
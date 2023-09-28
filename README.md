# Repository for Model Editing
To install required packages
```
pip install requirements.txt
```

To run adapter run_controller(only option available rightnow)
```
python runner.py args
```
Required Args:
1. "-d", "--datasets", nargs='+', info="name of datasets space seperated(need one minimum) wikidata5m, counterfact", choices="wikidata5m", "counterfact"
2. "-dsplit", "--dataset_splits", type=int, info="divide dataset into x number of ratios for testing"
Optional Args:

## Task list:
- [ ] Add dataset loader from drive
- [ ] Add dataset constructor for COUNTERFACT dataset (Only one seen and the rest are for cosine sim)
- [ ] Add gridsearch or brute force search for layer and token index adapter run_controller testing 
- [ ] Use the FFN layer inside the adapter for testing single edit (freeze the rest of the layers)
- [ ] Run the gridsearch/bruteforce search on compute canada.
- [ ] Add T5 as encoder and new folders for encoder along with command line control for encoder. 


Paper Read list:
- [Inspecting and Editing Knowledge Representations in Language Models](https://arxiv.org/abs/2304.00740)
- [Transformer-Patcher: One Mistake worth One Neuron](https://arxiv.org/abs/2301.09785)
- [Detecting Edit Failures In Large Language Models: An Improved Specificity Benchmark](https://aclanthology.org/2023.findings-acl.733/)
- [Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/abs/2305.13172)
- [Evaluating the Ripple Effects of Knowledge Editing in Language Models](https://arxiv.org/abs/2307.12976)

# CaMeLS: Context-aware Meta-learned Loss Scaling

## What is this repo?

This repo includes a reference implementation of the CaMeLS metalearning approach to online adaptation [Meta-Learning Online Adaptation of Language Models](https://arxiv.org/abs/2305.15076).

The files in this repo are:

- `run.py`: main function to be called for both training of CaMeLS and evaluation of online adaptaiton. 
- `exp_datasets`: datasets classes surrounding document-question pairs. 
- `subroutines.py`: definition of subroutines for QA fine-tuning and top_k evaluation
- `weight_model.py` - define weight model class and subclasses


## Datasets
We conduct experiments using three datasets:

**StreamingQA:** to use, first download the csvs files at (https://drive.google.com/drive/u/1/folders/17qcGurJznFPta9Qo0z6YGtwh1fwRHm2f). https://github.com/deepmind/streamingqa). CSV files contain the subset of the StreamingQA dataset we use for our experiements. 

**SQuAD:** directly uses the huggingface dataset class and does not require additional file to be downloaded

**ArchivalQA:** the quesiton-article pairings can be referated by running the ArchivalQA_processing.ipynb notebook. Questions can be found here: https://drive.google.com/drive/u/0/folders/15JMtkJAqtZsKr_P-0jH4iFy2EOri4GbR and the NYT corpus here:  https://catalog.ldc.upenn.edu/LDC2008T19.

## QA model pretraining

The question-answering trained language models used for CaMeLS meta-training and online-adaptation evaluation are first pretrained on thier respective datasets using `train_qa.py` ie
`python train_qa.py dataset=squad model=gpt2-xl`

## CaMeLS metatraining

To train a CaMeLS weighting model using a base model saved as a checkpoint via `save_pretrained`:
`python run.py task=train dataset=archivalqa base_model=PATH_TO_QA_PROXY_MODEL.ckpt`

we can also load a base model from a state dict. 

`python run.py task=train dataset=archivalqa base_model=distilgpt2 base_model_state_dict=/qa_tuned/distilgpt2/state_dict.pt`

## online adaptation evaluation

Arguments: 

  **model**- the weighting model/strategy to use. either 'uniform', 'ssm', or 'fc', corresponding to unfiorm fine tuning, fine tuning only on words in salient spance, and CaMeLs
  
  **dataset** - the dataset used. By default, evaluations will run using the test split (specified on a per-dataset basis in conf/dataset). To run evaluations on the validation set, overwrite the correpsonding parameters. 
  
  **downsample_to** - for evaluation, we typically consider many streams of documents sampled from the test split. downsample_to = k corresponds to sampling k documents form the test set
  
  **seed** - the random seed used generate the test stream
  
  **qa_lt_final** - whether to additionally light tune the adapted qa model before evaluation. Used for the uniform + qa_tune baseline
  
  **eval_every_k** - by default -1. If not -1, we evaluate the base model after adapatation on every _eval_every_k_ articles. 
  

To evaluate using a saved CaMeLS weight model:
`python run.py task=eval dataset=streamingqa model=fc weight_model_path=path/to/CaMeLS_checkpoint.pt base_model=path/to/qa/model_ckpt lr=2.5e-05`

as in the train code, the base model can also be specificed using a state_dict:

`python run.py task=eval dataset=streamingqa model=fc weight_model_path=path/to/CaMeLS_checkpoint.pt base_model=gpt2-xl base_model_state_dict=/qa_tuned/gpt2-xl/state_dict.pt lr=2.5e-05`

To perform a learning rate sweep of the Salient Span Masking baseline: 
`python run.py -m task=eval dataset=streamingqa model=ssm base_model=path/to/qa/model_ckpt lr=.0001,.000025,.00000625,.0000015625 test_path=path/to/val_data.csv`

## A worked example
In this example we will train weight model on distilgpt2 fine tuned for QA on SQuAD. Then we will use the resulting weight model for online adaptation of gpt2-xl. 

### Step 1: Set up environment
First, create a virtualenv and install the dependencies. 

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

Additionlly, chance the 'CACHE' dir variable in 'util.py' if you wish.

### Step 2: Run QA pretrining for the base model to be used during CaMeLS pretraining and to be used for evaluation.
Pretrain 2 qa_models. 

    python train_qa.py dataset=squad model=distilgpt2
    python train_qa.py dataset=squad model=gpt2-xl

### Step 2: Train a CaMeLS weight model


### Step 3: Evaluate a trained weight model for Online Adapation




## Citing CaMeLS
If CaMeLS or this repository is useful in your own research, you can use the following BibTeX entry:

    @misc{hu2023metalearning,
      title={Meta-Learning Online Adaptation of Language Models}, 
      author={Nathan Hu and Eric Mitchell and Christopher D. Manning and Chelsea Finn},
      year={2023},
      eprint={2305.15076},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

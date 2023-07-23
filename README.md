
# CaMeLS: Context-aware Meta-learned Loss Scaling

## What is this repo?

This repo includes a reference implementation of the CaMeLS metalearning approach to online adaptation as in [Meta-Learning Online Adaptation of Language Models](https://arxiv.org/abs/2305.15076).

The files in this repo are:

- `run.py`: main function to be called for both training of CaMeLS and evaluation of online adaptaiton.
- `weight_model.py` - the core implementation of a CaMeLS weight model and baseline loss weighting strategies.
- `train_qa.py` - script to fine tune language models for question-answering on a given dataset.
- `util.py` - various helper functions.
- `exp_datasets`: datasets classes surrounding document-question pairs. 
- `subroutines.py`: definition of subroutines for QA tuning and top_k evaluation.


## Datasets
We conduct experiments using three datasets:

**[StreamingQA:](https://github.com/deepmind/streamingqa)** The CSV files containing the subset of the StreamingQA dataset we use for our experiements can be downloaded [here](https://drive.google.com/drive/folders/1Xod97TmnjmbGDiyOHfuEUZ14tSL3qA-X?usp=drive_link).

**SQuAD:** We directly uses the huggingface dataset class and do not require additional files to be manually downloaded.

**ArchivalQA:** The quesiton-article pairings can be recrated by running the `ArchivalQA_processing.ipynb` notebook using the [ArchivalQA questions](https://drive.google.com/drive/u/0/folders/15JMtkJAqtZsKr_P-0jH4iFy2EOri4GbR) and the [NYT corpus](https://catalog.ldc.upenn.edu/LDC2008T19).

**OpenWebText:** we additionally used a small sample of open webtext to enfore local updates during CaMeLS meta training. It can be downloaded [here](https://drive.google.com/drive/folders/1Xod97TmnjmbGDiyOHfuEUZ14tSL3qA-X?usp=drive_link).

## QA model pretraining

The language models trained for question-answering are used for CaMeLS meta-training and online-adaptation evaluation. Running `train_qa.py` will load a pretrained language model from Huggingface and fine tune that model for question-answering on a dataset. For example:
`python train_qa.py dataset=squad model=gpt2-xl`

Key `train_qa.py` arguments:
- `dataset`: the dataset to fine tune on. One of `archivalqa`, `squad`, `streamingqa`.  
- `model`: the pretrained language model to load fron huggingface. Valid values include: `gpt2-xl`, `EleutherAI/gpt-neo-1.3b`
, or `EleutherAI/gpt-j-6b`.


## CaMeLS meta-training and evaluation

Depending on the value of the `task` argument, `run.py` is used for both training of CaMeLS and evaluation of CaMeLS and baseline online adaptation methods. In both settings, the following common arguments are used:

Key `run.py` arguments:
- `model`: Weight model configuration corresponding to the loss weighting strategy. During evaluation, can be set to `uniform`, `ssm`, or `CaMeLS` corresponding to token less scaling methods. Should be set to `model=CaMeLS` for all meta-training runs.
- `dataset`: dataset used for the given task.
- `base_model`: used to specifiy the base language model. To be passed as the `pretrained_model_name_or_path` argument of `AutoModelForCausalLM.from_pretrained()`. During training, the base model updated during the inner step of optimization. During evaluation, the base model is updated on a stream of documents then evaluated question answering.
- `base_model_state_dict_path`: We can set the base model parameters by loading a state dictionary from the specified path. 


### CaMeLS meta-training

Running `run.py` with `task=train` lets us train CaMeLS weighting models. For example:

```python run.py task=train model=CaMeLS dataset=archivalqa base_model=distilgpt2 base_model_state_dict=/qa_tuned/distilgpt2/state_dict.pt```

Key `run.py task=train` arguments:

### Online Adaptation Evaluation

Key `run.py task=eval` arguments:
- `downsample_to`: for evaluation, we typically consider many streams of documents sampled from the test split. downsample_to = k corresponds to sampling k documents form the test set
- `seed`: the random seed used generate the test stream
- `qa_lt_final`:  whether to additionally light tune the adapted qa model before evaluation. Used for the uniform + qa_tune baseline
- `lr`: the learning rate used to update the base model on the stream of documents. We note that for all online adaptation methods, performance was very sensitive to learning rate. 

To evaluate using a saved CaMeLS weight model:
`python run.py task=eval dataset=streamingqa model=CaMeLS weight_model_path=path/to/CaMeLS_checkpoint.pt base_model=path/to/qa/model_ckpt lr=2.5e-05`

To perform a learning rate sweep of the Salient Span Masking baseline: 
`python run.py -m task=eval dataset=streamingqa model=ssm base_model=path/to/qa/model_ckpt lr=.0001,.000025,.00000625,.0000015625 test_path=path/to/val_data.csv`

## A worked example
In this example we will train weight model on distilgpt2 fine tuned for QA on SQuAD. Then we will use the resulting weight model for online adaptation of gpt2-xl. At each step, the outputs are generated in subdirectory begining with `outputs/TASK/DATASET/`

### Step 1: Set up environment
First, create a virtualenv and install the dependencies. 

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

Additionlly, change the `CACHE_DIR` dir variable in `util.py` if you wish.

For experiements using StreamingQA, download corresponding CSV files locally. Then change the value of `data_dir` in `conf/dataset/streamingqa` and `qa_conf/dataset/streamingqa` accordingly.

For metatraining using OpenWebText to enfore local updates, change the value of `web_text_csv` in `conf/task/train` to be the absolute path to the local OpenWebText csv.

### Step 2: Run QA pretrining for the base model to be used during CaMeLS pretraining and to be used for evaluation.
Next we train 2 language models for question answering. 

    python train_qa.py dataset=streamingqa model=distilgpt2
    python train_qa.py dataset=streamingqa model=gpt2-xl

The models with, the lowest validation NLL, and highest validation F1s score and and highest exact match are saved in the 'checkpoints' subdirectory. (For small models which fail to achieve significant F1 or EM scores, the model with the lowest NLL is used. Otherwise, our experiements chose the models with highest F1). 

### Step 2: Train a CaMeLS weight model

We now use the smaller qa-tuned distilgpt2 as the _base_model_ used to train a CaMeLS weighting model

    python run.py task=train model=CaMeLS dataset=streamingqa base_model=distilgpt2 base_model_state_dict={ABSOLUTE_PATH_TO_DISITLGPT2_STATE_DICT}

Sample model importance weights are generated periodicaly during training. Model checkpoints are saved every epoch and at each validation step with lower loss than all previous steps.

### Step 3: Evaluate a trained weight model for Online Adapation

Lastly, we will use our trained weighting model to adapt a gpt2-xl which we previously fined tuned for question answering. We set `dowmsample_to=1665` to adapt on a stream of 1665 sampled articles.

    python run.py task=eval model=CaMeLS weight_model_path={PATH_TO_WEIGHT_MODEL_CHECKPOINT} dataset=streamingqa base_model=gpt2-xl base_model_state_dict={ABSOLUTE_PATH_TO_GPT2XL_STATE_DICT} downsample_to=1665 lr=2.5e-5

To perform this same evaluation using a uniform fine tuning baseline

    python run.py task=eval model=uniform dataset=streamingqa base_model=gpt2-xl base_model_state_dict={ABSOLUTE_PATH_TO_GPT2XL_STATE_DICT} downsample_to=1665 lr=2.5e-5

And for only fine tuning on salient spans (first line only needs to be run the first time).

    python -m spacy download en_core_web_sm
    python run.py task=eval model=ssm dataset=streamingqa base_model=gpt2-xl base_model_state_dict={ABSOLUTE_PATH_TO_GPT2XL_STATE_DICT} downsample_to=1665 lr=2.5e-5

The model generations, per question F1 and EM values, and average F1 and EM scores are generated in the output csv file.
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

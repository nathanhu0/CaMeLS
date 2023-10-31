import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import GPT2TokenizerFast, AutoTokenizer
from datasets import load_dataset
from util import CACHE_DIR, shuffle_groups, return_k_unique
import copy

# Define your custom SequentialSampler
class RangeSampler(Sampler):
    def __init__(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index
        super().__init__(range(start_index, end_index))

    def __len__(self):
        return self.end_index - self.start_index
    
    def __iter__(self):
        return iter(range(self.start_index, self.end_index))

class TextAndQuestionDataset(Dataset):
    def __init__(self, max_text_len = 1024, max_question_len = 128, device = None, loc = False, qa_only = False, qa_for_generation=False, max_answer_len=24, tokenizer = 'gpt2', prompt_samples = -1, pad_qa_for_gen=True, include_eos = True):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir = CACHE_DIR)
        else:
            self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.max_text_len = max_text_len
        self.qa_for_generation = qa_for_generation
        self.qa_only = qa_only
        self.max_question_len = max_question_len 
        self.max_answer_len = max_answer_len
        self.loc = loc
        self.prompt_samples = prompt_samples
        self.pad_qa_for_gen = pad_qa_for_gen
        self.include_eos = include_eos
    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")
    
    def get_qa(self, idx):
        #return text corresponding to a question and answer pair at index idx
        #we expect answer to not have a space at the beginning
        raise NotImplementedError("Subclasses must implement get_qa")
    
    def get_text(self, idx):
        #return text corresponding to the passage with information at index idx
        raise NotImplementedError("Subclasses must implement get_text")
    
    def tok_qa_for_training(self, idx):
        question, answer = self.get_qa(idx)
        if self.include_eos:
            answer = answer + self.tokenizer.eos_token
        tok_answer = self.tokenizer(' '+answer, return_tensors="pt")
        #in order to create a mask of target_ids which only computes loss on the questions answer, we tokenize the question and answer separately then concatenate
        tok_question = self.tokenizer(question, return_tensors="pt")
        qa_ids = torch.cat([tok_question['input_ids'], (tok_answer['input_ids'])], 1)
        
        if qa_ids.shape[1] > self.max_question_len + self.max_answer_len:
            print(f'total question len {qa_ids.shape[1]} excedes max_question len f{self.max_question_len}. Truncating:')
            print(idx)
            num_to_truncate = qa_ids.shape[1] - self.max_question_len
            qa_ids = qa_ids[:, num_to_truncate:]
            tok_question['input_ids'] = tok_question['input_ids'][:, num_to_truncate:]
            tok_question['attention_mask'] = tok_question['attention_mask'][:, num_to_truncate:]
            
        n_pad = self.max_question_len - qa_ids.shape[1]
        qa_attention = torch.cat([tok_question['attention_mask'], (tok_answer['attention_mask'])], 1)
        qa_target_ids = qa_ids.clone()
        qa_target_ids[:, :tok_question['input_ids'].shape[1]] = -100
        qa_ids = torch.nn.functional.pad(qa_ids, (0, n_pad), value = self.tokenizer.pad_token_id)
        qa_attention = torch.nn.functional.pad(qa_attention, (0, n_pad), value = 0)
        qa_target_ids = torch.nn.functional.pad(qa_target_ids, (0, n_pad), value = -100)
        
        return qa_ids, qa_attention, qa_target_ids

    
    def tok_qa_for_generation(self, idx):
        question, answer = self.get_qa(idx)
        if self.include_eos:
            answer = answer + self.tokenizer.eos_token
        if self.pad_qa_for_gen:
            self.tokenizer.padding_side = 'left'
            tok_question = self.tokenizer(question, max_length=self.max_question_len-self.max_answer_len, padding='max_length', truncation=True, return_tensors="pt")
            self.tokenizer.padding_side = 'right'
        else:
            tok_question = self.tokenizer(question, return_tensors="pt")
        tok_answer = self.tokenizer(' '+answer, max_length=self.max_answer_len, padding='max_length', return_tensors="pt", truncation = True)
        return {'gen_q_ids': tok_question['input_ids'].squeeze().to(self.device), 
                'gen_q_attn_mask': tok_question['attention_mask'].squeeze().to(self.device),
                'question_text': question,
                'answer_text': answer,
                'answer_ids': tok_answer['input_ids'].squeeze().to(self.device), 
                'answer_mask': tok_answer['attention_mask'].squeeze().to(self.device)}
    
    def __getitem__(self, idx):
        qa_ids, qa_attention, qa_target_ids = self.tok_qa_for_training(idx)
        if self.loc:
            return {'loc_ids': qa_ids.squeeze().to(self.device), 
                    'loc_attention': qa_attention.squeeze().to(self.device), 
                    'loc_mask': torch.roll(qa_target_ids.squeeze().to(self.device) != -100, -1, 0)}
        if self.qa_only:
            return_dic =  {'idx': torch.tensor(idx).to(self.device),
                    'qa_ids': qa_ids.squeeze().to(self.device), 
                    'qa_attention': qa_attention.squeeze().to(self.device),
                    'qa_target_ids': qa_target_ids.squeeze().to(self.device)}
        else:
            text = self.tokenizer(self.get_text(idx), max_length=self.max_text_len ,padding='max_length', truncation=True, return_tensors="pt" )
            return_dic =  {'idx': torch.tensor(idx).to(self.device),
                    'text_ids': text['input_ids'].squeeze().to(self.device), 
                    'text_attention': text['attention_mask'].squeeze().to(self.device), 
                    'qa_ids': qa_ids.squeeze().to(self.device), 
                    'qa_attention': qa_attention.squeeze().to(self.device),
                    'qa_target_ids': qa_target_ids.squeeze().to(self.device)}
        if self.qa_for_generation:
            return_dic.update(self.tok_qa_for_generation(idx))
        
        return return_dic

#%%
class StreamingQADataset(TextAndQuestionDataset):
    def __init__(self, csv_path, downsample_to = -1, **kwargs):
        self.csv_path = csv_path
        self.data_frame = pd.read_csv(csv_path)
        if downsample_to != -1 and downsample_to < len(self.data_frame):
            print('downsampling from ', len(self.data_frame), ' to ', downsample_to, ' examples')
            self.data_frame = self.data_frame.sample(downsample_to)
        else:
            self.data_frame = self.data_frame.sample(frac=1)
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.data_frame)
    
    def get_qa(self, idx):
        row = self.data_frame.iloc[idx]
        answers = row['answers'].split("\\")
        answer = min(answers, key = len)
        question = row['question'].strip() 
        return question, answer
    
    def get_text(self, idx):
        return self.data_frame.iloc[idx]['text']
    
class SquadDataset(TextAndQuestionDataset):
    
    def __init__(self, split, start_idx = 0, end_idx = -1, shuffle_by='title', downsample_to=-1, downsample_by='context',**kwargs):
        squad_ds = load_dataset('squad', split=split,cache_dir=CACHE_DIR)
        if end_idx == -1:
            end_idx = len(squad_ds)
        squad_ds = squad_ds.select(list(range(start_idx,end_idx)))
        self.data_frame = pd.DataFrame(squad_ds)
        self.data_frame = shuffle_groups(self.data_frame, shuffle_by)
        if downsample_to > 0:
            self.data_frame = return_k_unique(self.data_frame, downsample_to, downsample_by)
        super().__init__(**kwargs)
        
    def __len__(self):
        return len(self.data_frame)
        
    def get_qa(self, idx):
        question = self.data_frame.iloc[idx]['question'].strip() 
        answer = min(self.data_frame.iloc[idx]['answers']['text'], key = len).strip()
        if answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        return question, answer
    
    def get_text(self, idx):
        return self.data_frame.iloc[idx]['context']
  
    def get_deduplicated_dataset(self):
        new_squad_ds = copy.deepcopy(self)
        new_squad_ds.data_frame = self.data_frame.drop_duplicates(subset=['context'])
        return new_squad_ds

class ArchivalQADataset(TextAndQuestionDataset):
    def __init__(self, csv_path, full_passage = False, shuffle_by='doc_id', downsample_to=-1,downsample_by='ans_paragraph', **kwargs):
        self.csv_path = csv_path
        self.full_passage = full_passage
        self.data_frame = pd.read_csv(csv_path)
        #we sort pre shuffle to make sure that for any given doc_id, the examples are in increasing order of para_num
        self.data_frame.sort_values('para_num', kind='stable', inplace=True)
        self.data_frame = shuffle_groups(self.data_frame, shuffle_by)
        if downsample_to > 0:
            self.data_frame = return_k_unique(self.data_frame, downsample_to, downsample_by)
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.data_frame)
    
    def get_qa(self, idx):
        row = self.data_frame.iloc[idx]
        answer = row['answer']
        if answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        question = row['question'].strip() 
        return question, answer
    
    def get_text(self, idx):
        if self.full_passage:
            return self.data_frame.iloc[idx]['ans_text']
        return self.data_frame.iloc[idx]['ans_paragraph']

    def get_deduplicated_dataset(self):
        new_arch_ds = copy.deepcopy(self)
        if self.full_passage:
            new_arch_ds.data_frame = self.data_frame.drop_duplicates(subset=['ans_text'])
        else:
            new_arch_ds.data_frame = self.data_frame.drop_duplicates(subset=['ans_paragraph'])
        return new_arch_ds

class WebTextDataset(Dataset):
    def __init__(self, csv_path, 
                 max_text_len = 1024, device_ = None, loc = False, tokenizer='gpt2'):
        self.csv_path = csv_path
        self.device = device_ if device_ is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_frame = pd.read_csv(csv_path)
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir = CACHE_DIR)
        else:
            self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.max_text_len = max_text_len
        self.loc = loc
    
    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        text = self.tokenizer(row['raw_text'], max_length=self.max_text_len ,padding='max_length', truncation=True, return_tensors="pt" )
        if self.loc:
            return {'loc_ids': text['input_ids'].squeeze().to(self.device), 
                    'loc_attention': text['attention_mask'].squeeze().to(self.device),
                    'loc_mask': text['attention_mask'].squeeze().to(self.device)}
        else:
            return {'input_ids': text['input_ids'].squeeze().to(self.device), 
                    'attention_mask': text['attention_mask'].squeeze().to(self.device)}
# %%


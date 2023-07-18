#%%
import os
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from exp_datasets import StreamingQADataset, SquadDataset, ArchivalQADataset, RACEDataset
from util import set_seed, decode_to_clean_text, exact_match, clean_up
import hydra
import wandb
import torch
from tqdm import tqdm
import numpy as np
from util import CACHE_DIR
from subroutines import qa_light_tune_early_stop
#%%
@hydra.main(config_path='qa_conf', config_name='config')
def run(args):
    wandb.init(config=args, project=args.wandb_project, name=args.wandb_run_name, entity="temporal-lms", settings=wandb.Settings(start_method='fork'))
    
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir = CACHE_DIR)
    if args.dataset == 'streamingqa':
        train_dataloader = DataLoader(StreamingQADataset(args.lt_train_path, tokenizer=tokenizer, qa_only=True, include_eos=args.qa_eos), batch_size=args.lt_batch_size, shuffle=True)
        val_dataloader = DataLoader(StreamingQADataset(args.lt_val_path, tokenizer=tokenizer, qa_only=True, qa_for_generation=True, include_eos=args.qa_eos), batch_size=args.lt_batch_size,
                            shuffle=False)
    elif args.dataset == 'squad':
        train_dataloader = DataLoader(SquadDataset(args.qa_lt_split, args.qa_lt_start_idx, args.qa_lt_end_idx, tokenizer=tokenizer, qa_only=True, include_eos=args.qa_eos), batch_size=args.lt_batch_size, shuffle=True)
        val_dataloader = DataLoader(SquadDataset(args.qa_lt_val_split, args.qa_lt_val_start_idx, args.qa_lt_val_end_idx, tokenizer=tokenizer,qa_only=True, include_eos=args.qa_eos), batch_size=args.lt_batch_size, shuffle=False)
    elif args.dataset == 'archivalqa':
        train_dataloader = DataLoader(ArchivalQADataset(args.lt_train_path, tokenizer=tokenizer,qa_only=True, include_eos=args.qa_eos), batch_size=args.lt_batch_size, shuffle=True)
        val_dataloader = DataLoader(ArchivalQADataset(args.lt_val_path, tokenizer=tokenizer,qa_only=True, qa_for_generation=True, include_eos=args.qa_eos), batch_size=args.lt_batch_size,
                            shuffle=False)
    elif args.dataset in ['race', 'race_mc']:
        train_dataloader = DataLoader(RACEDataset(args.lt_train_path, mcqa=args.mcqa, tokenizer=tokenizer,qa_only=True, include_eos=args.qa_eos), batch_size=args.lt_batch_size, shuffle=True)
        val_dataloader = DataLoader(RACEDataset(args.lt_val_path, mcqa=args.mcqa, tokenizer=tokenizer,qa_only=True, qa_for_generation=True, include_eos=args.qa_eos), batch_size=args.lt_batch_size, shuffle=False)
    else:
        print('dataset not supported for evaluation')
        raise NotImplementedError
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base_lm = AutoModelForCausalLM.from_pretrained(args.model, cache_dir = CACHE_DIR).to(device) 
    if args.grad_checkpointing:
        print("Enabling Gradient checkpointing")
        base_lm.transformer.gradient_checkpointing = True
    
    base_lm = base_lm
    
    if args.load_state_dict_path is not None:
        print('loading state dict')
        state_dict = torch.load(args.load_state_dict_path, map_location=device)
        base_lm.load_state_dict(state_dict)
    base_lm.to(device)
    
    
    qa_light_tune_early_stop(train_dataloader, val_dataloader, save_path=args.save_path, max_steps=args.train_steps, val_steps=args.val_steps, lr=args.lr, device=device, model = base_lm, include_question = False, grad_accumulation_steps = args.grad_acc_steps, resume=False, optimizer = args.optimizer, stopping_metric = 'nll', stop_k = args.stop_k, seed = 42, debug=False, early_stop = args.early_stop, wandb_log=True, grad_clip_thresh = args.grad_clip_thresh, save_best_metrics=['exact_match', 'nll', 'max_f1'])
    wandb.finish()


if __name__ == '__main__':
    run()
# %%

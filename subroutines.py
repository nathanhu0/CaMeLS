#%%
from transformers import Adafactor, GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer
from exp_datasets import StreamingQADataset
import torch
from tqdm import tqdm
from util import weighted_lm_loss, decode_to_clean_text, exact_match, CACHE_DIR, set_seed, debug_memory, f1_score
import numpy as np
from copy import deepcopy
import csv
import pickle
import math
import os
from collections import defaultdict
import wandb
import traceback

#%%
def gen_save(model, path):
    if getattr(model, "save_pretrained", None) is not None:
        model.save_pretrained(path)
    else:
        torch.save(model, path)

def get_optimizer(model, optimizer, lr):
    if optimizer == 'sgd':
        return torch.optim.SGD(list(model.parameters()), lr=lr)
    elif optimizer == 'adafactor':
        return Adafactor(list(model.parameters()), lr=lr, scale_parameter=False, relative_step=False)
    elif optimizer == 'adam':
        return torch.optim.Adam(list(model.parameters()), lr=lr)
    else:
        raise NameError('unknown optimizer type')

#qa_eval and validate are the same function, but qa_eval writes per line generations to a csv file

def qa_eval(dataloader, log_path, model = None, load_path = None, device = None, top_k = 1, diversity_penalty = 10., num_beam_groups = 4, num_beams = 12):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(load_path, cache_dir = CACHE_DIR).to(device) 
    model.eval()
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir = CACHE_DIR)
    total_cnt = 0
    em_correct = 0
    avg_f1s = []
    max_f1s = []
    with open(log_path, 'w', newline='') as writefile:  
        writer = csv.writer(writefile)
        for batch in tqdm(dataloader): 
            
            outs = model.generate(
                    input_ids = batch['gen_q_ids'],
                    attention_mask=batch["gen_q_attn_mask"],
                    use_cache=True,
                    max_length=batch['gen_q_ids'].shape[1]+16,
                    num_return_sequences=top_k,
                    num_beam_groups=num_beam_groups,
                    num_beams=num_beams,
                    diversity_penalty=diversity_penalty,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            dec = decode_to_clean_text(tokenizer, outs)
            texts = decode_to_clean_text(tokenizer, batch['gen_q_ids'])
            targets = decode_to_clean_text(tokenizer, batch['answer_ids'])
            for i in range(len(batch['gen_q_ids'])):
                total_cnt+=1
                answer = targets[i]
                
                predicted_answers = [dec[i*top_k + j][len(texts[i]):] for j in range(top_k)]
                em = 0
                f1s = []
                for pred_ans in predicted_answers:
                    if exact_match(pred_ans, answer, match_length = False):
                        em = 1  
                    f1s.append(f1_score(pred_ans, answer))
                em_correct += em
                writer.writerow([texts[i], answer, predicted_answers, f1s, em])
                avg_f1s.append(np.mean(f1s))
                max_f1s.append(np.max(f1s))
        writer.writerow(['EM', em_correct, em_correct / total_cnt])
        writer.writerow(['avg_f1', np.mean(avg_f1s), np.std(avg_f1s)])
        writer.writerow(['max_f1', np.mean(max_f1s), np.std(max_f1s)])
    print('done evaluating:', em_correct, em_correct / total_cnt)
    return em_correct, em_correct/total_cnt


def validate(model, tokenizer, dataloader, top_k = 1, greedy = False):
    em_correct = 0

    total_nll = 0
    total_tokens = 0
    avg_f1s = []
    max_f1s = []
    for batch in tqdm(dataloader): 
        #
        if 'gen_q_ids' in batch:
            if greedy:
                outs = model.generate(
                        input_ids = batch['gen_q_ids'],
                        attention_mask=batch["gen_q_attn_mask"],
                        use_cache=True,
                        max_length=batch['gen_q_ids'].shape[1]+16,
                        num_return_sequences=top_k,
                        num_beams=1,
                        do_sample=False,
                        early_stopping=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
            else:
                outs = model.generate(
                        input_ids = batch['gen_q_ids'],
                        attention_mask=batch["gen_q_attn_mask"],
                        use_cache=True,
                        max_length=batch['gen_q_ids'].shape[1]+16,
                        num_return_sequences=top_k,
                        num_beam_groups=4,
                        num_beams=12,
                        diversity_penalty=10.,
                        early_stopping=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
            dec = decode_to_clean_text(tokenizer, outs)
            texts = decode_to_clean_text(tokenizer, batch['gen_q_ids'])
            targets = decode_to_clean_text(tokenizer, batch['answer_ids'])
            for i in range(len(batch['gen_q_ids'])):
                answer = targets[i]
                
                predicted_answers = [dec[i*top_k + j][len(texts[i]):] for j in range(top_k)]
                em = 0
                f_1s = []
                for pred_ans in predicted_answers:
                    if exact_match(pred_ans, answer, match_length = False):
                        em = 1  
                    f_1s.append(f1_score(pred_ans, answer))
                em_correct += em
                avg_f1s.append(np.mean(f_1s))
                max_f1s.append(np.max(f_1s))

        with torch.no_grad():
            batch_nll = model(input_ids = batch['qa_ids'], attention_mask = batch['qa_attention'], labels = batch['qa_target_ids']).loss.item()
        n_tokens = (batch['qa_target_ids'] != -100).sum().item()
        total_nll += n_tokens*batch_nll
        total_tokens += n_tokens
    avg_nll = total_nll/total_tokens
    return {'exact_match': em_correct, 'nll': avg_nll, 'avg_f1': np.mean(avg_f1s), 'max_f1': np.mean(max_f1s)}


def qa_light_tune_early_stop(train_dataloader, val_dataloader, save_path, max_steps, val_steps, lr, device, model = None, load_path = None, grad_accumulation_steps = 1, resume=False, optimizer = 'adam', stopping_metric = 'nll', stop_k = 1, seed = 42, debug=False, early_stop = True, wandb_log=True, grad_clip_thresh = 1.0e9, save_best_metrics = [], name='', delete_checkpoints = False):
    
    save_path = save_path+ 'checkpoints'
    os.makedirs(save_path, exist_ok=True)
    if debug:
        debug_memory('start qa_light_tune_early_stop') 
        
    if model is None:
        print('loading model from path')
        model = AutoModelForCausalLM.from_pretrained(load_path, cache_dir = CACHE_DIR).to(device) 
    
    if debug:
        debug_memory('post optional model load') 
    
    model.train()
        
    cur_steps = 0
    tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir = CACHE_DIR)
    if isinstance(optimizer, str):
        optimizer = get_optimizer(model, optimizer, lr)
    light_tune_metrics = defaultdict(lambda: [])
    def early_stop(sm):
        if len(light_tune_metrics[sm]) > stop_k:
            if sm in ['nll' or 'ppl']:
                return min(light_tune_metrics[sm][:-stop_k]) < min(light_tune_metrics[sm][-stop_k:])
            else:
                #if our metric has not changed, we default to nll
                if max(light_tune_metrics[sm]) == min(light_tune_metrics[sm]):
                    print('early stop metric not changing, defaulting to nll')
                    return early_stop('nll')    
                return max(light_tune_metrics[sm][:-stop_k]) > max(light_tune_metrics[sm][-stop_k:])
        return False
    if resume and os.path.exists(os.path.join(save_path,'_val_metrics.pkl')):
        with open(os.path.join(save_path,'_val_metrics.pkl'), 'rb') as handle:
            light_tune_metrics = pickle.load(handle)
        cur_steps = light_tune_metrics['steps'][-1]
        del model
        model = AutoModelForCausalLM.from_pretrained(os.path.join(save_path, f'steps-{cur_steps}.ckpt'), cache_dir = CACHE_DIR).to(device) 
        
    if debug:
        debug_memory('post optional checkpoint load') 
    debug_post_loss = debug
    if stopping_metric not in save_best_metrics:
        save_best_metrics.append(stopping_metric)
    best_val_metrics = defaultdict(lambda: -1e9)
    if 'nll' in save_best_metrics:
        best_val_metrics['nll'] = 1e9
        
    while cur_steps < max_steps:
        set_seed(seed*max_steps + cur_steps)
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            if cur_steps % (val_steps*grad_accumulation_steps) == 0:
                if debug:
                    debug_memory('start of val')
                model.eval()
                val_metrics = validate(model, tokenizer, val_dataloader, greedy = True)
                model.train()
                light_tune_metrics['steps'].append(cur_steps)
                for k, v in val_metrics.items():
                    light_tune_metrics[k].append(v)
                
                for metric, value in val_metrics.items():
                    if metric in save_best_metrics:
                        if (metric in ['nll', 'ppl']) and (value < best_val_metrics[metric]):
                            best_val_metrics[metric] = value
                            torch.save(model.state_dict(), os.path.join(save_path, f'best_{metric}.pt'))
                        elif (metric not in ['nll', 'ppl']) and (value > best_val_metrics[metric]):
                            best_val_metrics[metric] = value
                            torch.save(model.state_dict(), os.path.join(save_path, f'best_{metric}.pt'))
                    
                if debug:
                    debug_memory(f'--qa_lt_val--  cur_steps: {cur_steps}, val_metrics: {val_metrics}')
                else:
                    print(f'--qa_lt_val--  cur_steps: {cur_steps}, val_metrics: {val_metrics}')
                if wandb_log: wandb.log({f'{name}qa_lt_val_{k}':v for k, v in val_metrics.items()})
                with open(os.path.join(save_path,'_val_metrics.pkl'), 'wb') as f:
                    pickle.dump(dict(light_tune_metrics), f)
                if early_stop and early_stop(stopping_metric):
                    print('EARLY STOPPING')
                    if debug:
                        print('early stopping')
                    cur_steps = max_steps
                    break
            
            loss = model(input_ids = batch[ 'qa_ids'], attention_mask = batch['qa_attention'], labels = batch[ 'qa_target_ids']).loss
            if wandb_log: wandb.log({'qa_lt_train_loss': loss})
            loss = loss/grad_accumulation_steps
            loss.backward() 
            if debug_post_loss:
                debug_memory('post loss backward')
                debug_post_loss = False
                
            if (cur_steps+1) % grad_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                if wandb_log: wandb.log({'qa_lt_grad_norm': grad_norm}) 
                optimizer.step()
                optimizer.zero_grad()
            
            cur_steps += 1
            
    with open(os.path.join(save_path,'_val_metrics.pkl'), 'wb') as f:
        pickle.dump(dict(light_tune_metrics), f)
    del optimizer
    best_state_dict = torch.load(os.path.join(save_path, f'best_{stopping_metric}.pt'))
    model.load_state_dict(best_state_dict)
    if delete_checkpoints:
        for f in os.listdir(save_path):
            if f.endswith('.pt'):
                os.remove(os.path.join(save_path, f))
    return model

def get_opt_hash(optimizer):
    try:
        return optimizer.state_dict()['state'][0]['exp_avg'].sum()
    except:
        return 0

def weighted_train(weight_model, dataloader, n_epochs, lr, base_lm, save_dir, grad_accumulation_steps = 1, resume=False, optimizer = 'adam', seed = 42, save_model = False, debug=False, wandb_log=True, grad_clip_thresh = 1.0e9, optimizer_state_dict = None, save_steps = -1):
    #train base_lm for n_epochs on dataloaders contents, using custom weights from weight_model
    completed_epochs = 0
    if debug:
        debug_memory('starting weighted_train')
    if resume:
        checkpoint_path = None
        while completed_epochs < n_epochs:
            if os.path.exists(os.path.join(save_dir,f'ft-{completed_epochs}.ckpt')):
                checkpoint_path = os.path.join(save_dir,f'ft-{completed_epochs}.ckpt')
                completed_epochs += 1
            else:
                break
        if checkpoint_path is not None:
            print(f'loading checkpoing {checkpoint_path}. Starting from epoch {completed_epochs}')
            device = base_lm.device
            base_lm.cpu()
            base_lm = AutoModelForCausalLM.from_pretrained(checkpoint_path, cache_dir = CACHE_DIR).to(device) 
    if debug:
        debug_memory('loaded checkpoint')
    if isinstance(optimizer, str):
        optimizer = get_optimizer(base_lm, optimizer, lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        print('loaded optimizer state dict:', get_opt_hash(optimizer))
    for i_epoch in range(completed_epochs, n_epochs):
        print('Starting training on epoch',i_epoch)
        optimizer.zero_grad()
        set_seed(seed*n_epochs + i_epoch)
        for i_step, batch in tqdm(enumerate(dataloader) , total=len(dataloader)):
            with torch.no_grad():
                weights = weight_model(batch['text_ids'], batch['text_attention'], idx = batch['idx'])
            targets = batch['text_ids'].clone()
            targets[batch['text_attention']!=1] = -100
            loss = weighted_lm_loss(base_lm, batch['text_ids'],targets,
                                        batch['text_attention'], weights)
            if wandb_log: wandb.log({'weighted_train_loss': loss}, commit=False)
            if wandb_log: wandb.log({'batch_hash': batch['text_ids'].sum().item()})
            wandb.log({'optimizer_state_hash': get_opt_hash(optimizer)}, commit=False)
            loss = loss/grad_accumulation_steps
            loss.backward()
            if (i_step+1) % grad_accumulation_steps == 0:
                grad = torch.nn.utils.clip_grad_norm_(base_lm.parameters(), grad_clip_thresh)
                if wandb_log: wandb.log({'weighted_train_grad_norm': grad})
                optimizer.step()
                optimizer.zero_grad()
            if save_steps > 0 and (i_step+1) % save_steps == 0:
                torch.save(base_lm.state_dict(), os.path.join(save_dir,f'ft-{i_epoch}-{i_step}.pt'))
                
        if len(dataloader)%grad_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        if (save_model or save_steps>0) and i_epoch == n_epochs - 1:
            if debug:
                debug_memory('saving checkpoint')
            torch.save(base_lm.state_dict(), os.path.join(save_dir,f'ft-{i_epoch}.pt'))
    if debug:
        debug_memory('finished weighted_train')
    return base_lm, optimizer

def qa_ppl_eval(dataloader, log_path, model = None, load_path = None, device = 'cuda'):
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(load_path, cache_dir = CACHE_DIR).to(device) 
    total_nll = 0
    total_tokens = 0
    set_seed(0)
    model.eval()
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch_nll = model(input_ids = batch['qa_ids'], attention_mask = batch['qa_attention'], 
                                                        labels = batch['qa_target_ids']).loss.item()
        n_tokens = (batch['qa_target_ids'] != -100).sum()
        total_nll += n_tokens*batch_nll
        total_tokens += n_tokens
    
    with open(log_path, 'w', newline='') as writefile:  
        writer = csv.writer(writefile)
        writer.writerow(['total_tokens', total_tokens, 'total_nll', total_nll])
        writer.writerow(['avg_nll', total_nll/total_tokens, 'avg ppl', math.exp(total_nll/total_tokens)])



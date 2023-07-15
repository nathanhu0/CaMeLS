#%%
import torch.nn as nn
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import string
import re
import warnings
import GPUtil
import getpass
import os

import spacy
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List
from collections import Counter

warnings.filterwarnings("ignore", message="UserWarning: Passing `max_length` to BeamSearchScorer is deprecated and has no effect. `max_length` should be passed directly to `beam_search(...)")


if os.path.exists('/scr-ssd'):
    CACHE_DIR = '/scr-ssd/' + getpass.getuser()
elif os.path.exists('/scr/scr-with-most-space'):
    CACHE_DIR = '/scr/scr-with-most-space/' + getpass.getuser()
else:
    CACHE_DIR = '/scr/' + getpass.getuser()
#CACHE_DIR=f'/iris/u/{getpass.getuser()}/cache/'
if not os.path.exists(CACHE_DIR):
    print('making cache_dir:', CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok = True)
 

def get_most_frequent_item(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None

#given a list of tokens, return a list of pos tags. to convert between tokenizers, we take the mode across characters in the underlying string
def get_pos_from_toks(toks, tokenizer, nlp = None):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    text_by_toks = [tokenizer.decode(t, clean_up_tokenization_spaces=False) for t in toks]
    text= ''.join(text_by_toks)
    pos = [None for _ in range(len(text))]
    doc = nlp(text)
    cur_idx = 0
    for token in doc:
        start_idx = text.find(token.text, cur_idx)
        for j in range(len(token)):
            pos[start_idx + j] = token.pos_    
        cur_idx = start_idx + len(token)
    tok_lens = [len(t) for t in text_by_toks]
    prefix_sum = np.cumsum(tok_lens, dtype=np.int32)
    return  [get_most_frequent_item(pos[prefix_len-tok_len:prefix_len]) for tok_len,prefix_len in zip(tok_lens, prefix_sum)] 

def get_nes_from_toks(toks, tokenizer, nlp = None, entities_to_ignore = []):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    text_by_toks = [tokenizer.decode(t, clean_up_tokenization_spaces=False) for t in toks]
    text= ''.join(text_by_toks)
    is_ne = [0 for _ in range(len(text))]
    doc = nlp(text)
    cur_idx = 0
    for token in doc:
        start_idx = text.find(token.text, cur_idx)
        for j in range(len(token)):
            if token.ent_type_ and token.ent_type_ not in entities_to_ignore:
                is_ne[start_idx + j] = 1
            else:
                is_ne[start_idx + j] = 0 
        cur_idx = start_idx + len(token)
    tok_lens = [len(t) for t in text_by_toks]
    prefix_sum = np.cumsum(tok_lens, dtype=np.int32)
    return  [max(is_ne[prefix_len-tok_len:prefix_len]) for tok_len,prefix_len in zip(tok_lens, prefix_sum)] 
 
def decode_to_clean_text(tokenizer, ids):
    gen_text = tokenizer.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    return list(map(str.strip, gen_text))

def clean_up(text):
    text =text.replace('<pad>', '')
    text = text.replace('</s>', '')
    text = text.replace(".", '')
    text = text.replace(',', '')
    text = text.replace("'", '')
    text = text.replace('"', '')
    return text 

def debug_memory(message = ''):
    print(message)
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    print(print(GPUtil.showUtilization()))
    
#lol this is funny, this code i took from ckl, removed the t5 stuff, and now its ecavtly the code in the squad codebase
def normalize_answer(s):
    #this is frankenstien code from ckl, some of the cleaning is not needed since we arent working w t5
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, ground_truth, match_length = False):
    norm_pred = normalize_answer(prediction)
    norm_truth = normalize_answer(ground_truth)
    if not match_length:
        norm_pred = norm_pred[:len(norm_truth)]
    return norm_pred == norm_truth

#taken from squad codebase
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
    

def weighted_lm_loss(model, input_ids, target_ids, attention_mask, weights):
    outputs = model(input_ids=input_ids,
                attention_mask=attention_mask,
                labels = target_ids
            )
    loss_fn = nn.CrossEntropyLoss(ignore_index = -100, reduction = 'none')
    batch_size = len(input_ids)
    reshaped_logits = outputs.logits[:, :-1, :].reshape(-1, outputs.logits.shape[-1])
    reshaped_labels = target_ids[:, 1:].reshape(-1)
    l = loss_fn(reshaped_logits, reshaped_labels)
    return (l.reshape(batch_size, -1)*weights[:, 1:]).mean()

#from ckl
def set_seed(seed):
    random.seed(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#from mend
def kl_loc_loss(pre, post, mask=None):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.view(-1, pre.shape[-1])
    post_ = post.view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

def generation_matches(model: AutoModelForCausalLM, batch, tokenizer, top_k = 1, frac = False, **kwargs):
    #returns the number of questions the model matches exactly
    em_correct = 0
    outs = model.generate(
                batch['gen_q_ids'],
                attention_mask=batch["gen_q_attn_mask"],
                use_cache=True,
                max_length=batch["gen_q_ids"].shape[1]+16,
                num_return_sequences=top_k, 
                num_beam_groups=kwargs.num_beam_groups if 'num_beam_groups' in kwargs else 4,
                num_beams=kwargs.num_beams if 'num_beams' in kwargs else 12,
                diversity_penalty=kwargs.diversity_penalty if 'diversity_penalty' in kwargs else 10.,
                early_stopping=kwargs.early_stopping if 'early_stopping' in kwargs else True,
                pad_token_id=tokenizer.eos_token_id
            )
    dec = decode_to_clean_text(tokenizer, outs)
    texts = decode_to_clean_text(tokenizer, batch['gen_q_ids'])
    targets = decode_to_clean_text(tokenizer, batch['gen_q_ids'])
    for i in range(len(batch['gen_q_ids'])):
        answer = targets[i]
        predicted_answers = [dec[i*top_k + j][len(texts[i]):] for j in range(top_k)]
        em = 0
        for pred_ans in predicted_answers:
            if exact_match(pred_ans, answer, match_length = False):
                em = 1  
        em_correct += em
    return em_correct/len(batch['gen_q_ids']) if frac else em_correct

def generate_fast(model, tok, prompts, n_gen_per_prompt = 1, top_k = 5, max_out_len = 50):
    """
    FROM ROME 
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        prompts: List[str],
        n_gen_per_prompt: int = 1,
        top_k: int = 5,
        max_out_len: int = 50,
    """

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    """
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]"""

    return txt




# %%
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List


def create_colored_text(words: List[str], data: List[float], font_path='DejaVuSansMono.ttf', pos_cmap=None, neg_cmap=None, max_intensity=.8) -> Image:
    # Create a colormap that maps your data range to colors
    
    if pos_cmap is None:
        pos_cmap = plt.cm.get_cmap('Reds')
    if neg_cmap is None:
        neg_cmap = plt.cm.get_cmap('Blues')
    max_mag = max(abs(np.min(data)), np.max(data))
    cmap = lambda x: pos_cmap(x*max_intensity/max_mag) if x > 0 else neg_cmap(-x*max_intensity/max_mag)
    max_width = 800
    line_height = 25
    
    # Set the font
    font = ImageFont.truetype(font_path, 16)
    # Find the maximum font size of all the words
    max_font_size = max([font.getbbox(word)[3] for word in words])
    # Initialize the x- and y-coordinates for drawing the words
    x = 0
    y = 0
    
    for word in words:
        word_width = font.getlength(word)
        if x + word_width > max_width:
            # Move to the next line
            x = 0
            y += line_height
        x += word_width
    
    final_height = y + line_height
    # Create a new image with a white background
    image = Image.new('RGB', (max_width, final_height), (255, 255, 255))
    # Get a drawing context
    draw = ImageDraw.Draw(image)
    x = 0
    y = 0
    # Iterate over the words in the text passage
    for i, word in enumerate(words):
        # Get the numeric value for the current word
        value = data[i]
        # Map the numeric value to a color from the colormap
        color = cmap(value)
        # Get the color in 8-bit RGB format
        rgb_color = tuple(int(c * 255) for c in color[:3])
        word_width = font.getlength(word)
        # Check if the word fits on the current line
        if x + word_width > max_width:
            # Move to the next line
            x = 0
            y += line_height
        # Draw the word with the mapped color and black foreground color
        draw.rectangle([(x, y), (x + word_width, y + max_font_size)], fill=rgb_color)
        draw.text((x, y), word, font=font, fill=(0, 0, 0))
        # Increment the x-coordinate for drawing the next word
        x += word_width
    image = image.crop((0, 0, max_width, y + line_height)).resize((max_width, y + line_height))
    return image
# %%

def shuffle_groups(df, group_col):
    """
    Shuffles the order of groups in a Pandas DataFrame without shuffling the order of items within each group.

    Parameters:
    - df: the input DataFrame
    - group_col: the name of the column containing the groups to be shuffled

    Returns:
    - a shuffled copy of the input DataFrame
    """
    # Get a list of unique groups
    groups = df[group_col].unique()

    # Shuffle the list of groups
    np.random.shuffle(groups)

    # Define a sorting key that sorts by the shuffled order of groups
    def sort_key(row):
        return np.argwhere(groups == row[group_col])[0][0]

    df['temp'] = df.apply(sort_key, axis=1)
    shuffled_df = df.sort_values('temp', kind='stable').drop('temp', axis=1).reset_index(drop=True)
    return shuffled_df

#given a pd dataframe, return a head of the dataframe such that column column has k unique values
def return_k_unique(df, k, column): 
    if k >= len(df[column].unique()):
        return df
    else:
        values_to_keep = df[column].unique()[:k]
        return df[df.apply(lambda x: x[column] in values_to_keep, axis=1)]



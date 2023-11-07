#%%
import torch.nn as nn
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import string
import re
import warnings
import getpass
import os

import spacy
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List
from collections import Counter
# %%

#change this to your cache dir
CACHE_DIR = ' /scr/scr-with-most-space/nathu'

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

def debug_memory(message = ''):
    print(message)
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    

def normalize_answer(s):
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



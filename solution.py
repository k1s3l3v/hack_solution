import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from transformers import BertModel, BertConfig, TFBertModel

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np

import re
from nltk.corpus import stopwords


def get_solution(
    summarize_type="all_comments", data_path="dataset.jsonl", path_to_save="result.json"
):
    russian_stopwords = stopwords.words("russian")

    CURSE_STRING = re.compile(
        "(?:ху[ий]|ху[ие]в[ыи]|бл[я]|г[оa]вн[оаы]|еб[аи]т[ьи]|п[ие]зд[аы]|сук[аи]|пид[оа]р|сучк[аи])"
    )

    def get_prepared(string):
        string = re.sub(r"#\w+", "", string).strip().lower()
        string = re.sub(CURSE_STRING, "", string)
        words_list = re.sub("[^а-яА-ЯëёЁ]+", " ", string).strip().split()
        return [word for word in words_list if word not in russian_stopwords]

    import json

    PATH = data_path
    data_df = pd.read_json(data_path, lines=True)
    data = []

    with open(PATH, "r") as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        data.append(json.loads(json_str))

    from collections import defaultdict

    posts_bert = defaultdict(list)
    for d in data:
        if "root_id" in d:
            text = get_prepared(d["text"])
            if text is not None and text != []:
                posts_bert[d["root_id"]].append(text)
    posts_bert = defaultdict(list)
    posts_hash = dict()
    for d in data:
        if "root_id" in d:
            text = get_prepared(d["text"])
            if text is not None and text != []:
                posts_bert[d["root_id"]].append((text, d["hash"]))
        else:
            posts_hash[d["id"]] = d["hash"]

    from scipy.spatial import distance

    bm = BertModel(BertConfig())
    for param in bm.base_model.parameters():
        param.requires_grad = False
    bm.eval()

    def get_embeddings(string):
        berted = bm(**tokenizer(string, return_tensors="pt"))
        return berted.pooler_output[0].detach().numpy()

    def get_summarize_for_post(post):
        post_comments = []
        for post__ in post:
            post_ = post__[0]
            comments = np.array([get_embeddings(x[0]) for x in post_])
            comments_result = np.mean(comments.T, axis=1)
            post_comments.append(comments_result)
        hashes = [x[1] for x in post]
        post_result = np.mean(np.array(post_comments).T, axis=1)
        print([distance.cosine(post_result, post_) for post_ in post_comments])
        xd = np.argmax([distance.cosine(post_result, post_) for post_ in post_comments])
        return post[xd], hashes[xd]

    res = []

    for post_hash, post in posts_bert.items():
        dct = {}
        summary, comment_hash = get_summarize_for_post(post)
        dct["summary"] = summary[0][0]
        dct["post_hash"] = post_hash
        dct["comment_hash"] = [comment_hash]
        res.append(dct)
    with open(path_to_save, "w") as f:
        f.write(json.dumps(res))


if __name__ == "__main__":
    get_solution()

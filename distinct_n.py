"""
File: distinct_n.py
Author: Lei Liu
Date: Dec 2, 2022
Description: Implementation of distinct n-gram.
"""
import collections


# Split each response in the list of responses into the following format: [word_1, word_2,..., word_n]
def split_sentences(responses, tokenizer):
    # Remove the special character used by the mT5 tokenizer
    token_to_del = "▁"
    responses = [[token for token in tokenizer.tokenize(response) if token != token_to_del] for response in responses]
    return responses


# Compute distinct n-grams scores (i.e. dist-1/2)
#
# Note: this function is implemented based on Yizhe Zhang's code at the following link:
# https://github.com/microsoft/DialoGPT/blob/master/dstc/metrics.py
def distinct_ngrams(preds):
    distinct_ngrams_scores = {}
    total_ngrams = [0.0, 0.0]
    unique_ngrams = [collections.defaultdict(int), collections.defaultdict(int)]

    for pred in preds:
        for n in range(1, 3):
            for idx in range(len(pred) - n + 1):
                ngram = ' '.join(pred[idx:idx + n])
                unique_ngrams[n - 1][ngram] = 1
                total_ngrams[n - 1] += 1

    for n in range(1, 3):
        distinct_ngrams_scores["distinct-%d" % n] = len(unique_ngrams[n - 1].keys()) / total_ngrams[n - 1] * 100

    return distinct_ngrams_scores

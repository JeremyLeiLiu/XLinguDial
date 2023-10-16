"""
File: download_mt5_base.py
Author: Lei Liu
Date: Nov 22, 2022

Description: Download the mT5-base model from HuggingFace Transformers to local directory.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load mT5 tokenizer and mT5-base model
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")

# Save both the tokenizer and model
tokenizer.save_pretrained("mt5-base")
model.save_pretrained("mt5-base")

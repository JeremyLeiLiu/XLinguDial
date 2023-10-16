#!/usr/bin/env python
# coding=utf-8
"""
File: run_dialogue.py
Author: Lei Liu

Description: Official repository for the Short Research Paper "Prompt Learning to Mitigate Catastrophic Forgetting in 
Cross-lingual Transfer for Open-domain Dialogue Generation" accepted for presentation at the 46th International ACM 
SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23).



<ACKNOWLEDGEMENTS>
A. Research Grants
This research is supported by the Natural Sciences and Engineering Research Council (NSERC) of Canada, the York
Research Chairs (YRC) program and an ORF-RE (Ontario Research Fund Research Excellence) award in BRAIN Alliance.
Particularly, Lei Liu, the first author of this paper, is supported by the SIGIR Student Travel Award and Academic
Excellence Fund for presenting this work at SIGIR '23.

B. Computing Resources
Computations were made on the supercomputer Béluga, managed by Calcul Québec and the Digital Research Alliance of Canada.



<CITATION>
Please cite our paper if you use any modules of our code.

@inproceedings{liu-etal-2023-prompt,
    author = {Liu, Lei and Huang, Jimmy Xiangji},
    title = {Prompt Learning to Mitigate Catastrophic Forgetting in Cross-Lingual Transfer for Open-Domain Dialogue Generation},
    year = {2023},
    isbn = {9781450394086},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3539618.3592043},
    doi = {10.1145/3539618.3592043},
    booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {2287–2292},
    numpages = {6},
    keywords = {catastrophic forgetting, few-shot cross-lingual transfer, dialogue generation, prompt learning, multitask learning},
    location = {Taipei, Taiwan},
    series = {SIGIR '23}
}
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
# modules for distinct n-gram
from distinct_n import split_sentences, distinct_ngrams

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    """
    ************************************************************
    The following four arguments are used for prompt learning.
    ************************************************************
    """

    # task-specific prefix/prompt for the source text, e.g. "context" in response generation task
    prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text."}
    )
    # task-specific prefix/prompt for the target text, e.g. "response" in response generation task
    target_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every target text."}
    )
    # The first two mask tokens used in pre-training mT5 models
    mask_token_1: Optional[str] = field(
        default=None, metadata={"help": "The first mask token that is used in pre-training mT5 models."}
    )
    mask_token_2: Optional[str] = field(
        default=None, metadata={"help": "The second mask token that is used in pre-training mT5 models."}
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("The training/validation file is needed.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # See all possible arguments by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: please provide CSV/JSON training and evaluation files.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    # In distributed training, the load_dataset function guarantees that only one local process
    # can concurrently download the dataset.
    datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined.")

    # Task-specific prompts for "context" and "response", respectively
    prefix = data_args.prefix if data_args.prefix is not None else ""
    target_prefix = data_args.target_prefix if data_args.target_prefix is not None else ""

    # Mask tokens for prompt-based learning approaches
    mask_token_1 = data_args.mask_token_1 if data_args.mask_token_1 is not None else ""
    mask_token_2 = data_args.mask_token_2 if data_args.mask_token_2 is not None else ""

    # Print out all four arguments used for prompt learning
    print("\n*********************************************************")
    print("Prompt for source text: " + prefix)
    print("Prompt for target text: " + target_prefix)
    print("Mask token 1: " + mask_token_1)
    print("Mask token 2: " + mask_token_2)
    print("***********************************************************\n")

    # Get the names of all the columns in the training, validation and test set, respectively
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Please specify the column names for both 'context' and 'response' in CSV/JSON files using the `text_column` and
    # `summary_column` arguments. Otherwise, by default, the first column and the second column would be regarded as
    # the 'context' and 'response' respectively.
    #
    # Get the column name for source text, i.e. 'context').
    if data_args.text_column is None:
        raise ValueError("Make sure that `text_column` corresponds to the column of `context` in your dataset.")
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Get the column name for target text, i.e. 'response'.
    if data_args.summary_column is None:
        raise ValueError("Make sure that `summary_column` corresponds to the column of `response` in your dataset.")
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        source_texts = examples[text_column]
        target_texts = examples[summary_column]

        # Format the source texts (i.e. 'contexts') depending on the values of all four arguments for prompt learning
        temp_source_texts = []
        if prefix and target_prefix:
            if mask_token_1 and mask_token_2:
                for source in source_texts:
                    temp_source_texts.append(prefix + " " + source + " " + target_prefix + " " + mask_token_1)
                source_texts = temp_source_texts
            else:
                for source in source_texts:
                    temp_source_texts.append(prefix + " " + source + " " + target_prefix)
                source_texts = temp_source_texts
        else:
            if mask_token_1 and mask_token_2:
                for source in source_texts:
                    temp_source_texts.append(source + " " + mask_token_1)
                source_texts = temp_source_texts
            else:
                for source in source_texts:
                    temp_source_texts.append(source)
                source_texts = temp_source_texts

        # Setup the tokenizer for source texts, i.e. 'contexts'
        model_inputs = tokenizer(source_texts, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Format the target texts (i.e. 'responses') depending on values of all four arguments for prompt learning
        temp_target_texts = []
        if prefix and target_prefix:
            if mask_token_1 and mask_token_2:
                for target in target_texts:
                    temp_target_texts.append(mask_token_1 + " " + target + " " + mask_token_2)
                target_texts = temp_target_texts
            else:
                for target in target_texts:
                    temp_target_texts.append(target)
                target_texts = temp_target_texts
        else:
            if mask_token_1 and mask_token_2:
                for target in target_texts:
                    temp_target_texts.append(mask_token_1 + " " + target + " " + mask_token_2)
                target_texts = temp_target_texts
            else:
                for target in target_texts:
                    temp_target_texts.append(target)
                target_texts = temp_target_texts

        # Setup the tokenizer for target texts, i.e. 'responses'
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(target_texts, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100
        # when we want to ignore padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize the training set
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    # Tokenize the validation set
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    # Tokenize the test set
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Automatic evaluation metrics including sacreBLEU and distinct n-grams
    #
    # The script for sacreBLEU is adapted from HuggingFace datasets at the following link:
    # https://github.com/huggingface/datasets/tree/main/metrics/sacrebleu
    metric_sacrebleu = load_metric('../sacrebleu/sacrebleu.py')

    # For each batch, compute both sacreBLEU and distinct n-grams
    def compute_metrics(eval_preds):
        all_results = {}
        preds, labels = eval_preds

        # Get the 'predictions', i.e. 'responses' generated by models.
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # When it comes to the prompt-based learning approaches,
        # make sure to REMOVE the additional MASK TOKENS in the 'predictions'.
        if mask_token_1 and mask_token_2:
            temp_preds = []
            for x in decoded_preds:
                if x.startswith(mask_token_1 + " ") and x.endswith(" " + mask_token_2):
                    temp_preds.append(x[len(mask_token_1) + 1:-len(mask_token_2) - 1])
                elif x.startswith(mask_token_1 + " "):
                    temp_preds.append(x[len(mask_token_1) + 1:])
                elif x.endswith(" " + mask_token_2):
                    temp_preds.append(x[:-len(mask_token_2) - 1])
                else:
                    temp_preds.append(x)
            decoded_preds = temp_preds
        # Remove white spaces in the 'predictions'
        decoded_preds = [pred.strip() for pred in decoded_preds]

        # The 'ground-truth responses', i.e. labels
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # When it comes to prompt-based learning approaches,
        # make sure to REMOVE the additional MASK TOKENS in the 'ground-truth responses' as well.
        if mask_token_1 and mask_token_2:
            temp_labels = []
            for x in decoded_labels:
                if x.startswith(mask_token_1 + " ") and x.endswith(" " + mask_token_2):
                    temp_labels.append(x[len(mask_token_1) + 1:-len(mask_token_2) - 1])
                elif x.startswith(mask_token_1 + " "):
                    temp_labels.append(x[len(mask_token_1) + 1:])
                elif x.endswith(" " + mask_token_2):
                    temp_labels.append(x[:-len(mask_token_2) - 1])
                else:
                    temp_labels.append(x)
            decoded_labels = temp_labels
        # Remove white spaces in the 'ground-truth responses'
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute distinct n-grams scores (i.e. dist-1/2)
        #
        # Tokenize all the 'predictions' in order to compute the distinct n-grams scores.
        decoded_preds_split = split_sentences(decoded_preds, tokenizer)
        result_distinct = distinct_ngrams(decoded_preds_split)
        all_results = {**all_results, **result_distinct}

        # Compute sacreBLEU scores (i.e. sacreBLEU-score/1/2/bp).
        # IMPORTANT: for sacreBLEU, do NOT tokenize both the 'ground-truth responses' and 'predictions' here.
        result_sacrebleu = metric_sacrebleu.compute(predictions=[pred.strip() for pred in decoded_preds],
                                                    references=[[label.strip()] for label in decoded_labels])
        # Extract the metric results for sacreBLEU
        all_results['sacrebleu_score'] = result_sacrebleu['score']
        for i in range(2):
            all_results['sacrebleu_%d' % (i + 1)] = result_sacrebleu['precisions'][i]
        all_results['sacrebleu_bp'] = result_sacrebleu['bp']

        return all_results

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                # For prompt-based learning approaches, make sure to REMOVE the additional MASK TOKENS.
                if mask_token_1 and mask_token_2:
                    temp_predictions = []
                    for prediction in predictions:
                        if prediction.startswith(mask_token_1 + " ") and prediction.endswith(" " + mask_token_2):
                            temp_predictions.append(prediction[len(mask_token_1) + 1:-len(mask_token_2) - 1])
                        elif prediction.startswith(mask_token_1 + " "):
                            temp_predictions.append(prediction[len(mask_token_1) + 1:])
                        elif prediction.endswith(" " + mask_token_2):
                            temp_predictions.append(prediction[:-len(mask_token_2) - 1])
                        else:
                            temp_predictions.append(prediction)
                    predictions = temp_predictions

                # Remove white spaces in the 'predictions'.
                predictions = [prediction.strip() for prediction in predictions]

                # Output the 'predictions' into a *.txt file
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        trainer.push_to_hub(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

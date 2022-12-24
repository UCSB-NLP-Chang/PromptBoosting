import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import logging
import time
import os
import numpy as np
import torch

import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification, AdamW, 
    BertTokenizer, BertForSequenceClassification,
    EvalPrediction,
    )
from transformers import HfArgumentParser, TrainingArguments, set_seed, Trainer

from src.finetuning.dataloader import (LocalAGDataset, LocalMRDataset, LocalNLIDataset, LocalSSTDataset, 
                                        LocalTrecDataset, LocalIMDbDataset,
                                        remove_unused_columns,
                                        )
# from src.finetuning.trainer import Trainer
from src.finetuning.processor import compute_metrics_mapping
from src.data_util import get_class_num, get_task_type
from src.utils import ROOT_DIR, write_performance, MODEL_CACHE_DIR

import wandb

logger = logging.getLogger(__name__)


@dataclass
class MyArguments:
    # use_wandb: bool = field(default = False)
    dataset: str = field(default = 'sst')
    model: str = field(default = 'roberta')
    fewshot: bool = field(default = False)
    low: bool = field(default = False)
    low_mode: str = field(default = 'low-resource-16valid')
    fewshot_k: int = field(default = 16)
    fewshot_seed: int = field(default = 100)

if __name__ == '__main__':
    parser = HfArgumentParser((MyArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    dataset = args.dataset
    model_type = args.model
    fewshot = args.fewshot
    fewshot_k = args.fewshot_k
    fewshot_seed = args.fewshot_seed
    low = args.low
    low_mode = args.low_mode

    sentence_pair = get_task_type(dataset)
    num_labels = get_class_num(dataset)
    assert not (fewshot and low), "fewshot and low resource can not be true!"

    if training_args.report_to == ['wandb']:
        use_wandb = True
    else:
        use_wandb = False
    if use_wandb:
        wandb_name = f"{model_type}-{dataset}"
        if fewshot:
            wandb_name += f"-{fewshot_k}shot-seed{fewshot_seed}"
        elif low:
            wandb_name += f"-low{fewshot_k}-seed{fewshot_seed}"
        os.environ["WANDB_PROJECT"] = f'finetuning-{dataset}'
        training_args.run_name = wandb_name
        training_args.report_to = 'wandb',   ## parameters:  run_name;  to set project name, use os.environ["WANDB_PROJECT"] = "huggingface"
        os.environ["WANDB_PROJECT"] = f"finetuning-{dataset}"
    # else:
        # os.environ["WANDB_DISABLED"] = "true"
        # training_args.report_to = []

    if model_type == 'roberta':
        cache_dir = os.path.join(MODEL_CACHE_DIR, 'roberta_model/roberta-large/')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large', cache_dir = cache_dir)
        model = RobertaForSequenceClassification.from_pretrained('roberta-large', cache_dir = cache_dir, num_labels = num_labels)
        model_fn = RobertaForSequenceClassification
    elif model_type == 'bert':
        cache_dir = os.path.join(MODEL_CACHE_DIR, 'bert_model/bert-large-uncased/')
        model = BertForSequenceClassification.from_pretrained('bert-large-uncased', cache_dir = cache_dir, num_labels = num_labels)
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', cache_dir = cache_dir)
        model_fn = BertForSequenceClassification
    else:
        raise NotImplementedError

    if dataset == 'sst':
        all_dataset = LocalSSTDataset(tokenizer = tokenizer, fewshot = fewshot, k = fewshot_k, rand_seed = fewshot_seed, low_resource = low, low_resource_mode = low_mode)
    elif dataset in ['rte', 'qnli', 'mnli', 'snli']:
        all_dataset = LocalNLIDataset(dataset_name = dataset, tokenizer = tokenizer, fewshot = fewshot, k = fewshot_k, rand_seed = fewshot_seed, low_resource = low, low_resource_mode = low_mode)
    elif dataset == 'trec':
        all_dataset = LocalTrecDataset(tokenizer = tokenizer, fewshot = fewshot, k = fewshot_k, rand_seed = fewshot_seed, low_resource = low, low_resource_mode = low_mode)
    elif dataset == 'mr':
        all_dataset = LocalMRDataset(tokenizer = tokenizer, fewshot = fewshot, k = fewshot_k, rand_seed = fewshot_seed, low_resource = low, low_resource_mode = low_mode)
    elif dataset == 'agnews':
        all_dataset = LocalAGDataset(tokenizer = tokenizer, fewshot = fewshot, k = fewshot_k, rand_seed = fewshot_seed, low_resource = low, low_resource_mode = low_mode)
    elif dataset == 'imdb':
        all_dataset = LocalIMDbDataset(tokenizer = tokenizer, fewshot = fewshot, k = fewshot_k, rand_seed = fewshot_seed, low_resource = low, low_resource_mode = low_mode)

    train_dataset = all_dataset.train_dataset
    valid_dataset = all_dataset.valid_dataset
    test_dataset = all_dataset.test_dataset

    train_dataset = remove_unused_columns(model, train_dataset)
    valid_dataset = remove_unused_columns(model, valid_dataset)
    test_dataset = remove_unused_columns(model, test_dataset)

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            logits = predictions            
            preds = np.argmax(logits, axis=1)
            label_ids = p.label_ids
            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn
    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        compute_metrics = build_compute_metrics_fn(args.dataset),
        data_collator = all_dataset.data_collator,
    )

    if training_args.do_train:
        trainer.train()
        # Reload the best checkpoint (for eval)
        # model = model_fn.from_pretrained(training_args.output_dir)
        # model = model.to(training_args.device)
        # trainer.model = model
        # model.tokenizer = tokenizer

    # Evaluation
    final_result = {
    }

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [test_dataset]

        for eval_dataset in eval_datasets:
            output = trainer.evaluate(eval_dataset=eval_dataset)
            # eval_result = output.metrics 
            eval_results.update(output)

    print(eval_results)

    if (fewshot or low):
        # wandb_name += f"-{fewshot_k}shot-seed{fewshot_seed}"
        fewshot_seed = fewshot_seed
        fewshot_k = fewshot_k
    else:
        fewshot_seed = 0
        fewshot_k = 'full'
    wandb_id = wandb.run.id
    # seed = 
    data_dict = {'test_acc': eval_results['eval_acc'], 'wandb_id': wandb_id, 'fewshot_seed': fewshot_seed, 'fewshot_k': fewshot_k}
    required_keys = ['wandb_id', 'fewshot_k', 'fewshot_seed','test_acc']
    filename = f'finetuning-{model_type}-{dataset}.csv'

    file_dir = os.path.join(ROOT_DIR, 'stat_data_file')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_addr = os.path.join(file_dir, filename)
    write_performance(file_addr, data_dict, required_keys)



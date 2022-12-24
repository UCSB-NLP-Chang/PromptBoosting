import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np 
import torch

import tqdm
import time

from src.multicls_trainer import PromptBoostingTrainer
from src.ptuning import RoBERTaVTuningClassification
from src.saver import TestPredictionSaver
from src.template import TemplateManager
from src.utils import ROOT_DIR, MODEL_CACHE_DIR, BATCH_SIZE, create_logger
from src.data_util import get_class_num, load_dataset, get_task_type, get_template_list

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = 'sst')
parser.add_argument("--model", type = str, default = 'roberta')
parser.add_argument("--pred_cache_dir", type = str, default = '')
parser.add_argument("--use_part_templates", action = 'store_true')
parser.add_argument("--start_idx", type = int, default = 0)
parser.add_argument("--end_idx", type = int, default = 10)

parser.add_argument("--sort_dataset", action = 'store_true')

parser.add_argument("--fewshot", action = 'store_true')
parser.add_argument("--fewshot_k", type = int, default = 0)
parser.add_argument("--fewshot_seed", type = int, default = 100, choices = [100, 13, 21, 42, 87])
parser.add_argument("--filter_templates", action = 'store_true')

args = parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    device = torch.device('cuda')
    dataset = args.dataset
    sentence_pair = get_task_type(dataset)
    num_classes = get_class_num(dataset)
    model = args.model

    pred_cache_dir = args.pred_cache_dir
    sort_dataset = args.sort_dataset
    fewshot = args.fewshot
    fewshot_k = args.fewshot_k
    fewshot_seed = args.fewshot_seed
    filter_templates = args.filter_templates

    train_dataset, valid_dataset, test_dataset = load_dataset(dataset_name = dataset, sort_dataset = sort_dataset, fewshot = fewshot, k = fewshot_k, rand_seed = fewshot_seed)

    num_test = len(test_dataset[0])
    test_labels = torch.LongTensor(test_dataset[1]).to(device)


    vtuning_model = RoBERTaVTuningClassification(model_type = 'roberta-large', cache_dir = os.path.join(MODEL_CACHE_DIR, 'roberta_model/roberta-large/'),
                                            device = device, verbalizer_dict = None, sentence_pair = sentence_pair)
    if filter_templates:
        template_dir_list = get_template_list(dataset, True, model = model, filter_num = 10)
    else:
        template_dir_list = get_template_list(dataset)
    template_manager = TemplateManager(template_dir_list = template_dir_list, output_token = vtuning_model.tokenizer.mask_token, max_template_num = 0,
                                        use_part_templates = True, start_idx = args.start_idx, end_idx = args.end_idx)

    dir_list = "\n\t".join(template_manager.template_dir_list)
    print(f"using templates from: {dir_list}",)

    trainer = PromptBoostingTrainer(adaboost_lr = 1.0, num_classes = num_classes, adaboost_maximum_epoch = 100)

    save_dir = os.path.join(ROOT_DIR, f'cached_test_preds/{dataset}/')
    prediction_saver = TestPredictionSaver(save_dir = os.path.join(ROOT_DIR, f'cached_test_preds/{dataset}/'), model_name = model)
    
    word2idx = vtuning_model.tokenizer.get_vocab()
    all_templates = template_manager.get_all_template()
    for template in all_templates:
        template.visualize()
        test_probs = trainer.pre_compute_logits(vtuning_model, template, test_dataset)
        prediction_saver.save_preds(template, test_probs)

    end_time = time.time()
    print(f"time used: {end_time - start_time}")


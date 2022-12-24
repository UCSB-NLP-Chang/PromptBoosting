import numpy as np 
import torch

import tqdm
import time
import os

from src.multicls_trainer import PromptBoostingTrainer
from src.ptuning import BaseModel, OPTVTuningClassification, RoBERTaVTuningClassification
from src.saver import PredictionSaver, TestPredictionSaver
from src.template import SentenceTemplate, TemplateManager
from src.utils import ROOT_DIR, BATCH_SIZE, create_logger, MODEL_CACHE_DIR
from src.data_util import get_class_num, get_weak_cls_num, load_dataset, get_task_type, get_template_list

import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--adaboost_lr", type = float, default = 1.0)
parser.add_argument("--dataset", type = str, default = 'sst')
parser.add_argument("--model", type = str, default = 'roberta')
parser.add_argument("--template_name", type = str, default = 't5_template3')
parser.add_argument("--stop_criterior", type = str, default = 'best')
parser.add_argument("--label_set_size", type = int, default = 5)
parser.add_argument("--max_template_num", type = int, default = 0)

parser.add_argument("--pred_cache_dir", type = str, default = '')
parser.add_argument("--use_logits", action = 'store_true')
parser.add_argument("--use_wandb", action = 'store_true')
parser.add_argument("--change_template", action = 'store_true')
parser.add_argument("--manual", action = 'store_true')

parser.add_argument("--use_part_templates", action = 'store_true')
parser.add_argument("--start_idx", type = int, default = 0)
parser.add_argument("--end_idx", type = int, default = 10)

parser.add_argument("--second_best", action = 'store_true')
parser.add_argument("--sort_dataset", action = 'store_true')

parser.add_argument("--fewshot", action = 'store_true')
parser.add_argument("--fewshot_k", type = int, default = 0)
parser.add_argument("--fewshot_seed", type = int, default = 100, choices = [100, 13, 21, 42, 87])
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cuda')
    adaboost_lr = args.adaboost_lr
    template_name = args.template_name
    dataset = args.dataset
    sentence_pair = get_task_type(dataset)
    num_classes = get_class_num(dataset)
    adaboost_weak_cls = get_weak_cls_num(dataset)
    model = args.model

    pred_cache_dir = args.pred_cache_dir
    sort_dataset = args.sort_dataset
    stop_criterior = args.stop_criterior
    use_logits = args.use_logits
    use_wandb = args.use_wandb
    label_set_size = args.label_set_size
    max_template_num = args.max_template_num

    adaboost_maximum_epoch = 20000

    fewshot = args.fewshot
    fewshot_k = args.fewshot_k
    fewshot_seed = args.fewshot_seed

    filter_templates = args.filter_templates

    suffix = ""
    if args.use_part_templates:
        suffix = f"range({args.start_idx}-{args.end_idx})"
    if filter_templates:
        suffix += f"filtered"

    logger, log_dir = create_logger(logger_name='ensemble_novalid',filename = f'{model}-{dataset}-{suffix}-novalid')
    wandb_name = f"{model}-{dataset}-{suffix}-novalid"

    if use_wandb:
        if fewshot:
            wandb_name += f"-{fewshot_k}shot-seed{fewshot_seed}"
        wandb.init(project = f'vtuning-{dataset}', name = f'{wandb_name}')


    train_dataset, _, test_dataset = load_dataset(dataset_name = dataset, sort_dataset = sort_dataset, fewshot = fewshot, k = fewshot_k, rand_seed = fewshot_seed,
                                                 use_valid_for_train = True)
    # label_token_set_path = ROOT_DIR + 'label_maps/sst_roberta_template1.json'

    num_training = len(train_dataset[0])
    num_test = len(test_dataset[0])
    train_labels = torch.LongTensor(train_dataset[1]).to(device)
    test_labels = torch.LongTensor(test_dataset[1]).to(device)

    weight_tensor = torch.ones(num_training, dtype = torch.float32).to(device) / num_training

    if model == 'roberta':
        vtuning_model = RoBERTaVTuningClassification(model_type = 'roberta-large', cache_dir = os.path.join(MODEL_CACHE_DIR, 'roberta_model/roberta-large/'),
                                                device = device, verbalizer_dict = None, sentence_pair = sentence_pair)
    elif model == 'opt1.3b':
        vtuning_model = OPTVTuningClassification(model_type = 'facebook/opt-1.3b', cache_dir = os.path.join(MODEL_CACHE_DIR, 'opt_model/opt-1.3b/'),
                                                device = device, verbalizer_dict = None, sentence_pair = sentence_pair)

    template_dir_list = get_template_list(dataset)
    template_manager = TemplateManager(template_dir_list = template_dir_list, output_token = vtuning_model.tokenizer.mask_token, max_template_num = max_template_num,
                                        use_part_templates = args.use_part_templates, start_idx = args.start_idx, end_idx = args.end_idx)

    dir_list = "\n\t".join(template_manager.template_dir_list)
    print(f"using templates from: {dir_list}",)

    trainer = PromptBoostingTrainer(adaboost_lr = adaboost_lr, num_classes = num_classes, adaboost_maximum_epoch = adaboost_maximum_epoch)

    if pred_cache_dir != '':
        prediction_saver = PredictionSaver(save_dir = os.path.join(ROOT_DIR, pred_cache_dir, 'novalid/'), model_name = model,
                                            fewshot = fewshot, fewshot_k = fewshot_k, fewshot_seed = fewshot_seed,
                                            )
    else:
        prediction_saver = PredictionSaver(model_name = model,
                                            fewshot = fewshot, fewshot_k = fewshot_k, fewshot_seed = fewshot_seed,        
                                            )
    test_pred_saver = TestPredictionSaver(save_dir = os.path(ROOT_DIR, f'cached_test_preds/{dataset}/'), model_name = model)
    train_probs, valid_probs = [],[]

    word2idx = vtuning_model.tokenizer.get_vocab()
    for model_id in tqdm.tqdm(range(adaboost_weak_cls)):
        if args.change_template:
            del train_probs
            del valid_probs
            template = template_manager.change_template()
            template.visualize()
            cached_preds, flag = prediction_saver.load_preds(template)
            if not flag:
                train_probs = trainer.pre_compute_logits(vtuning_model, template, train_dataset,)
                valid_probs = []
                prediction_saver.save_preds(template, train_probs, valid_probs)
            else:
                train_probs, valid_probs = cached_preds

        trainer.record_dataset_weights(weight_tensor)

        verbalizer, train_error,train_acc, wrong_flags,train_preds= trainer.train(train_dataset, vtuning_model, train_probs, train_labels,
                                                                                weight_tensor = weight_tensor,label_set_size = label_set_size,
                                                                                )
        print(verbalizer)
        if train_error < 1 - (1 / (num_classes)):
            print(f"\tmodel {model_id + 1} finished")
            print(f"\ttrain error {train_error}, train_acc {train_acc}")

            logger.info(f"\tmodel {model_id + 1} finished")
            logger.info(f"\ttrain error {train_error}, train_acc {train_acc}")
            succ_flag = True
        else:
            print(f"error {train_error}; train_acc {train_acc}\n Ensemble is worse than random, ensemble can not be fit.")
            logger.info(f"error {train_error}; train_acc {train_acc}\n Ensemble is worse than random, ensemble can not be fit.")
            continue

        alpha, weight_tensor = trainer.adaboost_step(train_error, wrong_flags, weight_tensor)
        print(f"\talpha {alpha}")
        logger.info(f"\talpha {alpha}")

        trainer.save_prediction(train_preds, split = 'train')

        train_ensemble_acc = trainer.ensemble_result(train_labels, split = 'train')

        trainer.best_epoch = len(trainer.model_weight_tensor)

        trainer.save_weak_learner(verbalizer, template.template_name)

        tolog = {
            'train_error': train_error,
            'alpha': alpha,
            'train_acc': train_acc,
            'ensemble_train_acc': train_ensemble_acc,
        }
        if use_wandb:
            wandb.log(tolog)


    print(f"finish training with {len(trainer.model_weight_tensor)} weak classifier")
    print(f"best ensemble classfier: 0 - {trainer.best_epoch}")
    logger.info(f"finish training with {len(trainer.model_weight_tensor)} weak classifier")
    logger.info(f"best ensemble classfier: 0 - {trainer.best_epoch}")
    
    all_template_used = template_manager.get_all_template()
    test_ensemble_acc = trainer.final_eval(test_dataset, vtuning_model, all_template_used, test_pred_saver)

    print(f"best test acc {test_ensemble_acc}")
    logger.info(f"best test acc {test_ensemble_acc}")

    to_log = {"best_test":test_ensemble_acc}
    wandb.log(to_log)

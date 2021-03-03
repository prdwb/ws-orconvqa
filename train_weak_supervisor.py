#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[2]:


from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob
import timeit
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import pytrec_eval

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertTokenizer, BertForQuestionAnswering)

from transformers import AdamW, get_linear_schedule_with_warmup

from utils import (WeakSupervisorDataset,
                   RawResult, write_predictions, write_weak_supervisor_predictions,
                   RawResultExtended, write_predictions_extended,
                   get_retrieval_metrics, weak_supervisor_eval)


# In[3]:


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer)
}


# In[4]:


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# In[5]:


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(t_total * args.warmup_portion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = {k: v.to(args.device) for k, v in batch.items()}
            inputs = {'input_ids':       batch['input_ids'],
                      'attention_mask':  batch['input_mask'],
                      'start_positions': batch['start_position'],
                      'end_positions':   batch['end_position']}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch['segment_ids']

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss', (tr_loss - logging_loss)/args.logging_steps, global_step)                    
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(
                        output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


# In[6]:


def evaluate(args, model, tokenizer, prefix=""):
    if prefix == 'test':
        eval_file = args.test_file
    else:
        eval_file = args.dev_file

    DatasetClass = WeakSupervisorDataset
    dataset = DatasetClass(eval_file, args.max_seq_length, tokenizer,
                           args.load_small, is_training=False)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    predict_dir = os.path.join(args.output_dir, 'predictions')
    if not os.path.exists(predict_dir) and args.local_rank in [-1, 0]:
        os.makedirs(predict_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(
    #     dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        
        example_ids = batch['example_id']
        batch = {k: v.to(args.device)
                 for k, v in batch.items() if k != 'example_id'}
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'],
                      'attention_mask': batch['input_mask']
                      }
            if args.model_type != 'distilbert':
                # XLM don't use segment_ids
                inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch['segment_ids']
            # example_ids = batch['example_id']
            outputs = model(**inputs)

        for i, example_id in enumerate(example_ids):
            result = RawResult(unique_id=example_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]),
                               retrieval_logits=[1])  # retrieval_logits is not used            
            all_results.append(result)
            
    examples = dataset.all_examples
    features = dataset.all_features
    # assert len(examples) == len(dataset), (len(examples), len(dataset))
    # assert len(features) == len(dataset), (len(features), len(dataset))

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)",
                evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(
        predict_dir, "instance_predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(
        predict_dir, "instance_nbest_predictions_{}.json".format(prefix))
    output_final_prediction_file = os.path.join(
        predict_dir, "final_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            predict_dir, "instance_null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None


    all_predictions = write_predictions(examples, features, all_results, args.n_best_size,
                                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                                        args.version_2_with_negative, args.null_score_diff_threshold)
    write_weak_supervisor_predictions(all_predictions, output_final_prediction_file)
    eval_metrics = weak_supervisor_eval(eval_file, output_final_prediction_file)

    metrics_file = os.path.join(
        predict_dir, "metrics_{}.json".format(prefix))
    with open(metrics_file, 'w') as fout:
        json.dump(eval_metrics, fout)

    return eval_metrics
    # python scorer.py --val_file /mnt/scratch/chenqu/orconvqa/v3/quac/original/val_v0.2.json --model_output /mnt/scratch/chenqu/orconvqa_output/final_predictions_.json --o eval.json
    # Evaluate with the official SQuAD script
    # evaluate_options = EVAL_OPTS(data_file=args.predict_file,
    #                              pred_file=output_prediction_file,
    #                              na_prob_file=output_null_log_odds_file)
    # results = evaluate_on_squad(evaluate_options)
    # return results


# In[7]:


parser = argparse.ArgumentParser()

# Required parameters

# quac:
# parser.add_argument("--train_file", default='/mnt/scratch/chenqu/orconvqa/paraphrase/final/train.txt',
#                     type=str, required=False,
#                     help="json for training. ")
# parser.add_argument("--dev_file", default='/mnt/scratch/chenqu/orconvqa/paraphrase/final/dev.txt',
#                     type=str, required=False,
#                     help="json for predictions.")

# coqa:
# parser.add_argument("--train_file", default='/mnt/scratch/chenqu/orconvqa/coqa/paraphrase/train.txt',
#                     type=str, required=False,
#                     help="json for training. ")
# parser.add_argument("--dev_file", default='/mnt/scratch/chenqu/orconvqa/coqa/paraphrase/dev.txt',
#                     type=str, required=False,
#                     help="json for predictions.")

# coqa v2:
parser.add_argument("--train_file", default='/mnt/scratch/chenqu/orconvqa/coqa/paraphrase/train_v2.txt',
                    type=str, required=False,
                    help="json for training. ")
parser.add_argument("--dev_file", default='/mnt/scratch/chenqu/orconvqa/coqa/paraphrase/dev_v2.txt',
                    type=str, required=False,
                    help="json for predictions.")

# parser.add_argument("--test_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/test_7.txt',
#                     type=str, required=False,
#                     help="json for predictions.")
parser.add_argument("--model_type", default='bert', type=str, required=False,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
# bert-tiny
# parser.add_argument("--model_name_or_path", default='google/bert_uncased_L-2_H-128_A-2', type=str, required=False,
#                     help="Path to pre-trained model or shortcut name")
# bert-mini
parser.add_argument("--model_name_or_path", default='google/bert_uncased_L-4_H-256_A-4', type=str, required=False,
                    help="Path to pre-trained model or shortcut name")
# bert-small
# parser.add_argument("--model_name_or_path", default='google/bert_uncased_L-4_H-512_A-8', type=str, required=False,
#                     help="Path to pre-trained model or shortcut name")
parser.add_argument("--output_dir", default='/mnt/scratch/chenqu/orconvqa_output/weak_supervisor_9/', type=str, required=False,
                    help="The output directory where the model checkpoints and predictions will be written.")

# Other parameters
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="/mnt/scratch/chenqu/huggingface_cache/", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")

parser.add_argument('--version_2_with_negative', default=True, type=str2bool, required=False,
                    help='If true, the SQuAD examples contain some that do not have an answer.')
parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                    help="If null_score - best_non_null is greater than the threshold predict null.")

parser.add_argument("--max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=384, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument("--max_query_length", default=125, type=int,
                    help="The maximum number of tokens for the question. Questions longer than this will "
                         "be truncated to this length.")
parser.add_argument("--do_train", default=True, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_eval", default=True, type=str2bool,
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_test", default=False, type=str2bool,
                    help="Whether to run eval on the test set.")
parser.add_argument("--evaluate_during_training", default=False, type=str2bool,
                    help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", default=True, type=str2bool,
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=1e-4, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=4.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_portion", default=0.1, type=float,
                    help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
parser.add_argument("--n_best_size", default=20, type=int,
                    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--max_answer_length", default=40, type=int,
                    help="The maximum length of an answer that can be generated. This is needed because the start "
                         "and end predictions are not conditioned on one another.")
parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                         "A number of warnings are expected for a normal SQuAD evaluation.")

parser.add_argument('--logging_steps', type=int, default=5,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=5000,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", default=True, type=str2bool,
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", default=False, type=str2bool,
                    help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', default=False, type=str2bool,
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16', default=False, type=str2bool,
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--server_ip', type=str, default='',
                    help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='',
                    help="Can be used for distant debugging.")

parser.add_argument("--load_small", default=False, type=str2bool, required=False,
                    help="whether to load just a small portion of data during development")
parser.add_argument("--num_workers", default=4, type=int, required=False,
                    help="number of workers for dataloader")

args, unknown = parser.parse_known_args()

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

# Setup distant debugging if needed
if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(
        address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1
args.device = device

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
               args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

# Set seed
set_seed(args)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                      cache_dir=args.cache_dir if args.cache_dir else None)
tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                            do_lower_case=args.do_lower_case,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
model = model_class.from_pretrained(args.model_name_or_path,
                                    from_tf=bool(
                                        '.ckpt' in args.model_name_or_path),
                                    config=config,
                                    cache_dir=args.cache_dir if args.cache_dir else None)

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

model.to(args.device)

logger.info("Training/evaluation parameters %s", args)

# Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
# Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
# remove the need for this code, but it is still valid.
if args.fp16:
    try:
        import apex
        apex.amp.register_half_function(torch, 'einsum')
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

# Training
if args.do_train:
    DatasetClass = WeakSupervisorDataset
    train_dataset = DatasetClass(args.train_file, args.max_seq_length, tokenizer,
                                 args.load_small, is_training=True)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s",
                global_step, tr_loss)

# Save the trained model and the tokenizer
if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model
    final_checkpoint_output_dir = os.path.join(
        args.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(final_checkpoint_output_dir):
        os.makedirs(final_checkpoint_output_dir)

    model_to_save.save_pretrained(final_checkpoint_output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(
        final_checkpoint_output_dir, 'training_args.bin'))

    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(
        final_checkpoint_output_dir, config=config, force_download=True)
    tokenizer = tokenizer_class.from_pretrained(
        args.output_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)


# In[8]:


# Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory

results = {}
max_f1 = 0.0
best_metrics = {}
if args.do_eval and args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
        checkpoints = list(os.path.dirname(c) for c in sorted(
            glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("transformers.modeling_utils").setLevel(
            logging.WARN)  # Reduce model loading logs

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        # Reload the model
        global_step = checkpoint.split(
            '-')[-1] if len(checkpoints) > 1 else ""
        model = model_class.from_pretrained(
            checkpoint, force_download=True)
        model.to(args.device)

        # Evaluate
        result = evaluate(args, model, tokenizer, prefix=global_step)
        if result['f1'] > max_f1:
            max_f1 = result['f1']
            best_metrics['f1'] = result['f1']
            best_metrics['overlap_f1'] = result['overlap_f1']
            best_metrics['cannotanswer_f1'] = result['cannotanswer_f1']
            best_metrics['global_step'] = global_step

        for key, value in result.items():
            tb_writer.add_scalar(
                'eval_{}'.format(key), value, global_step)

        result = dict((k + ('_{}'.format(global_step) if global_step else ''), v)
                      for k, v in result.items())
        results.update(result)

best_metrics_file = os.path.join(
    args.output_dir, 'predictions', 'best_metrics.json')
with open(best_metrics_file, 'w') as fout:
    json.dump(best_metrics, fout)

logger.info("Results: {}".format(results))
logger.info("best metrics: {}".format(best_metrics))


# In[10]:


# if args.do_test and args.local_rank in [-1, 0]:
#     best_global_step = best_metrics['global_step']
#     best_checkpoint = os.path.join(
#         args.output_dir, 'checkpoint-{}'.format(best_global_step))
#     logger.info("Test the best checkpoint: %s", best_checkpoint)

#     model = model_class.from_pretrained(
#         best_checkpoint, force_download=True)
#     model.to(args.device)

#     # Evaluate
#     result = evaluate(args, model, tokenizer, prefix='test')

#     test_metrics_file=os.path.join(
#         args.output_dir, 'predictions', 'test_metrics.json')
#     with open(test_metrics_file, 'w') as fout:
#         json.dump(result, fout)

#     logger.info("Test Result: {}".format(result))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer,
                          AlbertConfig, AlbertTokenizer,
                          BertForQuestionAnswering)
from pathlib import Path
import argparse
import logging
import random
import glob
import timeit
import json
import linecache
import faiss
import numpy as np
import pickle as pkl
from collections import defaultdict
from tqdm import tqdm, trange
import pytrec_eval
import scipy as sp
from copy import copy
import spacy
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from retriever_utils import RetrieverDataset
from modeling import Pipeline, AlbertForRetrieverOnlyPositivePassage, BertForOrconvqaGlobal
from scorer import quac_eval, f1_score
from utils import (LazyQuacDatasetGlobal, RawResult,
                   write_predictions, write_final_predictions,
                   get_retrieval_metrics, gen_reader_features,
                   is_whitespace, QuacExample, convert_example_to_feature,
                   write_weak_supervisor_predictions, get_train_retriever_run)



try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


# In[ ]:





# In[2]:


logger = logging.getLogger(__name__)

ALL_MODELS = list(BertConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer),
    'retriever': (AlbertConfig, AlbertForRetrieverOnlyPositivePassage, AlbertTokenizer),
    'supervisor': (BertConfig, BertForQuestionAnswering, BertTokenizer),
}


# In[3]:


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


# In[4]:


def train(args, train_dataset, model, retriever_tokenizer, reader_tokenizer):
    """ Train the model """
    global em_answer_found_num
    
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
        # to compensate skipped steps:
        # args.num_train_epochs = args.num_train_epochs * 3
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
        # model.to(f'cuda:{model.device_ids[0]}')

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
    retriever_tr_loss, retriever_logging_loss = 0.0, 0.0
    reader_tr_loss, reader_logging_loss = 0.0, 0.0
    qa_tr_loss, qa_logging_loss = 0.0, 0.0
    rerank_tr_loss, rerank_logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=args.local_rank not in [-1, 0])
    num_has_positive_per_train_epoch = 0
    train_info_file = os.path.join(args.output_dir, "train_info.txt")
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])

        all_train_retriever_run = {}  # retriever results per epoch during training
        # measure how many positive passages identified by weak supervision are gold passages
        all_train_gold_passage_hit_run = {}

        for step, batch in enumerate(epoch_iterator):
            model.eval()  # we first get query representations in eval mode
            qids = np.asarray(batch['qid']).reshape(-1).tolist()
            # print('qids', qids)
            question_texts = np.asarray(
                batch['question_text']).reshape(-1).tolist()
            # print('question_texts', question_texts)
            answer_texts = np.asarray(
                batch['answer_text']).reshape(-1).tolist()
            # print('answer_texts', answer_texts)
            answer_starts = np.asarray(
                batch['answer_start']).reshape(-1).tolist()
            # print('answer_starts', answer_starts)
            query_reps = gen_query_reps(args, model, batch)

            if args.weak_supervision == 'none':
                retrieval_results = retrieve(args, qids, qid_to_idx, query_reps,
                                             passage_ids, passage_id_to_idx, passage_reps,
                                             qrels, qrels_sparse_matrix,
                                             gpu_index, include_positive_passage=True)
            else:
                retrieval_results = retrieve_weak_supervision(args, qids, qid_to_idx, query_reps,
                                                              passage_ids, passage_id_to_idx, passage_reps,
                                                              gpu_index, answer_texts, answer_starts)
            # obtain these values before dropping cannotanswer instances
            # for monitoring the training process
            pids_for_reader = retrieval_results['pids_for_reader']
            # print('pids_for_reader', pids_for_reader)
            passages_for_reader = retrieval_results['passages_for_reader']
            labels_for_reader = retrieval_results['labels_for_reader']

            # convert retriever results to qrel evaluation format:
            train_retriever_run, train_gold_passage_hit_run = get_train_retriever_run(
                qids, pids_for_reader, labels_for_reader)
            all_train_retriever_run.update(train_retriever_run)
            all_train_gold_passage_hit_run.update(train_gold_passage_hit_run)

            if args.weak_supervision != 'none':

                if args.drop_cannotanswer:
                    weak_answer_texts = retrieval_results['weak_answer_texts']
                    # print(weak_answer_texts)
                    kept_idx = np.where(
                        np.array(weak_answer_texts) != 'CANNOTANSWER')
                    for k, v in retrieval_results.items():
                        if not isinstance(v, int) and len(v) > 0:
                            retrieval_results[k] = np.asarray(v)[kept_idx]

                    for k, v in batch.items():
                        # print(k, type(v), v)
                        if isinstance(v, torch.Tensor):
                            batch[k] = v[kept_idx]

                    qids = np.asarray(qids)[kept_idx]
                    question_texts = np.asarray(question_texts)[kept_idx]

                answer_texts = retrieval_results['weak_answer_texts']
                answer_starts = retrieval_results['weak_answer_starts']
                num_has_positive_per_train_epoch += retrieval_results['num_has_positive']

            if (args.weak_supervision != 'none' and
                args.drop_cannotanswer and
                retrieval_results['num_has_positive'] == 0):
                continue

            # obtain these values again after dropping cannotanswer instances
            pids_for_reader = retrieval_results['pids_for_reader']
            # print('pids_for_reader', pids_for_reader)
            passages_for_reader = retrieval_results['passages_for_reader']
            labels_for_reader = retrieval_results['labels_for_reader']

            if args.early_loss:
                passage_reps_for_retriever = retrieval_results['passage_reps_for_retriever']
                labels_for_retriever = retrieval_results['labels_for_retriever']

            # skip this step if there are instances are discarded
            # if len(pids_for_reader) == 0:
            #     print('all instances are skipped in this batch')
            #     continue

            if args.real_joint_learn:
                passage_reps_for_reader = retrieval_results['passage_reps_for_reader']

            model.train()

            if args.early_loss:
                inputs = {'query_input_ids': batch['query_input_ids'].to(args.device),
                          'query_attention_mask': batch['query_attention_mask'].to(args.device),
                          'query_token_type_ids': batch['query_token_type_ids'].to(args.device),
                          'passage_rep': torch.from_numpy(passage_reps_for_retriever).to(args.device),
                          'retrieval_label': torch.from_numpy(labels_for_retriever).to(args.device)}
                retriever_outputs = model.retriever(**inputs)
                # model outputs are always tuple in transformers (see doc)
                retriever_loss = retriever_outputs[0]

            if args.real_joint_learn:
                inputs = {'query_input_ids': batch['query_input_ids'].to(args.device),
                          'query_attention_mask': batch['query_attention_mask'].to(args.device),
                          'query_token_type_ids': batch['query_token_type_ids'].to(args.device),
                          'passage_rep': torch.from_numpy(passage_reps_for_reader).to(args.device)}
                retriever_outputs_for_reader = model.retriever(**inputs)
                retriever_logits_for_reader = retriever_outputs_for_reader[0]
                # print('retriever_logits_for_reader',
                #       retriever_logits_for_reader.size())

            reader_batch = gen_reader_features(qids, question_texts, answer_texts, answer_starts,
                                               pids_for_reader, passages_for_reader, labels_for_reader,
                                               reader_tokenizer, args.reader_max_seq_length, is_training=True)

            reader_batch = {k: v.to(args.device)
                            for k, v in reader_batch.items()}
            inputs = {'input_ids':       reader_batch['input_ids'],
                      'attention_mask':  reader_batch['input_mask'],
                      'token_type_ids':  reader_batch['segment_ids'],
                      'start_positions': reader_batch['start_position'],
                      'end_positions':   reader_batch['end_position'],
                      'retrieval_label': reader_batch['retrieval_label']}
            if args.real_joint_learn:
                inputs['retriever_logits'] = retriever_logits_for_reader
            # print(reader_batch['start_position'])
            # print(answer_texts)
            reader_outputs = model.reader(**inputs)
            reader_loss, qa_loss, rerank_loss = reader_outputs[0:3]

            if args.early_loss:
                loss = retriever_loss + reader_loss
            else:
                loss = reader_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if args.early_loss:
                    retriever_loss = retriever_loss.mean()
                reader_loss = reader_loss.mean()
                qa_loss = qa_loss.mean()
                rerank_loss = rerank_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                if args.early_loss:
                    retriever_loss = retriever_loss / args.gradient_accumulation_steps
                reader_loss = reader_loss / args.gradient_accumulation_steps
                qa_loss = qa_loss / args.gradient_accumulation_steps
                rerank_loss = rerank_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if args.early_loss:
                retriever_tr_loss += retriever_loss.item()
            reader_tr_loss += reader_loss.item()
            qa_tr_loss += qa_loss.item()
            rerank_tr_loss += rerank_loss.item()
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
                    if args.early_loss:
                        tb_writer.add_scalar(
                            'retriever_loss', (retriever_tr_loss - retriever_logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'reader_loss', (reader_tr_loss - reader_logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'qa_loss', (qa_tr_loss - qa_logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'rerank_loss', (rerank_tr_loss - rerank_logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss
                    if args.early_loss:
                        retriever_logging_loss = retriever_tr_loss
                    reader_logging_loss = reader_tr_loss
                    qa_logging_loss = qa_tr_loss
                    rerank_logging_loss = rerank_tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    retriever_model_dir = os.path.join(output_dir, 'retriever')
                    reader_model_dir = os.path.join(output_dir, 'reader')
                    if not os.path.exists(retriever_model_dir):
                        os.makedirs(retriever_model_dir)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if not os.path.exists(reader_model_dir):
                        os.makedirs(reader_model_dir)

                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    retriever_model_to_save = model_to_save.retriever
                    retriever_model_to_save.save_pretrained(
                        retriever_model_dir)
                    reader_model_to_save = model_to_save.reader
                    reader_model_to_save.save_pretrained(reader_model_dir)

                    torch.save(args, os.path.join(
                        output_dir, 'training_args.bin'))

                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # print(all_train_retriever_run)
        assert len(all_train_retriever_run) == len(train_dataset), (
            len(all_train_retriever_run), len(train_dataset))
        assert len(all_train_gold_passage_hit_run) == len(train_dataset), (
            len(all_train_gold_passage_hit_run), len(train_dataset))
        # print('all_train_retriever_run', all_train_retriever_run)
        # print('all_train_gold_passage_hit_run', all_train_gold_passage_hit_run)

        train_retriever_run_file = os.path.join(
            args.output_dir, "train_retriever_run_{}.json".format(global_step))
        with open(train_retriever_run_file, 'w') as fout:
            json.dump(all_train_retriever_run, fout)

        train_retriever_metrics = recall_evaluator.evaluate(
            all_train_retriever_run)
        train_retriever_recall_list = [v['recall_5']
                                       for v in train_retriever_metrics.values()]
        train_retriever_recall = np.average(train_retriever_recall_list)
        tb_writer.add_scalar('train_retriever_recall',
                             train_retriever_recall, global_step)
        # print('train_retriever_recall_list', train_retriever_recall_list)

        # measure how many positive passages identified by weak supervision are gold passages
        gold_hit_metrics = p1_evaluator.evaluate(
            all_train_gold_passage_hit_run)
        gold_hit_list = [v['P_1'] for v in gold_hit_metrics.values()]
        gold_hit_p1 = np.average(gold_hit_list)
        tb_writer.add_scalar('gold hit percent', gold_hit_p1, global_step)
        # print('gold_hit_list', gold_hit_list)

        # print('num_has_positive_per_train_epoch', num_has_positive_per_train_epoch)
        if args.weak_supervision != 'none':
            # sanitiy check
            num_has_positive_per_train_epoch_2nd_approach = 0
            for v in all_train_gold_passage_hit_run.values():
                for vv in v.values():
                    num_has_positive_per_train_epoch_2nd_approach += vv
            num_has_positive_per_train_epoch_2nd_approach = int(
                num_has_positive_per_train_epoch_2nd_approach)
            assert num_has_positive_per_train_epoch_2nd_approach == num_has_positive_per_train_epoch, (
                num_has_positive_per_train_epoch_2nd_approach, num_has_positive_per_train_epoch)

        tb_writer.add_scalar('num_has_positive_per_train_epoch (weak supervision only)',
                             num_has_positive_per_train_epoch, global_step)
        num_has_positive_per_train_epoch_percent = num_has_positive_per_train_epoch /             len(train_dataset)
        tb_writer.add_scalar('num_has_positive_per_train_epoch_percent(weak supervision only)',
                             num_has_positive_per_train_epoch_percent, global_step)

        epoch_train_info = {'step': global_step,
                            'num_has_positive_per_train_epoch (weak supervision only)': num_has_positive_per_train_epoch,
                            'num_has_positive_per_train_epoch_percent (weak supervision only)': num_has_positive_per_train_epoch_percent,
                            'gold hit percent': gold_hit_p1,
                            'train_retriever_recall': train_retriever_recall,
                            'em_answer_found_num (em+learned only)': em_answer_found_num}
        with open(train_info_file, 'a') as fout:
            fout.write(json.dumps(epoch_train_info, indent=1) + '\n')

        num_has_positive_per_train_epoch = 0
        em_answer_found_num = 0

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


# In[5]:


def evaluate(args, model, retriever_tokenizer, reader_tokenizer, prefix=""):
    if prefix == 'test':
        eval_file = args.test_file
        orig_eval_file = args.orig_test_file
    else:
        eval_file = args.dev_file
        orig_eval_file = args.orig_dev_file
    pytrec_eval_evaluator = evaluator

    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    DatasetClass = RetrieverDataset
    dataset = DatasetClass(eval_file, retriever_tokenizer,
                           args.load_small, args.history_num,
                           query_max_seq_length=args.retriever_query_max_seq_length,
                           is_pretraining=args.is_pretraining,
                           given_query=True,
                           given_passage=False,
                           include_first_for_retriever=args.include_first_for_retriever)

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
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    retriever_run_dict, rarank_run_dict = {}, {}
    examples, features = {}, {}
    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        qids = np.asarray(batch['qid']).reshape(-1).tolist()
        # print(qids)
        question_texts = np.asarray(
            batch['question_text']).reshape(-1).tolist()
        answer_texts = np.asarray(
            batch['answer_text']).reshape(-1).tolist()
        answer_starts = np.asarray(
            batch['answer_start']).reshape(-1).tolist()
        query_reps = gen_query_reps(args, model, batch)

        # if args.weak_supervision == 'none':
        # during evaluation, no weak answers need to be matched
        # so we use retrieve instead of retrieve_weak_supervision
        retrieval_results = retrieve(args, qids, qid_to_idx, query_reps,
                                     passage_ids, passage_id_to_idx, passage_reps,
                                     qrels, qrels_sparse_matrix,
                                     gpu_index, include_positive_passage=False)
#         else:
#             retrieval_results = retrieve_weak_supervision(args, qids, qid_to_idx, query_reps,
#                                                           passage_ids, passage_id_to_idx, passage_reps,
#                                                           gpu_index, answer_texts, answer_starts)

#         if args.weak_supervision != 'none':

#             weak_answer_texts = retrieval_results['weak_answer_texts']
#             weak_answer_starts = retrieval_results['weak_answer_starts']
#             # if len(weak_answer_texts) > 0 and len(weak_answer_starts) > 0 and \
#             # (weak_answer_texts[0] != answer_texts[0] or weak_answer_starts[0] != answer_starts[0]):
#             #     print('answer_texts', answer_texts, 'weak_answer_texts', weak_answer_texts)
#             #     print('answer_starts', answer_starts, 'weak_answer_starts', weak_answer_starts)
#             answer_texts = weak_answer_texts
#             answer_starts = weak_answer_starts

        retriever_probs = retrieval_results['retriever_probs']
        # print('retriever_probs before', retriever_probs)
        pids_for_reader = retrieval_results['pids_for_reader']
        passages_for_reader = retrieval_results['passages_for_reader']
        labels_for_reader = retrieval_results['labels_for_reader']

        if args.real_joint_learn:
            passage_reps_for_reader = retrieval_results['passage_reps_for_reader']

        if args.real_joint_learn:
            with torch.no_grad():
                inputs = {'query_input_ids': batch['query_input_ids'].to(args.device),
                          'query_attention_mask': batch['query_attention_mask'].to(args.device),
                          'query_token_type_ids': batch['query_token_type_ids'].to(args.device),
                          'passage_rep': torch.from_numpy(passage_reps_for_reader).to(args.device)}
                retriever_outputs_for_reader = model.retriever(**inputs)
                retriever_logits_for_reader = retriever_outputs_for_reader[0]

        reader_batch, batch_examples, batch_features = gen_reader_features(qids, question_texts, answer_texts,
                                                                           answer_starts, pids_for_reader,
                                                                           passages_for_reader, labels_for_reader,
                                                                           reader_tokenizer,
                                                                           args.reader_max_seq_length,
                                                                           is_training=False)
        example_ids = reader_batch['example_id']
        # print('example_ids', example_ids)
        examples.update(batch_examples)
        features.update(batch_features)
        reader_batch = {k: v.to(args.device)
                        for k, v in reader_batch.items() if k != 'example_id'}
        with torch.no_grad():
            inputs = {'input_ids':      reader_batch['input_ids'],
                      'attention_mask': reader_batch['input_mask'],
                      'token_type_ids': reader_batch['segment_ids']}
            if args.real_joint_learn:
                inputs['retriever_logits'] = retriever_logits_for_reader
            outputs = model.reader(**inputs)

        retriever_probs = retriever_probs.reshape(-1).tolist()
        # print('retriever_probs after', retriever_probs)
        if args.real_joint_learn:
            for i, example_id in enumerate(example_ids):
                result = RawResult(unique_id=example_id,
                                   start_logits=to_list(outputs[0][i]),
                                   end_logits=to_list(outputs[1][i]),
                                   retrieval_logits=to_list(outputs[2][i]),
                                   retriever_prob=to_list(outputs[3][i][0]))  # [i][0] is equivalent to squeeze
                all_results.append(result)

        else:
            for i, example_id in enumerate(example_ids):
                result = RawResult(unique_id=example_id,
                                   start_logits=to_list(outputs[0][i]),
                                   end_logits=to_list(outputs[1][i]),
                                   retrieval_logits=to_list(outputs[2][i]),
                                   retriever_prob=retriever_probs[i])
                all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)",
                evalTime, evalTime / len(dataset))

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
    dialog_level_preds = write_final_predictions(all_predictions, output_final_prediction_file,
                                                 use_rerank_prob=args.use_rerank_prob,
                                                 use_retriever_prob=args.use_retriever_prob,
                                                 real_joint_learn=args.real_joint_learn,
                                                 involve_rerank_in_real_joint_learn=args.involve_rerank_in_real_joint_learn)
    eval_metrics = quac_eval(
        orig_eval_file, output_final_prediction_file)
    rerank_metrics = get_retrieval_metrics(
        pytrec_eval_evaluator, all_predictions, eval_retriever_probs=True)
    eval_metrics.update(rerank_metrics)
    
    # how many answers in dev/test prediction results come from the gold passage
    def get_eval_answer_from_gold_num(dialog_level_preds, qrels):
        eval_answer_from_gold_num = 0
        for v in dialog_level_preds:
            for each_qid, each_pid in zip(v['qid'], v['pid']):
                # if each_pid in qrels[each_qid]:
                if each_pid in qrels.get(each_qid, {}): # compatible with coqa, coqa has no qrels
                    eval_answer_from_gold_num += 1
        
        return eval_answer_from_gold_num
    
    eval_answer_from_gold_num = get_eval_answer_from_gold_num(dialog_level_preds, qrels)
    eval_answer_from_gold_persent = eval_answer_from_gold_num / len(dataset)
    eval_metrics['eval_answer_from_gold_persent'] = eval_answer_from_gold_persent

    metrics_file = os.path.join(
        predict_dir, "metrics_{}.json".format(prefix))
    with open(metrics_file, 'w') as fout:
        json.dump(eval_metrics, fout)

    return eval_metrics


# In[6]:


def gen_query_reps(args, model, batch):
    model.eval()
    batch = {k: v.to(args.device) for k, v in batch.items() 
             if k not in ['example_id', 'qid', 'question_text', 'answer_text', 'answer_start']}
    with torch.no_grad():
        inputs = {}
        inputs['query_input_ids'] = batch['query_input_ids']
        inputs['query_attention_mask'] = batch['query_attention_mask']
        inputs['query_token_type_ids'] = batch['query_token_type_ids']
        outputs = model.retriever(**inputs)
        query_reps = outputs[0]

    return query_reps


# In[7]:


def retrieve(args, qids, qid_to_idx, query_reps,
             passage_ids, passage_id_to_idx, passage_reps,
             qrels, qrels_sparse_matrix,
             gpu_index, include_positive_passage=False):
    query_reps = query_reps.detach().cpu().numpy()
    D, I = gpu_index.search(query_reps, args.top_k_for_retriever)

    pidx_for_retriever = np.copy(I)
    qidx = [qid_to_idx[qid] for qid in qids]
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, args.top_k_for_retriever, axis=1)
    labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray(
    )
    # print('labels_for_retriever before', labels_for_retriever)
    if include_positive_passage:
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_retriever)):
            has_positive = np.sum(labels_per_query)
            if not has_positive:
                positive_pid = list(qrels[qid].keys())[0]
                positive_pidx = passage_id_to_idx[positive_pid]
                pidx_for_retriever[i][-1] = positive_pidx
        labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray(
        )
        # print('labels_for_retriever after', labels_for_retriever)
        assert np.sum(labels_for_retriever) >= len(labels_for_retriever)
    pids_for_retriever = passage_ids[pidx_for_retriever]
    passage_reps_for_retriever = passage_reps[pidx_for_retriever]

    scores = D[:, :args.top_k_for_reader]
    retriever_probs = sp.special.softmax(scores, axis=1)
    pidx_for_reader = I[:, :args.top_k_for_reader]
    # print('pidx_for_reader', pidx_for_reader)
    # print('qids', qids)
    # print('qidx', qidx)
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, args.top_k_for_reader, axis=1)
    # print('qidx_expanded', qidx_expanded)

    labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray(
    )
    # print('labels_for_reader before', labels_for_reader)
    # print('labels_for_reader before', labels_for_reader)
    if include_positive_passage:
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_reader)):
            has_positive = np.sum(labels_per_query)
            if not has_positive:
                positive_pid = list(qrels[qid].keys())[0]
                positive_pidx = passage_id_to_idx[positive_pid]
                pidx_for_reader[i][-1] = positive_pidx
        labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray(
        )
        # print('labels_for_reader after', labels_for_reader)
        assert np.sum(labels_for_reader) >= len(labels_for_reader)
    # print('labels_for_reader after', labels_for_reader)
    pids_for_reader = passage_ids[pidx_for_reader]
    # print('pids_for_reader', pids_for_reader)
    passages_for_reader = get_passages(pidx_for_reader, args)
    # we do not need to modify scores and probs matrices because they will only be
    # needed at evaluation, where include_positive_passage will be false
    if args.real_joint_learn:
        passage_reps_for_reader = passage_reps[pidx_for_reader]

    return_dict = {'qidx': qidx,
                   'pidx_for_retriever': pidx_for_retriever,
                   'pids_for_retriever': pids_for_retriever,
                   'passage_reps_for_retriever': passage_reps_for_retriever,
                   'labels_for_retriever': labels_for_retriever,
                   'retriever_probs': retriever_probs,
                   'pidx_for_reader': pidx_for_reader,
                   'pids_for_reader': pids_for_reader,
                   'passages_for_reader': passages_for_reader,
                   'labels_for_reader': labels_for_reader}
    
    if args.real_joint_learn:
        return_dict['passage_reps_for_reader'] = passage_reps_for_reader
        
    return return_dict


# In[8]:


def retrieve_weak_supervision(args, qids, qid_to_idx, query_reps,
                              passage_ids, passage_id_to_idx, passage_reps,
                              gpu_index, answer_texts, answer_starts):
    find_weak_answer_funcs = {
        'em': find_weak_answer_em,
        'f1': find_weak_answer_f1,
        'learned': find_weak_answer_learned,
        'em+learned': find_weak_answer_em_learned}
    find_weak_answer_func = find_weak_answer_funcs[args.weak_supervision]

    query_reps = query_reps.detach().cpu().numpy()
    D, I = gpu_index.search(query_reps, args.top_k_for_retriever)

    pidx_for_retriever = I[:, :args.top_k_for_retriever]
    # qidx = [qid_to_idx[qid] for qid in qids]

    pidx_for_reader = I[:, :args.top_k_for_reader]
    pids_for_reader = passage_ids[pidx_for_reader]
    passages_for_reader = get_passages(pidx_for_reader, args)
    # print('passages_for_reader', type(passages_for_reader), passages_for_reader)

    # qidx_expanded = np.expand_dims(qidx, axis=1)
    # qidx_expanded = np.repeat(qidx_expanded, args.top_k_for_reader, axis=1)
    # true_labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()

    # print('pidx_for_reader', pidx_for_reader)
    # print('qids', qids)
    # print('qidx', qidx)

    num_has_positive = 0

    labels_for_reader = []
    weak_answer_starts = []
    weak_answer_texts = []
    # valid_idx = []  # queries with at least one gold passage identifed by weak supervision is valid
    # queries (training instances) w/o any gold passage will be discarded
    new_passages_for_reader_list = []
    for i, (qid, answer_text, answer_start, passages_per_query) in enumerate(zip(qids, answer_texts, answer_starts, passages_for_reader)):
        (labels_per_query, final_weak_answer_start, weak_answer_text,
         int_has_positive, new_passages_per_query) = find_weak_answer_func(
            args, qid, answer_text, passages_per_query)
        new_passages_for_reader_list.append(new_passages_per_query)

        # print('int_has_positive:', int_has_positive)
        # print('answer_start:', answer_start)
        # print('answer_text:', answer_text)
        # print('final_weak_answer_start:', final_weak_answer_start)
        # print('weak_answer_text:', weak_answer_text)
        # print('labels_per_query:', labels_per_query)
        # print('passages_per_query', passages_per_query)

        num_has_positive += int_has_positive
        labels_for_reader.append(copy(labels_per_query))
        weak_answer_starts.append(final_weak_answer_start)
        weak_answer_texts.append(weak_answer_text)
        # if weak_answer_starts[i] != -1 and weak_answer_starts[i] != answer_start:
        #     print('answer_start', answer_start, 'weak_answer_start', weak_answer_starts[i])
        #     print('true_labels_per_query', true_labels_for_reader[i], 'labels_per_query', labels_per_query)
        #     print('passages_for_reader', passages_for_reader)
    new_passages_for_reader = np.vstack(new_passages_for_reader_list)

    # print('new_passages_for_reader', type(new_passages_for_reader), new_passages_for_reader)

    scores = D[:, :args.top_k_for_reader]
    retriever_probs = sp.special.softmax(scores, axis=1)
    labels_for_reader = np.array(labels_for_reader)
    weak_answer_starts = np.array(weak_answer_starts)

    assert len(weak_answer_texts) == len(
        answer_texts), ('len (weak)_answer_texts', weak_answer_texts, answer_texts)
    # assert np.sum(labels_for_reader) >= len(labels_for_reader)

    if args.early_loss:
        pids_for_retriever = passage_ids[pidx_for_retriever]
        passage_reps_for_retriever = passage_reps[pidx_for_retriever]

        
        top_k_diff = args.top_k_for_retriever - args.top_k_for_reader
        assert top_k_diff >= 0, 'top_k_for_retriever should >= top_k_for_reader'
        batch_size, block_num = labels_for_reader.shape
        # since we have already identified a positive passage in top_k_for_reader,
        # the rest of top_k_for_retriever will be considered as negative passages
        labels_for_retriever = np.hstack((labels_for_reader,
                                         np.zeros((batch_size, top_k_diff), dtype=int)))
        # print('labels_for_reader', labels_for_reader)
        # print('labels_for_retriever', labels_for_retriever)
#         if args.top_k_for_retriever == args.top_k_for_reader:
#             # print('top_k_for_retriever == top_k_for_reader')
#             pids_for_retriever = np.copy(pids_for_reader)
#             passage_reps_for_retriever = passage_reps[pidx_for_retriever]
#             labels_for_retriever = np.copy(labels_for_reader)
#         else:
#             passages_for_retriever = get_passages(pidx_for_retriever, args)
#             labels_for_retriever = []
#             for i, (qid, answer_text, passages_per_query) in enumerate(zip(qids, answer_texts, passages_for_retriever)):
#                 labels_per_query, _, _, _, _ = find_weak_answer_func(
#                     args, qid, answer_text, passages_per_query)

#                 labels_for_retriever.append(copy(labels_per_query))

#             # print('labels_for_retriever', labels_for_retriever)
#             assert np.sum(labels_for_retriever) >= len(labels_for_retriever)

#             pids_for_retriever = passage_ids[pidx_for_retriever]
#             passage_reps_for_retriever = passage_reps[pidx_for_retriever]

    else:
        pidx_for_retriever = []
        pids_for_retriever = []
        passage_reps_for_retriever = []
        labels_for_retriever = []

    # print('pids_for_reader', pids_for_reader)

    # we do not need to modify scores and probs matrices because they will only be
    # needed at evaluation, where include_positive_passage will be false
    if args.real_joint_learn:
        passage_reps_for_reader = passage_reps[pidx_for_reader]

    return_dict = {# 'qidx': qidx,
                   'pidx_for_retriever': pidx_for_retriever,
                   'pids_for_retriever': pids_for_retriever,
                   'passage_reps_for_retriever': passage_reps_for_retriever,
                   'labels_for_retriever': np.array(labels_for_retriever),
                   'retriever_probs': retriever_probs,
                   'pidx_for_reader': pidx_for_reader,
                   'pids_for_reader': pids_for_reader,
                   'passages_for_reader': new_passages_for_reader,
                   'labels_for_reader': labels_for_reader,
                   'weak_answer_starts': weak_answer_starts,
                   'weak_answer_texts': weak_answer_texts,
                   'num_has_positive': num_has_positive}

    if args.real_joint_learn:
        # passages_for_reader = passages_for_reader[valid_idx]
        return_dict['passage_reps_for_reader'] = passage_reps_for_reader

    return return_dict


# In[9]:


def find_weak_answer_em(args, qid, answer_text, passages_per_query):
    labels_per_query = [0] * len(passages_per_query)
    has_positive = False
    final_weak_answer_text = None
    final_weak_answer_start = None
    for i, passage in enumerate(passages_per_query):
        weak_answer_start = passage.lower().find(answer_text.lower())
        if weak_answer_start != -1:
            has_positive = True
            labels_per_query[i] = 1
            final_weak_answer_text = passage[weak_answer_start: weak_answer_start + len(answer_text)]
            final_weak_answer_start = weak_answer_start
            break

    if not has_positive:
        # if no gold passage is identified, we assume the first retrieved passage is true for retriever
        # and cannotanswer is the weak answer for reader
        # labels_per_query[0] = 1
        final_weak_answer_start = 0
        final_weak_answer_text = 'CANNOTANSWER'  
    
    int_has_positive = int(has_positive)
    # print('qid', qid)
    # print('answer_text', answer_text)
    # print('final_weak_answer_text', final_weak_answer_text)
    # print('int_has_positive', int_has_positive)
    return (labels_per_query, final_weak_answer_start, final_weak_answer_text, 
            int_has_positive, passages_per_query)


# In[10]:


def find_weak_answer_f1(args, qid, answer_text, passages_per_query):
    labels_per_query = [0] * len(passages_per_query)
    
    has_positive = False
    max_f1 = -1.0
    best_passage_idx = -1
    best_weak_answer_text = None
    best_weak_answer_start = None
    
    for i, passage in enumerate(passages_per_query):
        doc = nlp(str(passage))
        sents = doc.sents
        for sent in sents:
            sent = sent.text
            weak_answer_f1 = f1_score(sent, answer_text)            
            sent_idx = passage.index(sent)
            if weak_answer_f1 > max_f1:
                max_f1 = weak_answer_f1
                best_passage_idx = i
                best_weak_answer_text = sent
                best_weak_answer_start = sent_idx
    # print('max_f1', max_f1)
    if max_f1 > 0:
        has_positive = True
        labels_per_query[best_passage_idx] = 1
    else:
        # if no gold passage is identified, we assume the first retrieved passage is true for retriever
        # and cannotanswer is the weak answer for reader
        # labels_per_query[0] = 1
        best_weak_answer_start = 0
        best_weak_answer_text = 'CANNOTANSWER'  

    int_has_positive = int(has_positive)
    
    return (labels_per_query, best_weak_answer_start, best_weak_answer_text, 
            int_has_positive, passages_per_query)


# In[11]:


def find_weak_answer_learned(args, qid, answer_text, passages_per_query):
    supervisor_outputs, examples = supervisor_inference(args, qid, answer_text, passages_per_query)
    # to deal with the inconsistent space in passage and answer:
    new_passages_per_qery = [' '.join(example.doc_tokens) for example in examples.values()]
    # print([example.example_id for example in examples.values()])
    # print(new_passages_per_qery)
    labels_per_query = [0] * len(passages_per_query)
    
    has_positive = False
    max_score = float('-inf')
    best_passage_idx = -1
    best_weak_answer_text = None
    best_weak_answer_start = None
    
    for i, (passage, supervisor_output) in enumerate(zip(new_passages_per_qery, supervisor_outputs)):
        weak_answer = supervisor_output['weak_answer']
        score = supervisor_output['score']
        if weak_answer != 'CANNOTANSWER' and weak_answer != 'empty':
            has_positive = True
            if score > max_score:
                max_score = score
                best_passage_idx = i
                best_weak_answer_text = weak_answer
                best_weak_answer_start = passage.find(weak_answer)
                assert best_weak_answer_start != -1, (weak_answer, answer_text, score, passage)
    # print('max_f1', max_f1)
    if has_positive:
        labels_per_query[best_passage_idx] = 1
    else:
        # if no gold passage is identified, we assume the first retrieved passage is true for retriever
        # and cannotanswer is the weak answer for reader
        # labels_per_query[0] = 1
        best_weak_answer_start = 0
        best_weak_answer_text = 'CANNOTANSWER'  

    int_has_positive = int(has_positive)
    # print('max_score', max_score, 'best_weak_answer_text', best_weak_answer_text)
    return (labels_per_query, best_weak_answer_start, best_weak_answer_text, 
            int_has_positive, np.array(new_passages_per_qery))


# In[12]:


def find_weak_answer_em_learned(args, qid, answer_text, passages_per_query):
    global em_answer_found_num
    
    method = None
    em_result = find_weak_answer_em(args, qid, answer_text, passages_per_query)
    em_weak_answer = em_result[2]
    if em_weak_answer != 'CANNOTANSWER':
        # print('using em', em_weak_answer)
        em_answer_found_num += 1
        return_result = em_result
        method = 'em'
    else:
        learned_result = find_weak_answer_learned(args, qid, answer_text, passages_per_query)
        # learned_weak_answer = learned_result[2]
        # print('using learned', 'em:', em_weak_answer, 'learned:', learned_weak_answer)
        return_result = learned_result
        method = 'learned'
    
    if args.case_study:
        (labels_per_query, final_weak_answer_start, weak_answer_text,
             int_has_positive, new_passages_per_query) = return_result
        
        if weak_answer_text != 'CANNOTANSWER':
            log_dict = {'method': method, 
                        'qid': qid, 
                        'answer': answer_text, 
                        'weak_answer': weak_answer_text, 
                        'weak_answer_start': final_weak_answer_start, 
                        'int_has_positive': int_has_positive, 
                        'labels_per_query': labels_per_query, 
                        'new_passages_per_query': new_passages_per_query.tolist()}

            with open(args.case_study_file, 'a') as fout:
                fout.write(json.dumps(log_dict, indent=1) + '\n')
        
    return return_result


# In[13]:


def supervisor_inference(args, qid, answer_text, passages_per_query):

    def gen_supervisor_feature(tokenizer, example_id, answer_text, paragraph_text):
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False

        example = QuacExample(
            example_id=example_id,
            qas_id=example_id,  # qas_id is the same with example id to keep it unique
            question_text=answer_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)

        feature = convert_example_to_feature(
            example, tokenizer, is_training=False)
        feature_dict = {'input_ids': np.asarray(feature.input_ids),
                        'segment_ids': np.asarray(feature.segment_ids),
                        'input_mask': np.asarray(feature.input_mask),
                        'example_id': feature.example_id}
        return feature_dict, example, feature

    def supervisor_collate(feature_dicts):
        collated = {}
        keys = feature_dicts[0].keys()
        for key in keys:
            if key != 'example_id':
                collated[key] = np.vstack([dic[key] for dic in feature_dicts])
                collated[key] = torch.from_numpy(collated[key])
        if 'example_id' in keys:
            collated['example_id'] = [dic['example_id']
                                      for dic in feature_dicts]

        return collated

    examples = {}
    features = {}
    supervisor_feature_dicts = []
    for i, passage in enumerate(passages_per_query):
        example_id = '{}*{}'.format(qid, i)
        supervisor_feature_dict, supervisor_example, supervisor_feature = gen_supervisor_feature(
            supervisor_tokenizer, example_id, answer_text, passage)
        supervisor_feature_dicts.append(supervisor_feature_dict)
        examples[example_id] = supervisor_example
        features[example_id] = supervisor_feature

    batch = supervisor_collate(supervisor_feature_dicts)

    all_results = []

    supervisor_model.eval()
    example_ids = batch['example_id']
    batch = {k: v.to(args.supervisor_device)
             for k, v in batch.items() if k != 'example_id'}
    with torch.no_grad():
        inputs = {'input_ids':      batch['input_ids'],
                  'attention_mask': batch['input_mask'],
                  'token_type_ids': batch['segment_ids']}
        outputs = supervisor_model(**inputs)

    for i, example_id in enumerate(example_ids):
        result = RawResult(unique_id=example_id,
                           start_logits=to_list(outputs[0][i]),
                           end_logits=to_list(outputs[1][i]),
                           retrieval_logits=[1])  # retrieval_logits is not used
        all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(
        args.output_dir, "instance_predictions_{}.json".format('supervisor_temp'))
    output_nbest_file = os.path.join(
        args.output_dir, "instance_nbest_predictions_{}.json".format('supervisor_temp'))
    output_final_prediction_file = os.path.join(
        args.output_dir, "final_predictions_{}.json".format('supervisor_temp'))
    output_null_log_odds_file = os.path.join(
        args.output_dir, "instance_null_odds_{}.json".format('supervisor_temp'))


    all_predictions = write_predictions(examples, features, all_results, args.n_best_size,
                                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                                        True, args.null_score_diff_threshold)
    supervisor_output = write_weak_supervisor_predictions(
        all_predictions, output_final_prediction_file)
    
    # print(supervisor_output)

    return supervisor_output, examples


# In[14]:


def get_passage(i, args):
    line = linecache.getline(args.blocks_path, i + 1)
    line = json.loads(line.strip())
    return line['text']
get_passages = np.vectorize(get_passage)


# In[15]:


parser = argparse.ArgumentParser()

# arguments shared by the retriever and reader

# quac:
parser.add_argument("--train_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/train_no_cannotanswer.txt',
                    type=str, required=False,
                    help="open retrieval quac json for training. ")
parser.add_argument("--dev_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/dev_no_cannotanswer.txt',
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--test_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/test_no_cannotanswer.txt',
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--orig_dev_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/quac/dev_no_cannotanswer.txt',
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--orig_test_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/quac/test_no_cannotanswer.txt',
                    type=str, required=False,
                    help="original quac json for evaluation.")

# coqa:
# parser.add_argument("--train_file", default='/mnt/scratch/chenqu/orconvqa/coqa/preprocessed/train_no_cannotanswer.txt',
#                     type=str, required=False,
#                     help="open retrieval quac json for training. ")
# parser.add_argument("--dev_file", default='/mnt/scratch/chenqu/orconvqa/coqa/preprocessed/dev_no_cannotanswer.txt',
#                     type=str, required=False,
#                     help="open retrieval quac json for predictions.")
# parser.add_argument("--test_file", default='/mnt/scratch/chenqu/orconvqa/coqa/preprocessed/test_no_cannotanswer.txt',
#                     type=str, required=False,
#                     help="open retrieval quac json for predictions.")
# parser.add_argument("--orig_dev_file", default='/mnt/scratch/chenqu/orconvqa/coqa/quac_format/dev_no_cannotanswer.txt',
#                     type=str, required=False,
#                     help="open retrieval quac json for predictions.")
# parser.add_argument("--orig_test_file", default='/mnt/scratch/chenqu/orconvqa/coqa/quac_format/test_no_cannotanswer.txt',
#                     type=str, required=False,
#                     help="original quac json for evaluation.")

parser.add_argument("--qrels", default='/mnt/scratch/chenqu/orconvqa/v5/retrieval/qrels.txt', type=str, required=False,
                    help="qrels to evaluate open retrieval")
# parser.add_argument("--blocks_path", default='/mnt/scratch/chenqu/orconvqa/v3/all/all_blocks.txt', type=str, required=False,
#                     help="all blocks text")
parser.add_argument("--blocks_path", default='/mnt/scratch/chenqu/orconvqa/v3/all/all_blocks.txt', type=str, required=False,
                    help="all blocks text")
parser.add_argument("--passage_reps_path", default='/mnt/scratch/chenqu/orconvqa/v5/passage_reps/combined/passage_reps.pkl',
                    type=str, required=False, help="passage representations")
parser.add_argument("--passage_ids_path", default='/mnt/scratch/chenqu/orconvqa/v5/passage_reps/combined/passage_ids.pkl',
                    type=str, required=False, help="passage ids")
parser.add_argument("--output_dir", default='/mnt/scratch/chenqu/orconvqa_output/weak_sup_test', type=str, required=False,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--load_small", default=True, type=str2bool, required=False,
                    help="whether to load just a small portion of data during development")
parser.add_argument("--num_workers", default=2, type=int, required=False,
                    help="number of workers for dataloader")

parser.add_argument("--global_mode", default=True, type=str2bool, required=False,
                    help="maxmize the prob of the true answer given all passages")
parser.add_argument("--real_joint_learn", default=True, type=str2bool, required=False,
                    help="involve retriever and reranker logits in start/end loss")
parser.add_argument("--involve_rerank_in_real_joint_learn", default=False, type=str2bool, required=False,
                    help="when real_joint_learn set to true, involve reranker logits in start/end loss")
parser.add_argument("--history_num", default=1, type=int, required=False,
                    help="number of history turns to use."
                         "-1 means to use rewrites as questions (without concat history turns)")
parser.add_argument("--prepend_history_questions", default=True, type=str2bool, required=False,
                    help="whether to prepend history questions to the current question")
parser.add_argument("--prepend_history_answers", default=False, type=str2bool, required=False,
                    help="whether to prepend history answers to the current question")

parser.add_argument("--do_train", default=True, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_eval", default=True, type=str2bool,
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_test", default=True, type=str2bool,
                    help="Whether to run eval on the test set.")
parser.add_argument("--best_global_step", default=40, type=int, required=False,
                    help="used when only do_test, this is override if do_eval")
parser.add_argument("--evaluate_during_training", default=False, type=str2bool,
                    help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", default=True, type=str2bool,
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_portion", default=0.1, type=float,
                    help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                         "A number of warnings are expected for a normal SQuAD evaluation.")

parser.add_argument('--logging_steps', type=int, default=1,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=20,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", default=True, type=str2bool,
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", default=False, type=str2bool,
                    help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', default=True, type=str2bool,
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

# retriever arguments
parser.add_argument("--retriever_config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--retriever_model_type", default='albert', type=str, required=False,
                    help="retriever model type")
parser.add_argument("--retriever_model_name_or_path", default='albert-base-v1', type=str, required=False,
                    help="retriever model name")
parser.add_argument("--retriever_tokenizer_name", default="albert-base-v1", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--retriever_cache_dir", default="/mnt/scratch/chenqu/huggingface_cache/albert_v1/", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
# use the pretrained retriever:
parser.add_argument("--retrieve_checkpoint",
                    default='/mnt/scratch/chenqu/orconvqa_output/retriever_33/checkpoint-5917', type=str,
                    help="generate query/passage representations with this checkpoint")
parser.add_argument("--retrieve_tokenizer_dir",
                    default='/mnt/scratch/chenqu/orconvqa_output/retriever_33/', type=str,
                    help="dir that contains tokenizer files")

# use the pretrained + concurrently learned retriever:
# parser.add_argument("--retrieve_checkpoint",
#                     default='/mnt/scratch/chenqu/orconvqa_output/repr_pipeline_18_3/checkpoint-47290/retriever/', type=str,
#                     help="generate query/passage representations with this checkpoint")
# parser.add_argument("--retrieve_tokenizer_dir",
#                     default='/mnt/scratch/chenqu/orconvqa_output/repr_pipeline_18_3/retriever/', type=str,
#                     help="dir that contains tokenizer files")

parser.add_argument("--given_query", default=True, type=str2bool,
                    help="Whether query is given.")
parser.add_argument("--given_passage", default=False, type=str2bool,
                    help="Whether passage is given. Passages are not given when jointly train")
parser.add_argument("--is_pretraining", default=False, type=str2bool,
                    help="Whether is pretraining. We fine tune the query encoder in retriever")
parser.add_argument("--include_first_for_retriever", default=True, type=str2bool,
                    help="include the first question in a dialog in addition to history_num for retriever (not reader)")
# parser.add_argument("--only_positive_passage", default=True, type=str2bool,
#                     help="we only pass the positive passages, the rest of the passges in the batch are considered as negatives")
parser.add_argument("--retriever_query_max_seq_length", default=128, type=int,
                    help="The maximum input sequence length of query.")
parser.add_argument("--retriever_passage_max_seq_length", default=384, type=int,
                    help="The maximum input sequence length of passage (384 + [CLS] + [SEP]).")
parser.add_argument("--proj_size", default=128, type=int,
                    help="The size of the query/passage rep after projection of [CLS] rep.")
parser.add_argument("--top_k_for_retriever", default=10, type=int,
                    help="retrieve top k passages for a query, these passages will be used to update the query encoder")
parser.add_argument("--use_retriever_prob", default=True, type=str2bool,
                    help="include albert retriever probs in final answer ranking")
parser.add_argument("--early_loss", default=True, type=str2bool,
                    help="aggresively fine-tune the retriever with top K_{rt} passages, K_{rt} >> 5")

# reader arguments
parser.add_argument("--reader_config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--reader_model_name_or_path", default='bert-base-uncased', type=str, required=False,
                    help="reader model name")
parser.add_argument("--reader_model_type", default='bert', type=str, required=False,
                    help="reader model type")
parser.add_argument("--reader_tokenizer_name", default="bert-base-uncased", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--reader_cache_dir", default="/mnt/scratch/chenqu/huggingface_cache/", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--reader_max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=384, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument('--version_2_with_negative', default=False, type=str2bool, required=False,
                    help='If true, the SQuAD examples contain some that do not have an answer.')
parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                    help="If null_score - best_non_null is greater than the threshold predict null.")
parser.add_argument("--reader_max_query_length", default=125, type=int,
                    help="The maximum number of tokens for the question. Questions longer than this will "
                         "be truncated to this length.")
parser.add_argument("--n_best_size", default=20, type=int,
                    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--max_answer_length", default=40, type=int,
                    help="The maximum length of an answer that can be generated. This is needed because the start "
                         "and end predictions are not conditioned on one another.")
parser.add_argument("--qa_loss_factor", default=1.0, type=float,
                    help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
parser.add_argument("--retrieval_loss_factor", default=0.0, type=float,
                    help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
parser.add_argument("--top_k_for_reader", default=5, type=int,
                    help="update the reader with top k passages")
parser.add_argument("--use_rerank_prob", default=True, type=str2bool,
                    help="include rerank probs in final answer ranking")

# weak_supervision
parser.add_argument("--weak_supervision", default='em+learned', type=str,
                    help="options: 1. none: add gold passage if not retrieved"
                         "         2. em: identify gold passage by exact match of the known answer, "
                         "                disgard training instance is no exact match"
                         "         3. f1: find a sentence in the retrieved passage that has the largest "
                         "                overlap (world-level f1) with the known answer"
                         "         4. learned: use a learned weak supervisor to find a span in the "
                         "                     retrieved passage that is a paraphrase to the known answer"
                         "         5. em+learned: use em first, if no weak answer, then use learned")
parser.add_argument("--supervisor_checkpoint",
                    default="/mnt/scratch/chenqu/orconvqa_output/weak_supervisor_2/checkpoint-25000/",
                    type=str,
                    help="checkpoint dir for the learned weak supervisor")
parser.add_argument("--drop_cannotanswer", default=True, type=str2bool,
                    help="whether to drop cannotanswer questions in weak superivision")
parser.add_argument("--case_study", default=True, type=str2bool,
                    help="log weak answers, retrieved passages, etc in em+learned mode for case studies")

args, unknown = parser.parse_known_args()

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
args.retriever_tokenizer_dir = os.path.join(args.output_dir, 'retriever')
args.reader_tokenizer_dir = os.path.join(args.output_dir, 'reader')
# Setup distant debugging if needed
if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(
        address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

# Setup CUDA, GPU & distributed training
# we now do not support distributed joint learning
# we will request two cards, one for torch and the other one for faiss
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args.n_gpu = torch.cuda.device_count() - 1
    args.n_gpu = 1
    # torch.cuda.set_device(0)
    if args.weak_supervision in ['learned', 'em+learned']:
        args.supervisor_device = torch.device(
            'cuda:1' if torch.cuda.is_available() and not args.no_cuda else "cpu")
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


model = Pipeline()

args.retriever_model_type = args.retriever_model_type.lower()
retriever_config_class, retriever_model_class, retriever_tokenizer_class = MODEL_CLASSES[
    'retriever']
retriever_config = retriever_config_class.from_pretrained(
    args.retrieve_checkpoint)

# load pretrained retriever
retriever_tokenizer = retriever_tokenizer_class.from_pretrained(
    args.retrieve_tokenizer_dir)
retriever_model = retriever_model_class.from_pretrained(
    args.retrieve_checkpoint, force_download=True)

model.retriever = retriever_model
# do not need and do not tune passage encoder
model.retriever.passage_encoder = None
model.retriever.passage_proj = None

args.reader_model_type = args.reader_model_type.lower()
reader_config_class, reader_model_class, reader_tokenizer_class = MODEL_CLASSES['reader']
reader_config = reader_config_class.from_pretrained(args.reader_config_name if args.reader_config_name else args.reader_model_name_or_path,
                                                    cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)
reader_config.num_qa_labels = 2
# this not used for BertForOrconvqaGlobal
reader_config.num_retrieval_labels = 2
reader_config.qa_loss_factor = args.qa_loss_factor
reader_config.retrieval_loss_factor = args.retrieval_loss_factor

reader_config.real_joint_learn = args.real_joint_learn
reader_config.involve_rerank_in_real_joint_learn = args.involve_rerank_in_real_joint_learn

reader_tokenizer = reader_tokenizer_class.from_pretrained(args.reader_tokenizer_name if args.reader_tokenizer_name else args.reader_model_name_or_path,
                                                          do_lower_case=args.do_lower_case,
                                                          cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)
reader_model = reader_model_class.from_pretrained(args.reader_model_name_or_path,
                                                  from_tf=bool(
                                                      '.ckpt' in args.reader_model_name_or_path),
                                                  config=reader_config,
                                                  cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)

model.reader = reader_model

if args.weak_supervision in ['learned', 'em+learned']:
    supervisor_config_class, supervisor_model_class, supervisor_tokenizer_class = MODEL_CLASSES[
        'supervisor']
    supervisor_config = supervisor_config_class.from_pretrained(
        args.supervisor_checkpoint)
    supervisor_tokenizer = supervisor_tokenizer_class.from_pretrained(
        str(Path(args.supervisor_checkpoint).parent))
    supervisor_model = supervisor_model_class.from_pretrained(
        args.supervisor_checkpoint)


if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

model.to(args.device)
if args.weak_supervision in ['learned', 'em+learned']:
    supervisor_model.to(args.supervisor_device)

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

logger.info(f'loading passage ids from {args.passage_ids_path}')
with open(args.passage_ids_path, 'rb') as handle:
    passage_ids = pkl.load(handle)

logger.info(f'loading passage reps from {args.passage_reps_path}')
with open(args.passage_reps_path, 'rb') as handle:
    passage_reps = pkl.load(handle)

logger.info('constructing passage faiss_index')
faiss_res = faiss.StandardGpuResources()
index = faiss.IndexFlatIP(args.proj_size)
index.add(passage_reps)
gpu_index = faiss.index_cpu_to_gpu(faiss_res, 1, index)

# logger.info(f'loading all blocks from {args.blocks_path}')
# with open(args.blocks_path, 'rb') as handle:
#     blocks_array = pkl.load(handle)


logger.info(f'loading qrels from {args.qrels}')
with open(args.qrels) as handle:
    qrels = json.load(handle)

passage_id_to_idx = {}
for i, pid in enumerate(passage_ids):
    passage_id_to_idx[pid] = i

qrels_data, qrels_row_idx, qrels_col_idx = [], [], []
# qid_to_idx = {}
qid_to_idx = defaultdict(lambda: 0)
for i, (qid, v) in enumerate(qrels.items()):
    qid_to_idx[qid] = i
    for pid in v.keys():
        qrels_data.append(1)
        qrels_row_idx.append(i)
        qrels_col_idx.append(passage_id_to_idx[pid])
qrels_sparse_matrix = sp.sparse.csr_matrix(
    (qrels_data, (qrels_row_idx, qrels_col_idx)))

evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank', 'recall'})

# used in training:
recall_evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recall'})
p1_evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'P.1'})

nlp = spacy.load("en_core_web_sm")

# when using em+learned, keep track of how many answers are found by em
em_answer_found_num = 0

if args.case_study:
    args.case_study_file = os.path.join(args.output_dir, "case_study.txt")


# In[16]:


# Training
if args.do_train:
    DatasetClass = RetrieverDataset
    train_dataset = DatasetClass(args.train_file, retriever_tokenizer,
                                 args.load_small, args.history_num,
                                 query_max_seq_length=args.retriever_query_max_seq_length,
                                 is_pretraining=args.is_pretraining,
                                 given_query=True,
                                 given_passage=False, 
                                 include_first_for_retriever=args.include_first_for_retriever)
    global_step, tr_loss = train(
        args, train_dataset, model, retriever_tokenizer, reader_tokenizer)
    logger.info(" global_step = %s, average loss = %s",
                global_step, tr_loss)

# Save the trained model and the tokenizer
if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    if not os.path.exists(args.retriever_tokenizer_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.retriever_tokenizer_dir)
    if not os.path.exists(args.reader_tokenizer_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.reader_tokenizer_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model
    final_checkpoint_output_dir = os.path.join(
        args.output_dir, 'checkpoint-{}'.format(global_step))
    final_retriever_model_dir = os.path.join(
        final_checkpoint_output_dir, 'retriever')
    final_reader_model_dir = os.path.join(
        final_checkpoint_output_dir, 'reader')
    if not os.path.exists(final_checkpoint_output_dir):
        os.makedirs(final_checkpoint_output_dir)
    if not os.path.exists(final_retriever_model_dir):
        os.makedirs(final_retriever_model_dir)
    if not os.path.exists(final_reader_model_dir):
        os.makedirs(final_reader_model_dir)

    retriever_model_to_save = model_to_save.retriever
    retriever_model_to_save.save_pretrained(
        final_retriever_model_dir)
    reader_model_to_save = model_to_save.reader
    reader_model_to_save.save_pretrained(final_reader_model_dir)

    retriever_tokenizer.save_pretrained(args.retriever_tokenizer_dir)
    reader_tokenizer.save_pretrained(args.reader_tokenizer_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(
        final_checkpoint_output_dir, 'training_args.bin'))

    # Load a trained model and vocabulary that you have fine-tuned
    model = Pipeline()

    model.retriever = retriever_model_class.from_pretrained(
        final_retriever_model_dir, force_download=True)
    model.retriever.passage_encoder = None
    model.retriever.passage_proj = None

    model.reader = reader_model_class.from_pretrained(
        final_reader_model_dir, force_download=True)

    retriever_tokenizer = retriever_tokenizer_class.from_pretrained(
        args.retriever_tokenizer_dir, do_lower_case=args.do_lower_case)
    reader_tokenizer = reader_tokenizer_class.from_pretrained(
        args.reader_tokenizer_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)


# In[17]:


# Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory

results = {}
max_f1 = 0.0
best_metrics = {}
if args.do_eval and args.local_rank in [-1, 0]:
    retriever_tokenizer = retriever_tokenizer_class.from_pretrained(
        args.retriever_tokenizer_dir, do_lower_case=args.do_lower_case)
    reader_tokenizer = reader_tokenizer_class.from_pretrained(
        args.reader_tokenizer_dir, do_lower_case=args.do_lower_case)
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
        checkpoints = sorted(list(os.path.dirname(os.path.dirname(c)) for c in
                                      glob.glob(args.output_dir + '/*/retriever/' + WEIGHTS_NAME, recursive=False)))
#         logging.getLogger("transformers.modeling_utils").setLevel(
#             logging.WARN)  # Reduce model loading logs

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        # Reload the model
        global_step = checkpoint.split(
            '-')[-1] if len(checkpoint) > 1 else ""
        print(global_step, 'global_step')
        model = Pipeline()
        model.retriever = retriever_model_class.from_pretrained(
            os.path.join(checkpoint, 'retriever'), force_download=True)
        model.retriever.passage_encoder = None
        model.retriever.passage_proj = None
        model.reader = reader_model_class.from_pretrained(
            os.path.join(checkpoint, 'reader'), force_download=True)
        model.to(args.device)

        # Evaluate
        result = evaluate(args, model, retriever_tokenizer,
                          reader_tokenizer, prefix=global_step)
        if result['f1'] > max_f1:
            max_f1 = result['f1']
            best_metrics = copy(result)
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

    all_results_file = os.path.join(
        args.output_dir, 'predictions', 'all_results.json')
    with open(all_results_file, 'w') as fout:
        json.dump(results, fout)

    logger.info("Results: {}".format(results))
    logger.info("best metrics: {}".format(best_metrics))


# In[18]:


if args.do_test and args.local_rank in [-1, 0]:    
    if args.do_eval:
        best_global_step = best_metrics['global_step'] 
    else:
        best_global_step = args.best_global_step
        retriever_tokenizer = retriever_tokenizer_class.from_pretrained(
            args.retriever_tokenizer_dir, do_lower_case=args.do_lower_case)
        reader_tokenizer = reader_tokenizer_class.from_pretrained(
            args.reader_tokenizer_dir, do_lower_case=args.do_lower_case)
    best_checkpoint = os.path.join(
        args.output_dir, 'checkpoint-{}'.format(best_global_step))
    logger.info("Test the best checkpoint: %s", best_checkpoint)

    model = Pipeline()
    model.retriever = retriever_model_class.from_pretrained(
        os.path.join(best_checkpoint, 'retriever'), force_download=True)
    model.retriever.passage_encoder = None
    model.retriever.passage_proj = None
    model.reader = reader_model_class.from_pretrained(
        os.path.join(best_checkpoint, 'reader'), force_download=True)
    model.to(args.device)

    # Evaluate
    result = evaluate(args, model, retriever_tokenizer,
                      reader_tokenizer, prefix='test')

    test_metrics_file = os.path.join(
        args.output_dir, 'predictions', 'test_metrics.json')
    with open(test_metrics_file, 'w') as fout:
        json.dump(result, fout)

    logger.info("Test Result: {}".format(result))


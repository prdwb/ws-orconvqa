# Weakly-Supervised Open-Retrieval Conversational Question Answering

This repo contains the code and data for our paper [Weakly-Supervised Open-Retrieval Conversational Question Answering]().  

### Data and checkpoints
Download [here](https://ciir.cs.umass.edu/downloads/ORConvQA/weak_supervision) (data will be uploaded soon). The data is distributed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license. We would like to thank the authors of [QuAC](http://quac.ai/) and [CoQA](https://stanfordnlp.github.io/coqa/) for making their data public so that we can build on them.

OR-QuAC/OR-CoQA files:  
* dev/test files in QuAC format. This is used in evaluation with the QuAC evaluation script.  
* Preprocessed train/dev/test files. Model input. Each line is an example.  
* Weak supervisor training data. For QuAC, we use a [diverse paraphrase model](https://github.com/martiansideofthemoon/style-transfer-paraphrasea) to generate paraphrases for the span answers to train the weak supervisor. For CoQA, we use the freeform answers provided by the dataset as natural paraphrases to the span answers.  

Checkpoints:
* Weak supervisor checkpoint.  
* Pipeline checkpoint (optional). This is not necessary to download if you want to train your own model.  
* Retriever checkpoint and collection files. Kindly refer to [here](https://github.com/prdwb/orconvqa-release) to download the pretrained retriever checkpoint and passage collection files.  

### Usage
Sample usage for CoQA. We use two GPUs for this part and one of the GPUs is for MIPS.
```
train_pipeline_weak_supervision.py
--train_file=coqa/preprocessed/train_no_cannotanswer.txt
--dev_file=coqa/preprocessed/dev_no_cannotanswer.txt
--test_file=coqa/preprocessed/test_no_cannotanswer.txt
--orig_dev_file=coqa/quac_format/dev_no_cannotanswer.txt
--orig_test_file=coqa/quac_format/test_no_cannotanswer.txt
--qrels=qrels.txt (this is a placeholder as we have no qrels for CoQA)
--blocks_path=all_blocks.txt
--passage_reps_path=passage_reps.pkl
--passage_ids_path=passage_ids.pkl
--output_dir=output_dir
--load_small=False
--history_num=6
--do_train=True
--do_eval=True
--do_test=True
--per_gpu_train_batch_size=2
--per_gpu_eval_batch_size=4
--learning_rate=5e-5
--max_steps=-1
--num_train_epochs=5.0
--logging_steps=5
--save_steps=5000
--overwrite_output_dir=False
--eval_all_checkpoints=True
--fp16=True
--retriever_cache_dir=path_to_huggingface_albert_v1_cache (optional)
--retrieve_checkpoint=path_to_retriever_checkpoint/checkpoint-xxx 
--retrieve_tokenizer_dir=path_to_retriever_checkpoint
--top_k_for_retriever=100
--use_retriever_prob=True
--reader_cache_dir=path_to_huggingface_bert_cache (optional)
--qa_loss_factor=1.0
--retrieval_loss_factor=0.0 (disable reranker)
--top_k_for_reader=5
--include_first_for_retriever=True
--real_joint_learn=False (a deprecated function, will be removed in a future version)
--involve_rerank_in_real_joint_learn=False (a deprecated function, will be removed in a future version)
--weak_supervision=em+learned
--use_rerank_prob=False (disable reranking)
--early_loss=True
--supervisor_checkpoint=path_to_weak_supervisor_checkpoint/checkpoint-xxx/
--drop_cannotanswer=True
--version_2_with_negative=False
--case_study=False
--max_answer_length=8
```


### Environment
* Install [Huggingface Transformers](https://github.com/huggingface/transformers), [Faiss](https://github.com/facebookresearch/faiss), and [pytrec_eval](https://github.com/cvangysel/pytrec_eval)
* Developed with Python 3.7, Torch 1.2.0, and Transformers 2.3.0


### Citation
```
@inproceedings{wsorconvqa,
  title={{Weakly-Supervised Open-Retrieval Conversational Question Answering}},
  author={Chen Qu and Liu Yang and Cen Chen and W. Bruce Croft and Kalpesh Krishna and Mohit Iyyer},
  booktitle={ECIR},
  year={2021}
}
```
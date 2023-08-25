import glob
import logging
import math
import json
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Any, Dict, Optional, Union, Tuple
# # 特殊注入use_reentrant 切换默认值
# from torch.utils.checkpoint import checkpoint as old_checkpoint
# def new_checkpoint(function, *args, use_reentrant: bool = False, **kwargs):
#     return old_checkpoint(
#         function,
#         *args,
#         use_reentrant=use_reentrant,
#         **kwargs,
#     )
# import torch.utils.checkpoint
# torch.utils.checkpoint.checkpoint = new_checkpoint

import torch
from torch import nn
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, IterableDataset, set_caching_enabled
from datasets.iterable_dataset import _BaseExamplesIterable, deepcopy, IterableDataset, Features, DatasetInfo
import torch.distributed as dist

set_caching_enabled(False)

import numpy as np
import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForTokenClassification,
    set_seed,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint, PredictionOutput, TrainOutput
# from collections import OrderedDict
# import evaluate
from typing import Iterator, List, Optional
import numpy as np
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class MyArguments:
    data_path: str = field(
        default="",
        metadata={"help": ("data_path")},
    )

    predict_output_path: str = field(
        default="",
        metadata={"help": ("predict_output_path")},
    )

    model_name: str = field(
        default="bigscience/bloom-560m",
        metadata={"help": ("The model checkpoint for weights initialization.Don't set if you want to train a model from scratch.")},
    )

    tokenizer_name: str = field(
        default="bigscience/bloom",
        metadata={"help": ("The model checkpoint for weights initialization.Don't set if you want to train a model from scratch.")},
    )

    train_max_len: int = field(default=1024, metadata={"help": "train_max_len"})

    gen_max_len: int = field(default=384, metadata={"help": "gen_max_len"})

    w_bit: int = field(default=4, metadata={"help": "weight quant bit, 16 is no quant"})

    use_kvcache: bool = field(default=False, metadata={"help": "use kvcache or not"})

    kvcache_bit: int = field(default=8, metadata={"help": "kvcache quant bits if use_kvcache opens"})

    kvcache_groupsize: int = field(default=128, metadata={"help": "kvcache quant groupsize if use_kvcache opens"})

    def __post_init__(self):
        pass


class MyTrainer(Seq2SeqTrainer):

    def __init__(self, my_args: MyArguments, args: Seq2SeqTrainingArguments, **kwargs):

        from transformers import LlamaTokenizer

        # 类型注释
        self.train_dataset: IterableDataset
        self.eval_dataset: IterableDataset
        self.args: Seq2SeqTrainingArguments

        tokenizer = LlamaTokenizer.from_pretrained(
            my_args.tokenizer_name,
            padding_side='left',
        )

        self.answer_start_ids = tokenizer("Helper: ", add_special_tokens=False).input_ids[:1]

        self.my_args = my_args

        def model_init():
            # from adapters.models.bloom.modeling_bloom import BloomForCausalLM
            from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoConfig
            from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, load_checkpoint_and_dispatch
            from awq.quantize.quantizer import real_quantize_model_weight
            from awq.quantize.pre_quant import apply_kvcache

            if my_args.w_bit == 16:
                model = LlamaForCausalLM.from_pretrained(
                    my_args.model_name,
                    torch_dtype=torch.bfloat16,
                    # adapter_size=my_args.adapter_size,
                )
            elif my_args.w_bit == 4:
                print("Loading pre-computed quantized weights...")
                config = AutoConfig.from_pretrained(my_args.model_name, trust_remote_code=True)
                with init_empty_weights():
                    model = AutoModelForCausalLM.from_pretrained(my_args.model_name, config=config, torch_dtype=torch.float16, trust_remote_code=True)
                from llama import load_quant
                # model = load_quant(my_args.model_name, '/root/workspace/external_data/pjllama13bv8/pjllama13bv8-gptq-w4-g64.bin', my_args.w_bit, groupsize=64, act_order=True)
                model = load_quant(my_args.model_name,
                                   '/root/workspace/external_data/pjllama13bv8/gptq-w4g64kv8g128-search-5.bin',
                                   my_args.w_bit,
                                   groupsize=64,
                                   act_order=False,
                                   use_kvcache=True,
                                   kvcache_bit=8,
                                   kvcache_groupsize=128)
                '''
                real_quantize_model_weight(
                    model, w_bit=my_args.w_bit, q_config=q_config, init_only=True)
                model = load_checkpoint_and_dispatch(
                    model, '/home/llm-awq/quant_cache/pjllama13bv7-w4-g128-weight-kvcache-32-debug.bin', device_map="balanced",
                    # TODO: can we remove this?
                    no_split_module_classes=[
                        "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"]
                )
                '''
            else:
                raise NotImplementedError("other wbit will support later.")

            # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
            # on a small vocab and want a smaller embedding size, remove this test.
            embedding_size = model.config.vocab_size
            logger.warning(("resize_token_embeddings", len(tokenizer), embedding_size))

            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))

            # 锁定fp以外其他参数
            # for name, p in model.named_parameters():
            #     if "adapter_" in name:
            #         p.requires_grad = True
            #     else:
            #         p.requires_grad = False

            if is_deepspeed_zero3_enabled():
                n_params = sum(dict((p.ds_id, p.ds_numel) for p in model.parameters()).values())
                trainable_n_params = sum(dict((p.ds_id, p.ds_numel) for p in model.parameters() if p.requires_grad).values())
            else:
                n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
                trainable_n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

            logger.info(f"Training new model from scratch - Trainable size={trainable_n_params/2**20:.2f}M params - Total size={n_params/2**20:.2f}M params")

            return model

        return super(MyTrainer, self).__init__(
            args=args,
            model=model_init(),
            tokenizer=tokenizer,
            # train_dataset=raw_train_datasets.shuffle(seed=args.seed, buffer_size=100000),
            # eval_dataset=raw_eval_dataset.take(my_args.max_eval_dataset_size),
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest"),
            **kwargs)

    # 生成映射
    def make_gen_map(self):
        tokenizer = self.tokenizer
        train_max_len = self.my_args.train_max_len
        gen_max_len = self.my_args.gen_max_len
        prompt = 'Instructions: You are PULSE, a large language model trained by OpenMedLab. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-06-28'
        answer_start_ids = self.answer_start_ids

        def gen_map(batch):
            batch_input_ids = []

            for question in batch['question']:
                input_ids = tokenizer(prompt, add_special_tokens=False).input_ids + [tokenizer.convert_tokens_to_ids("</s>")]

                input_ids += tokenizer("User: " + question, add_special_tokens=False).input_ids
                input_ids += [tokenizer.convert_tokens_to_ids("</s>")]
                input_ids += answer_start_ids

                batch_input_ids.append(input_ids[-(train_max_len - gen_max_len):])

            return {
                "input_ids": batch_input_ids,
            }

        return gen_map

    # 生成式预测
    # test_dataset 必须有dig 且最后一个对话为患者 ，生成下一句医生的回复
    def gen_predict(self, test_dataset: IterableDataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test", dry_run=False, **gen_kwargs) -> PredictionOutput:

        deal_test_dataset = test_dataset.map(
            self.make_gen_map(),
            batched=True,
            # num_proc=32,
            remove_columns=['type', "question", "reference_answer"],
            # desc="Running " + metric_key_prefix + " - predict_map",
        )

        #调整参数启动生成模式
        self.args.prediction_loss_only = False
        self.args.predict_with_generate = True

        if dry_run:
            return PredictionOutput(predictions=tuple(), label_ids=tuple(), metrics={})

        old_state = torch.random.get_rng_state()
        torch.manual_seed(self.args.seed)
        tmp_predict_output = super(MyTrainer, self).predict(
            test_dataset=deal_test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            # gen kargs
            max_length=self.my_args.train_max_len,
            num_beams=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.2,
            top_k=9,
            repetition_penalty=1.0,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("</s>"),
            **gen_kwargs)
        #还原
        torch.random.set_rng_state(old_state)

        tmp_prediction_texts = []

        for tmp_prediction, p_dataset_item in zip(tmp_predict_output.predictions, deal_test_dataset):
            p_dataset_item = p_dataset_item['input_ids']

            start_pos = 0
            while start_pos + len(p_dataset_item) < len(tmp_prediction):
                if np.all(tmp_prediction[start_pos:start_pos + len(p_dataset_item)] == p_dataset_item):
                    break
                start_pos += 1

            # 特殊情况
            if start_pos + len(p_dataset_item) >= len(tmp_prediction):
                tmp_prediction_texts.append("")
                continue

            new_start_pos = start_pos + len(p_dataset_item)
            new_end_pos = new_start_pos

            while new_end_pos < len(tmp_prediction) and tmp_prediction[new_end_pos] != self.tokenizer.convert_tokens_to_ids("</s>"):
                new_end_pos += 1

            # add answer_start_ids
            tmp_prediction_text = self.tokenizer.decode(self.answer_start_ids + tmp_prediction[new_start_pos:new_end_pos].tolist())

            if tmp_prediction_text[:8] == "Helper: ":
                tmp_prediction_text = tmp_prediction_text[8:]

            tmp_prediction_texts.append(tmp_prediction_text)

        # assert len(tmp_prediction_texts) == len(test_dataset)

        return PredictionOutput(predictions=tmp_prediction_texts, label_ids=None, metrics=tmp_predict_output.metrics)


def main():

    parser = HfArgumentParser((MyArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        tmp_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        tmp_args = parser.parse_args_into_dataclasses()

    my_args: MyArguments = tmp_args[0]
    training_args: Seq2SeqTrainingArguments = tmp_args[1]

    os.makedirs(training_args.logging_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(filename=os.path.join(training_args.logging_dir, "train.log"), encoding="utf8")],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.add_handler(logging.FileHandler(filename=os.path.join(training_args.logging_dir, "train.log"), encoding="utf8"))
    transformers.utils.logging.enable_explicit_format()

    # deepspeed logger
    from deepspeed.utils.logging import logger as deepspeed_logger
    deepspeed_logger.setLevel(log_level)
    deepspeed_logger.addHandler(logging.FileHandler(filename=os.path.join(training_args.logging_dir, "train.log"), encoding="utf8"))

    # Log on each process the small summary:
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
                   f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    logger.info("CUDA_VISIBLE_DEVICES = " + str(os.environ.get("CUDA_VISIBLE_DEVICES")))

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"My parameters {my_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                             "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our Trainer
    trainer = MyTrainer(
        my_args=my_args,
        args=training_args,
    )

    for test_file_path in sorted(glob.glob(os.path.join(my_args.data_path, "**/*.jsonl"), recursive=True)):
        predict_file_path = test_file_path.replace(my_args.data_path, my_args.predict_output_path)
        logger.info(f"run eval on {test_file_path}")
        logger.info(f"save eval on {predict_file_path}")

        if os.path.exists(predict_file_path) == True:
            logger.info(f"{predict_file_path} is finish, continue")
            continue

        test_dataset = load_dataset(
            "json",
            data_files=test_file_path,
            split="train",
        )

        predict_output = trainer.gen_predict(
            test_dataset=test_dataset,
            dry_run=False,
        )

        if trainer.is_world_process_zero():
            os.makedirs(os.path.dirname(predict_file_path), exist_ok=True)

            with open(predict_file_path, "w", encoding="utf8") as f:
                for test_dataset_item, predict_output_item in zip(test_dataset, predict_output.predictions):
                    f.write(
                        json.dumps(
                            {
                                "type": test_dataset_item["type"],
                                "question": test_dataset_item["question"],
                                "reference_answer": test_dataset_item["reference_answer"],
                                "predict_answer": predict_output_item.strip(),
                            },
                            ensure_ascii=False) + "\n")

        # dist.barrier()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

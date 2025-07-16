import lm_eval
from lm_eval.utils import setup_logging
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.models.utils import (
    stop_sequences_criteria
)

import torch
# from document_constrained_generation_causal_qwen import IndexBasedLogitsProcessor
from document_constrained_generation_causal_qwen_api import IndexBasedLogitsProcessor
from transformers import LogitsProcessorList
from index import FMIndex

setup_logging("DEBUG")

class FM_index_HFLM_api(HFLM):
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
                logits_processor=LogitsProcessorList([IndexBasedLogitsProcessor(
                    num_beams=1 if generation_kwargs.get("num_beams") == None else generation_kwargs.get("num_beams"),
                    min_new_tokens= 5 if generation_kwargs.get("min_new_tokens") == None else generation_kwargs.get("min_new_tokens"),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    force_decoding_from=None,
                    stop_at_count=0,
                    always_allow_eos=True,
                    forced_bos_token_id=None,
                    length_reward_factor=1,
                    end_marker=[32, 25]
                )]),
            )

class FM_index_HFLM(HFLM):
    def __init__(self, *args, index, **kwargs):
        super().__init__(*args, **kwargs)  # Pass everything through
        self.index = index

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
                logits_processor=LogitsProcessorList([IndexBasedLogitsProcessor(
                    num_beams=generation_kwargs.get("num_beams"),
                    min_new_tokens=generation_kwargs.get("min_new_tokens"),
                    index=self.index,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    force_decoding_from=None,
                    stop_at_count=0,
                    always_allow_eos=True,
                    forced_bos_token_id=None,
                    length_reward_factor=1,
                    end_marker=[32,25]
                )]),
            )

# index_dir = 'data/miniml/enwiki.qwen3_8b.fm_index'
# index = FMIndex.load(index_dir)

# initialize index
# find model file location
model_name = "Qwen/Qwen3-0.6B"
# lm_obj = FM_index_HFLM(pretrained=model_name, index=index)
# lm_obj = FM_index_HFLM_api(pretrained=model_name, device='mps')
lm_obj = FM_index_HFLM_api(pretrained=model_name, device='cpu')
# Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
# to include a set of tasks in a separate directory.
# task_manager = TaskManager()

# import inspect
# print(inspect.getfile(lm_eval.simple_evaluate))
results = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["nq_open"],
    num_fewshot=0
)

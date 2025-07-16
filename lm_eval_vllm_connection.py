import lm_eval
from lm_eval.utils import setup_logging
from lm_eval.models.vllm_causallms import VLLM, _vllm_mp_worker
from lm_eval.models.utils import undistribute

from typing import Optional, List
from more_itertools import distribute
import torch
# from document_constrained_generation_causal_qwen import IndexBasedLogitsProcessor
from document_constrained_generation_vllm_api_eval import IndexBasedLogitsProcessor
from transformers import LogitsProcessorList
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.utils import get_open_port
import ray
import os
import logging
from multiprocessing import Process, Queue
from queue import Empty


setup_logging("DEBUG")
eval_logger = logging.getLogger(__name__)


class FM_index_VLLM_api(VLLM):
    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        if generate:
            kwargs = self.modify_gen_kwargs(kwargs)
            sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop,
                                             logits_processors = [
                                                IndexBasedLogitsProcessor(
                                                    num_beams=1,
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    eos_token_id=self.tokenizer.eos_token_id,
                                                    stop_at_count=0,
                                                    always_allow_eos=True,
                                                    forced_bos_token_id=None,
                                                )
                                             ],
                                             **kwargs)
        else:
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
            )
        if self.data_parallel_size > 1 and not self.V1:
            # vLLM hangs if resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            @ray.remote
            def run_inference_one_model(
                model_args: dict,
                sampling_params: SamplingParams,
                requests: List[List[int]],
                lora_request: LoRARequest,
            ):
                llm = LLM(**model_args)
                return llm.generate(
                    prompt_token_ids=requests,
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
            inputs = (
                (self.model_args, sampling_params, req, self.lora_request)
                for req in requests
            )
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            return undistribute(results)
        elif self.data_parallel_size > 1:
            # based on https://github.com/vllm-project/vllm/blob/a04720bc36401d831cb048c3917b9e58173d9c1d/examples/offline_inference/data_parallel.py
            dp_size = self.data_parallel_size
            dp_master_ip = os.environ.get("VLLM_DP_MASTER_IP", "127.0.0.1")
            dp_master_port = os.environ.get("VLLM_DP_MASTER_PORT") or get_open_port()

            requests = (list(x) for x in distribute(self.data_parallel_size, requests))

            procs, resq = [], Queue()
            # We use Process as it is non-daemonic
            try:
                for rank, req in enumerate(requests):
                    proc = Process(
                        target=_vllm_mp_worker,
                        args=(
                            self.model_args.copy(),
                            sampling_params,
                            req,
                            self.lora_request,
                            resq,
                            dp_size,
                            rank,
                            dp_master_port,
                            dp_master_ip
                        ),
                    )
                    proc.start()
                    procs.append(proc)

                # Collect results
                rank_res = {}
                while len(rank_res) < len(procs):
                    try:
                        rank, result = resq.get(timeout=30)
                        if isinstance(result, dict) and "error" in result:
                            raise RuntimeError(result["error"])
                        rank_res[rank] = result
                    except Empty:
                        dead_procs = [
                            idx
                            for idx, p in enumerate(procs)
                            if not p.is_alive() and idx not in rank_res
                        ]
                        if dead_procs:
                            raise RuntimeError(
                                f"Worker processes {dead_procs} died unexpectedly"
                            )
                        continue

                results = [rank_res[i] for i in range(len(procs))]
                return undistribute(results)

            # cleanup
            finally:
                try:
                    resq.close()
                    resq.join_thread()
                except Exception:
                    eval_logger.debug(
                        "Failed to close vllm DP results queue", exc_info=True
                    )
                for proc in procs:
                    proc.join(timeout=10)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=5)
                        if proc.is_alive():
                            proc.kill()

        else:
            outputs = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request
            )
            return outputs



# model_name = "Qwen/Qwen3-0.6B"
model_name = "Qwen/Qwen3-8B"
# lm_obj = FM_index_VLLM_api(pretrained=model_name, device='cpu', max_model_len=29120)
lm_obj = FM_index_VLLM_api(pretrained=model_name,
                           # device='cuda',
                           # max_model_len=29120
                           )
results = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["nq_open"],
    log_samples=True,
    output_path=f"./results/{model_name}.json",
    num_fewshot=0,
    # limit=10
)

# include reasoning ? do I really need to wait for reasoning to be over?

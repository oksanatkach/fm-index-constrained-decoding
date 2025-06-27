from document_constrained_generation_vllm import IndexBasedLogitsProcessor
import vllm
from index import FMIndex

model_name = "meta-llama/Llama-3.2-1B-Instruct"

model = vllm.LLM(model=model_name, task="generate")

corpus = " ".join("""
They also were found to have perfectly coiffed hair, and wore what appeared to be Dior makeup.
“We were shocked to discover the unicorns,” said anthropologist Daniel St. Maurice. “They were
like nothing we had ever seen before. We had heard legends of the unicorns, but never thought
they actually existed.” When the scientists first arrived in the valley, the unicorns were
surprised and startled by the presence of humans, but were also excited. The unicorns welcomed
the researchers and explained that they had been waiting for them for a very long time. “The
unicorns said that they had been waiting for us for a very long time,” said Dr. St. Maurice.
“They said they had always known that humans would eventually discover them, but that they had
also always known that humans would be too stupid to realize the unicorns had been waiting for
them.”
""".split()).strip()

corpus = model.get_tokenizer()(' ' + corpus, add_special_tokens=False)['input_ids'] + [model.get_tokenizer().eos_token_id]
index = FMIndex()
index.initialize([corpus], in_memory=True)

prompt = "Paraphrase this sentence: The unicorns greeted the scientists, explaining that they had been expecting the encounter for a while."

output = model.generate(prompts=prompt,
                        sampling_params=vllm.SamplingParams(
                            logits_processors = [
                            # CiteFromPromptLogitsProcessor(model.get_tokenizer(), boost_factor=2.0),
                            IndexBasedLogitsProcessor(
                                    # num_beams=3,
                                    num_beams=1,
                                    index=index,
                                    pad_token_id=model.get_tokenizer().pad_token_id,
                                    eos_token_id=model.get_tokenizer().eos_token_id,
                                    force_decoding_from=None,
                                    stop_at_count=0,
                                    always_allow_eos=True,
                                    forced_bos_token_id=None,
                                )
                            ],
                            max_tokens=50,
                            min_tokens=1
                        ))
print(output[0].outputs[0].text)

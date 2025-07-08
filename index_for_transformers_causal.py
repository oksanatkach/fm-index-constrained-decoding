# from document_constrained_generation_causal import IndexBasedLogitsProcessor
from document_constrained_generation_causal_qwen import IndexBasedLogitsProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from logits_processor_zoo.transformers import CiteFromPromptLogitsProcessor
from index import FMIndex

# switch to causal LLMs
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.1-8B"
model_name = "Qwen/Qwen3-0.6B"
# model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

sentence = 'The unicorns greeted the scientists, explaining that they had been expecting the encounter for a while.'
prompt = f"Paraphrase this sentence: {sentence} Only reply with the resulting sentence. /no_think"
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
input_ids = tokenizer([text], return_tensors="pt").to(model.device)

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

corpus = tokenizer(' ' + corpus, add_special_tokens=False)['input_ids']# + [tokenizer.eos_token_id]
index = FMIndex()
index.initialize([corpus], in_memory=True)

# out_ids = model.generate(**input_ids, max_new_tokens=200, min_new_tokens=1, do_sample=False, num_beams=3,
out_ids = model.generate(**input_ids, max_new_tokens=50, min_new_tokens=10, do_sample=False, num_beams=2,
                              logits_processor=LogitsProcessorList([IndexBasedLogitsProcessor(
                                    num_beams=2,
                                    min_new_tokens=10,
                                    index=index,
                                    model_name=model_name,
                                    pad_token_id=model.config.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    force_decoding_from=None,
                                    stop_at_count=0,
                                    always_allow_eos=True,
                                    forced_bos_token_id=None,
                                    length_reward_factor=1
                                )]),
                              temperature=None, top_p=None, top_k=None)

gen_output = tokenizer.batch_decode(out_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
# print(out_ids)
print(gen_output)

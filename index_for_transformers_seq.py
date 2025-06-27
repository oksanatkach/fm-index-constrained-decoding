from document_constrained_generation_seq import IndexBasedLogitsProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessorList
from index import FMIndex

model_name = 'tuner007/pegasus_paraphrase'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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

prompt = 'The unicorns greeted the scientists, explaining that they had been expecting the encounter for a while.'
input_ids = tokenizer(prompt, return_tensors='pt')
out_ids = model.generate(**input_ids, max_new_tokens=50, min_new_tokens=1, do_sample=False, num_beams=1,
                              logits_processor=LogitsProcessorList([IndexBasedLogitsProcessor(
                                    num_beams=1,
                                    index=index,
                                    pad_token_id=model.config.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    force_decoding_from=None,
                                    stop_at_count=0,
                                    always_allow_eos=True,
                                    forced_bos_token_id=None,
                                )]),
                              temperature=None, top_p=None, top_k=None)

gen_output = tokenizer.batch_decode(out_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
print(gen_output)

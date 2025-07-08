# free decoding
# for each decoding step, record the logit of each token in the corpus
# record the highest logit at that step

from document_constrained_generation_causal import IndexBasedLogitsProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from logits_processor_zoo.transformers import CiteFromPromptLogitsProcessor
from index import FMIndex
import torch

import ast
logits = open('index_logits.txt', 'r').read()
logits = ast.literal_eval(logits)
logits = torch.tensor(logits)
# logits = logits.sum(dim=0)/logits.size(0)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


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

distinct = torch.LongTensor(index.occurring_distinct)
# print([tokenizer.decode(token) for token in distinct])
# print(distinct.shape)
# print(distinct)
# # mask[:, distinct] = self.BOOST

#################################

# sentence = 'The unicorns greeted the scientists, explaining that they had been expecting the encounter for a while.'
# prompt = f"Paraphrase this sentence: {sentence} Only reply with the resulting sentence."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True
# )
# input_ids = tokenizer([text], return_tensors="pt").to(model.device)
# out_ids = model.generate(**input_ids, max_new_tokens=50, min_new_tokens=1, do_sample=False, num_beams=1,
#                               temperature=None, top_p=None, top_k=None)
# gen_output = tokenizer.batch_decode(out_ids, skip_special_tokens=True,
#                                          clean_up_tokenization_spaces=False)
# print(out_ids)
# print(tokenizer.decode(out_ids[0][-21:], skip_special_tokens=True))



import html


def make_heatmap_html(generated_output, output_logits, corpus, logits, distinct, base_rgb=(0, 128, 0)):

    span_list = []
    for gen_ind in range(len(generated_output)):
        generated_text = tokenizer.decode(generated_output[:gen_ind+1], skip_special_tokens=True)
        gen_step_logits = logits[gen_ind]
        # 1) min-max-scale scores → [0, 1]
        lo, hi = min(gen_step_logits), max(gen_step_logits)
        span_list.append(f'<span>{generated_text}</span><br><br>')
        span_list.append(f'<span>Predicted token id: {generated_output[gen_ind]}</span><br><br>')
        span_list.append(f'<span>Predicted logit: {output_logits[gen_ind]}</span><br><br>')
        span_list.append(f'<span>Max token from corpus: {tokenizer.decode(distinct[gen_step_logits.argmax().item()])}</span><br><br>')
        span_list.append(f'<span>Max token id from corpus: {distinct[gen_step_logits.argmax().item()]}</span><br><br>')
        span_list.append(f'<span>Max logit from corpus: {gen_step_logits.max()}</span><br><br>')
        for token in corpus:
            if token in distinct:
                logit = logits[gen_ind][distinct.index(token)]
                alpha = 0 if hi == lo else (logit - lo) / (hi - lo)  # 0 … 1
                r, g, b = base_rgb
                style = f"background-color: rgba({r},{g},{b},{alpha:.2f}); padding:2px 3px;"
                token = tokenizer.decode(token)
                span = f"<span style='{style}'>{html.escape(token)}</span>"
                span_list.append(span)
        span_list.append('<br><span>#######################</span><br><br><br>')

    return " ".join(span_list)


generated_output =  [128000, 128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,
           2696,     25,   6790,    220,   2366,     18,    198,  15724,   2696,
             25,    220,    966,  12044,    220,   2366,     20,    271, 128009,
         128006,    882, 128007,    271,   4368,   1366,  10857,    420,  11914,
             25,    578,  88649,  44129,  44865,    279,  14248,     11,  26073,
            430,    814,   1047,   1027,  23132,    279,  13123,    369,    264,
           1418,     13,   8442,  10052,    449,    279,  13239,  11914,     13,
         128009, 128006,  78191, 128007,    271,    791,  88649,  44129,  44865,
            279,  14248,    449,    264,  22443,  16387,     11,    872,   6548,
          49025,    449,    264,  14392,   2840,    396,     13, 128009]
output_logits = [21.8234, 19.9609, 29.9056, 16.5310, 26.7365, 24.6051, 22.6145, 21.4905, 19.2065, 17.5379, 24.6125, 20.7494, 17.1723, 20.9897, 24.3174, 22.2543, 22.2325, 20.0974, 24.4756, 23.4660, 29.7652]
# print('####################')
# print(tokenizer.decode(generated_output[-21:], skip_special_tokens=True))

# logit_map = {distinct[ind].item(): logits[ind].item() for ind in range(len(distinct))}
# logit_map = {distinct[ind].item(): logits[11][ind].item() for ind in range(len(distinct))}

html_str = make_heatmap_html(generated_output[-21:-1], output_logits[:-1], corpus, logits, distinct.tolist())

# In a Jupyter notebook:
# from IPython.display import HTML
# HTML(html_str)

# Or, save to a file:
with open("token_heatmap.html", "w") as f:
    f.write("<body style='font-family:monospace'>" + html_str + "</body>")

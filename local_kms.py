from transformers import LongformerTokenizer, LongformerModel
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# 假设我们有一篇很长的文档，已经分成了多个段落
document_paragraphs = [
    "First long paragraph ...",
    "Second long paragraph ...",
    "Third long paragraph ...",
    # Assume there are more paragraphs
]

# 初始化BART分词器和模型用于摘要
summary_model_name = 'facebook/bart-large-cnn'
summary_tokenizer = BartTokenizer.from_pretrained(summary_model_name)
summary_model = BartForConditionalGeneration.from_pretrained(summary_model_name)

# 初始化Longformer分词器和模型用于处理长文档
longformer_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
longformer_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

# 函数来摘要每个段落
def summarize_paragraphs(paragraphs):
    summarized_paragraphs = []
    for paragraph in paragraphs:
        inputs = summary_tokenizer(paragraph, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = summary_model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
        summary = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summarized_paragraphs.append(summary)
    return summarized_paragraphs

# 函数来编码文档的各个段落
def encode_paragraphs_with_longformer(paragraphs):
    # 使用Longformer分词器将所有段落编码为一个大的输入序列
    inputs = longformer_tokenizer(paragraphs, return_tensors="pt", padding=True, truncation=True)
    # 获取模型的输出
    outputs = longformer_model(**inputs)
    return outputs.last_hidden_state

# 首先对每个段落进行摘要
summarized_paragraphs = summarize_paragraphs(document_paragraphs)

# 然后使用Longformer编码摘要后的段落
document_encoding = encode_paragraphs_with_longformer(summarized_paragraphs)

# 'document_encoding' 现在包含了整个文档的编码表示


import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

# 加载预训练的tokenizer和模型
cache_directory = "./cache"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir=cache_directory)
model = BertModel.from_pretrained('bert-base-uncased',cache_dir=cache_directory)

# 原始文本
text = "Hello, how are you?"

# 使用tokenizer进行标记化和编码
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# 添加特殊标记 [CLS] 和 [SEP]
token_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]

# 转换为PyTorch张量
token_tensor = torch.tensor(token_ids).unsqueeze(0)  # (batch_size=1, seq_length)

# 前向传播
outputs = model(token_tensor)

# 获取嵌入向量
embedding = outputs.last_hidden_state

# 可视化结果

# 打印标记序列
print("Tokens:", tokens)  # Tokens: ['hello', ',', 'how', 'are', 'you', '?']

# 打印标记ID
print("Token IDs:", token_ids)  # Token IDs: [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]

# 打印嵌入向量的形状
print("Embedding Shape:", embedding.shape)  # Embedding Shape: torch.Size([1, 8, 768])

# 取第一个样本的嵌入向量并转换为NumPy数组
embedding_array = embedding[0].detach().numpy()

# 可视化嵌入向量
plt.imshow(embedding_array, cmap='hot', interpolation='nearest')
plt.title("Embedding Visualization")
plt.colorbar()
plt.show()
plt.savefig('pic.png')

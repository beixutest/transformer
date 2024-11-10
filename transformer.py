import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import DataLoader, Dataset  
from transformers import BertTokenizer  
import math  
import pandas as pd

# 检查CUDA是否可用  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print("Using device:", device)  

# 初始化 tokenizer  
src_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  
tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  

# 数据集定义  
class TranslationDataset(Dataset):  
    def __init__(self, data, src_tokenizer, tgt_tokenizer):  
        self.data = data  
        self.src_tokenizer = src_tokenizer  
        self.tgt_tokenizer = tgt_tokenizer  

    def __len__(self):  
        return len(self.data)  

    def __getitem__(self, idx):  
        src, tgt = self.data[idx]  
        src_tokens = self.src_tokenizer.encode(src, add_special_tokens=True)  
        tgt_tokens = self.tgt_tokenizer.encode(tgt, add_special_tokens=True)  
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)  

# collate_fn 函数，用于批处理数据时进行填充  
def collate_fn(batch):  
    src_batch, tgt_batch = [], []  
    src_pad_idx = src_tokenizer.pad_token_id  
    tgt_pad_idx = tgt_tokenizer.pad_token_id  
    
    for src_item, tgt_item in batch:  
        src_batch.append(src_item)  
        tgt_batch.append(tgt_item)  
    
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=src_pad_idx, batch_first=True)  
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=tgt_pad_idx, batch_first=True)  
    
    return src_batch, tgt_batch  

# 数据准备  
data = [
("你好，同学。", "Hello, student."),
("你好，世界如此多姿多彩。", "Hello, the world is so colorful."),
("乌鲁木齐的天气今天真好。", "The weather in Urumqi is really nice today."),
("你好，新疆的朋友。", "Hello, friends from Xinjiang."),
("乌鲁木齐是一座美丽的城市。", "Urumqi is a beautiful city."),
("你好，很高兴见到你。", "Hello, it's nice to meet you."),
("乌鲁木齐的文化非常丰富。", "The culture of Urumqi is very rich."),
("你好，希望你今天过得愉快。", "Hello, I hope you have a good day."),
("乌鲁木齐的美食让人难忘。", "The food in Urumqi is unforgettable."),
("你好，欢迎来到我的家乡。", "Hello, welcome to my hometown."), 
("你好，乌鲁木齐。", "Hello, Urumqi.")
]  

# 创建数据集  
dataset = TranslationDataset(data, src_tokenizer, tgt_tokenizer)  

# 初始化 DataLoader  
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)  

# 位置编码类  
class PositionalEncoding(nn.Module):  
    def __init__(self, d_model, dropout=0.1, max_len=5000):  
        super(PositionalEncoding, self).__init__()  
        self.dropout = nn.Dropout(p=dropout)  

        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        
        # 计算位置编码  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  
        
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        
        pe = pe.unsqueeze(0)  # -> shape: (1, max_len, d_model)  
        self.register_buffer('pe', pe)  

    def forward(self, x):  
        seq_len = x.size(1)  
        x = x + self.pe[:, :seq_len, :]  
        return self.dropout(x)

    def get_positional_encoding(self, length):  
        # 获取指定长度的位置信息编码  
        return self.pe[0, :length, :].cpu().numpy()  # 确保从批次维度切片  

    def print_positional_encoding(self, length):  
        # 打印位置编码矩阵  
        pe_matrix = self.pe[0, :length, :].cpu().numpy()  # 确保从批次维度切片  
        df_pe = pd.DataFrame(pe_matrix)  
        print("位置编码矩阵:\n", df_pe)        


# Transformer 模型定义  
class TransformerModel(nn.Module):  
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):  
        super(TransformerModel, self).__init__()  
        self.d_model = d_model  
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)  
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)  
        self.positional_encoding = PositionalEncoding(d_model, dropout)  
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, batch_first=True)  
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)  

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask):  
        src = self.src_embedding(src) * math.sqrt(self.d_model)  
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)  
        src = self.positional_encoding(src)  
        tgt = self.positional_encoding(tgt)  
        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)  
        return self.fc_out(output)  

# 初始化模型  
src_vocab_size = src_tokenizer.vocab_size  
tgt_vocab_size = tgt_tokenizer.vocab_size  
model = TransformerModel(src_vocab_size, tgt_vocab_size).to(device)  

# 训练模型  
criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.pad_token_id)  
optimizer = optim.Adam(model.parameters(), lr=0.0001)  

# 创建掩码函数  
def create_mask(src, tgt):  
    src_seq_len = src.size(1)  
    tgt_seq_len = tgt.size(1)  
    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).type(torch.bool).to(device)  
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)  
    return src_mask, tgt_mask  

# 训练函数  
def train(model, data_loader, criterion, optimizer, num_epochs=100):  
    model.train()  
    for epoch in range(num_epochs):  
        for src, tgt in data_loader:  
            src, tgt = src.to(device), tgt.to(device)  
            tgt_input = tgt[:, :-1]  
            tgt_out = tgt[:, 1:]  

            src_mask, tgt_mask = create_mask(src, tgt_input)  
            logits = model(src, tgt_input, src_mask, tgt_mask, None)  
            optimizer.zero_grad()  

            logits = logits.view(-1, logits.shape[-1])  
            tgt_out = tgt_out.reshape(-1)  

            loss = criterion(logits, tgt_out)  
            loss.backward()  
            optimizer.step()  

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')  

# 开始训练  
train(model, data_loader, criterion, optimizer)  

# 翻译函数  
def translate(model, src_sentence, src_tokenizer, tgt_tokenizer):  
    model.eval()  

    # 打印分词和token ID  
    tokens = src_tokenizer.tokenize(src_sentence)  
    token_ids = src_tokenizer.encode(src_sentence, add_special_tokens=True)  
    print("Tokens:", tokens)  
    print("Token IDs:", token_ids)   
    
    # 分词和编码    
    src_tokens = src_tokenizer.encode(src_sentence, add_special_tokens=True)  
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)  
    tgt_tokens = [tgt_tokenizer.cls_token_id]  
    
    # 获取嵌入并进行缩放  
    with torch.no_grad():  
        src_embeddings = model.src_embedding(src_tensor)  
        # 打印原始嵌入  
        vectors = src_embeddings.squeeze(0).cpu().numpy()  
        df_raw_embedding = pd.DataFrame(vectors, index=[src_tokenizer.convert_ids_to_tokens(id) for id in src_tokens])  
        print("原始嵌入向量:\n", df_raw_embedding)   
        
        # 应用缩放  
        scaled_embeddings = src_embeddings * math.sqrt(model.d_model)  
        scaled_vectors = scaled_embeddings.squeeze(0).cpu().numpy()  
        df_scaled_embedding = pd.DataFrame(scaled_vectors, index=[src_tokenizer.convert_ids_to_tokens(id) for id in src_tokens])  
        print("缩放后的嵌入向量:\n", df_scaled_embedding)  
        
        # 打印位置编码矩阵  
        model.positional_encoding.print_positional_encoding(len(src_tokens))

        # 应用位置编码  
        src_with_pe = model.positional_encoding(scaled_embeddings)  
        combined_vectors = src_with_pe.squeeze(0).cpu().numpy()  
        df_combined = pd.DataFrame(combined_vectors, index=[src_tokenizer.convert_ids_to_tokens(id) for id in src_tokens])  
        print("带位置编码的源嵌入:\n", df_combined)
        

    # 翻译过程  
    max_translate_len = 50  
    for _ in range(max_translate_len):  
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)  
        src_mask, tgt_mask = create_mask(src_tensor, tgt_tensor)  
        with torch.no_grad():  
            output = model(src_tensor, tgt_tensor, src_mask, tgt_mask, None)  
        
        # 使用采样策略选择下一个token  
        next_token_prob = F.softmax(output[0, -1, :], dim=0).cpu()  
        next_token = torch.multinomial(next_token_prob, num_samples=1).item()  
        
        tgt_tokens.append(next_token)  

        if next_token == tgt_tokenizer.sep_token_id:  
            break  

    translation_tokens = tgt_tokenizer.convert_ids_to_tokens(tgt_tokens)  
    translation = tgt_tokenizer.convert_tokens_to_string(translation_tokens)  
    return translation   

# 执行翻译  
translation = translate(model, "你好，乌鲁木齐。", src_tokenizer, tgt_tokenizer)  
print("Translation:", translation)
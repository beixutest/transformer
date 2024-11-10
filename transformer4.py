import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import DataLoader, Dataset  
from transformers import BertTokenizer  
import math  
import pandas as pd  # 导入 pandas 库  

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
        position = torch.arange(0, max_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]  
        self.register_buffer('pe', pe)  

    def forward(self, x):  
        x = x + self.pe[:, :x.size(1), :].to(x.device)  
        return self.dropout(x)  

# Transformer 模型定义  
class TransformerModel(nn.Module):  
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,  
                 num_encoder_layers=6, num_decoder_layers=6,  
                 dim_feedforward=2048, dropout=0.1):  
        super(TransformerModel, self).__init__()  
        self.d_model = d_model  
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)  
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)  
        self.positional_encoding = PositionalEncoding(d_model, dropout)  
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers,  
                                          num_decoder_layers, dim_feedforward,  
                                          dropout, batch_first=True)  
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)  

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask):  
        src = self.src_embedding(src) * math.sqrt(self.d_model)  
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)  
        src = self.positional_encoding(src)  
        tgt = self.positional_encoding(tgt)  
        
        # 获取 Transformer 的输出  
        transformer_output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)  
        
        # 通过最终的线性层  
        output = self.fc_out(transformer_output)  
        
        return output  

# 初始化模型  
src_vocab_size = src_tokenizer.vocab_size  
tgt_vocab_size = tgt_tokenizer.vocab_size  
model = TransformerModel(src_vocab_size, tgt_vocab_size).to(device)  

# 训练模型  
criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_token_id)  
optimizer = optim.Adam(model.parameters(), lr=0.0001)  

# 创建掩码函数  
def create_mask(src, tgt):  
    src_seq_len = src.size(1)  
    tgt_seq_len = tgt.size(1)  
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)  
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)  
    return src_mask, tgt_mask  

# 训练函数  
def train(model, data_loader, criterion, optimizer, num_epochs=100):   
    model.train()  
    for epoch in range(num_epochs):  
        total_loss = 0  
        for src, tgt in data_loader:  
            src, tgt = src.to(device), tgt.to(device)  
            tgt_input = tgt[:, :-1]  
            tgt_out = tgt[:, 1:]  

            src_mask, tgt_mask = create_mask(src, tgt_input)  
            logits = model(src, tgt_input, src_mask, tgt_mask, None)  
            optimizer.zero_grad()  

            logits = logits.reshape(-1, logits.shape[-1])  
            tgt_out = tgt_out.reshape(-1)  

            loss = criterion(logits, tgt_out)  
            loss.backward()  
            optimizer.step()  
            total_loss += loss.item()  

        avg_loss = total_loss / len(data_loader)  
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')  

# 开始训练  
train(model, data_loader, criterion, optimizer)  

# 翻译函数  
def translate(model, src_sentence, src_tokenizer, tgt_tokenizer, max_len=50):  
    model.eval()  

    # 编码源句子  
    src_tokens = src_tokenizer.encode(src_sentence, add_special_tokens=True)  
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)  
    
    # 初始化目标序列  
    tgt_tokens = [tgt_tokenizer.cls_token_id]  
    
    # 用于存储每个时间步的前 5 个 token 及其概率  
    topk_tokens_per_step = []  

    for i in range(max_len):  
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)  
        src_mask, tgt_mask = create_mask(src_tensor, tgt_tensor)  
        with torch.no_grad():  
            output = model(src_tensor, tgt_tensor, src_mask, tgt_mask, None)  
        
        # 获取当前时间步的输出概率  
        logits = output[0, -1, :]  # 取出当前时间步的输出，形状为 [vocab_size]  
        probs = F.softmax(logits, dim=-1)  

        # 获取概率最高的前 5 个 token 及其概率  
        topk_probs, topk_indices = torch.topk(probs, k=5)  
        topk_probs = topk_probs.cpu().numpy()  
        topk_indices = topk_indices.cpu().numpy()  

        # 将 token ID 转换为对应的单词  
        topk_tokens = [tgt_tokenizer.convert_ids_to_tokens(int(idx)) for idx in topk_indices]  

        # 保存当前时间步的 top 5 tokens 及其概率  
        topk_tokens_per_step.append((topk_tokens, topk_probs))  

        # 选择概率最高的 token 作为下一个 token  
        next_token = topk_indices[0]  
        tgt_tokens.append(int(next_token))  

        # 如果生成了 [SEP] 标记，则停止  
        if next_token == tgt_tokenizer.sep_token_id:  
            break  

    # 将生成的 token 转换为字符串  
    translation = tgt_tokenizer.decode(tgt_tokens, skip_special_tokens=True)  
    print("Translation:", translation)  

    # 使用 pandas DataFrame 显示结果  
    # 初始化一个空的列表  
    data_rows = []  

    for step, (tokens, probs) in enumerate(topk_tokens_per_step):  
        step_dict = {'Step': step + 1}  
        for i, (token, prob) in enumerate(zip(tokens, probs)):  
            step_dict[f'Token_{i+1}'] = token  
            step_dict[f'Prob_{i+1}'] = round(float(prob), 4)  
        # 将 step_dict 添加到列表中  
        data_rows.append(step_dict)  

    # 循环结束后，创建 DataFrame  
    df = pd.DataFrame(data_rows)  

    # 设置显示选项，避免截断  
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.width', 1000)  

    print("\nDecoder Outputs (Top 5 tokens at each position):")  
    print(df)  

# 执行翻译并打印结果  
translate(model, "你好，乌鲁木齐。", src_tokenizer, tgt_tokenizer)
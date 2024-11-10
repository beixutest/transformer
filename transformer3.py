import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import DataLoader, Dataset  
from transformers import BertTokenizer  
import math  

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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)  

    def forward(self, x):  
        x = x + self.pe[:, :x.size(1)]  
        return self.dropout(x)  

# 自定义编码器层，用于插入打印函数  
class CustomEncoderLayer(nn.Module):  
    printed_first_add_norm = False  # 类变量，跟踪 Add & Norm 打印  
    printed_first_ffn = False  # 类变量，跟踪 FFN 打印  

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, print_flag=False):  
        super(CustomEncoderLayer, self).__init__()  
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)  
        self.linear1 = nn.Linear(d_model, dim_feedforward)  
        self.linear2 = nn.Linear(dim_feedforward, d_model)  

        self.dropout = nn.Dropout(dropout)  
        self.dropout1 = nn.Dropout(dropout)  
        self.dropout2 = nn.Dropout(dropout)  

        self.norm1 = nn.LayerNorm(d_model)  
        self.norm2 = nn.LayerNorm(d_model)  

        self.print_flag = print_flag  

    def forward(self, src, src_mask=None, src_key_padding_mask=None):  
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,  
                                  key_padding_mask=src_key_padding_mask)  
        src = src + self.dropout1(src2)  
        src = self.norm1(src)  

        if self.print_flag and not CustomEncoderLayer.printed_first_add_norm:  
            self.format_matrix(src, "Add & Norm")  
            CustomEncoderLayer.printed_first_add_norm = True  # 设置为已打印过  

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))  

        if self.print_flag and not CustomEncoderLayer.printed_first_ffn:  
            self.format_matrix(src2, " Feed Forward Network")  
            CustomEncoderLayer.printed_first_ffn = True  # 设置为已打印过  

        src = src + self.dropout2(src2)  
        src = self.norm2(src)  

        return src  

    def format_matrix(self, matrix, name):  
        matrix = matrix.detach().cpu()   
        for i in range(matrix.shape[0]):  # 遍历批次中的每个样本  
            for j in range(matrix.shape[1]):  # 遍历每个样本的所有时间步  
                row = matrix[i, j, :]  # 取每个样本的每个时间步  
                print(f"[{row[0].item():.2f}, {row[1].item():.2f}, ..., {row[-2].item():.2f}, {row[-1].item():.2f}]")  
        print(f"{name} matrix: Shape: {matrix.shape[1]}, {matrix.shape[2]}\n")


# 自定义编码器，将自定义的编码器层堆叠  
class CustomEncoder(nn.Module):  
    def __init__(self, encoder_layer, num_layers):  
        super(CustomEncoder, self).__init__()  
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])  
        self.num_layers = num_layers  

    def forward(self, src, mask=None, src_key_padding_mask=None):  
        output = src  

        for mod in self.layers:  
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)  

        return output  

# Transformer 模型定义  
class TransformerModel(nn.Module):  
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):  
        super(TransformerModel, self).__init__()  
        self.d_model = d_model  
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)  
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)  
        self.positional_encoding = PositionalEncoding(d_model, dropout)  

        # 使用自定义的编码器  
        encoder_layer = CustomEncoderLayer(d_model, nhead, dim_feedforward, dropout)  
        self.encoder = CustomEncoder(encoder_layer, num_encoder_layers)  

        # 解码器仍使用原始的Transformer解码器  
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)  
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)  

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)  

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):  
        src = self.src_embedding(src) * math.sqrt(self.d_model)  
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)  
        src = self.positional_encoding(src)  
        tgt = self.positional_encoding(tgt)  

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)  
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)  
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
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)  
    src_mask = None  
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
            src_key_padding_mask = (src == src_tokenizer.pad_token_id)  
            tgt_key_padding_mask = (tgt_input == tgt_tokenizer.pad_token_id)  
            optimizer.zero_grad()  

            logits = model(src, tgt_input, src_mask, tgt_mask, None, src_key_padding_mask, tgt_key_padding_mask, src_key_padding_mask)  

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

    # 重置打印标志  
    CustomEncoderLayer.printed_first_add_norm = False  
    CustomEncoderLayer.printed_first_ffn = False  

    # 启用打印  
    for layer in model.encoder.layers:  
        layer.print_flag = True  

    tokens = src_tokenizer.tokenize(src_sentence)  
    token_ids = src_tokenizer.encode(src_sentence, add_special_tokens=True)  
    print("Tokens:", tokens)  
    print("Token IDs:", token_ids)  
    
    src_tokens = src_tokenizer.encode(src_sentence, add_special_tokens=True)  
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)  
    tgt_tokens = [tgt_tokenizer.cls_token_id]  
    
    max_translate_len = 50  
    for _ in range(max_translate_len):  
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)  
        src_mask, tgt_mask = create_mask(src_tensor, tgt_tensor)  
        src_key_padding_mask = (src_tensor == src_tokenizer.pad_token_id)  
        tgt_key_padding_mask = (tgt_tensor == tgt_tokenizer.pad_token_id)  
        with torch.no_grad():  
            output = model(src_tensor, tgt_tensor, src_mask, tgt_mask, None, src_key_padding_mask, tgt_key_padding_mask, src_key_padding_mask)  
        
        next_token_prob = F.softmax(output[0, -1, :], dim=0).cpu()  
        next_token = torch.multinomial(next_token_prob, num_samples=1).item()  
        
        tgt_tokens.append(next_token)  

        if next_token == tgt_tokenizer.sep_token_id:  
            break  

    translation_tokens = tgt_tokenizer.convert_ids_to_tokens(tgt_tokens)  
    translation = tgt_tokenizer.convert_tokens_to_string(translation_tokens)  
    
    # 翻译后禁用打印  
    for layer in model.encoder.layers:  
        layer.print_flag = False  

    return translation  

# 执行翻译  
translation = translate(model, "你好，乌鲁木齐。", src_tokenizer, tgt_tokenizer)  
print("Translation:", translation)
import torch.nn.functional as F
import torch.nn as nn
import torch

#%%
class DeepNeo(nn.Module):
    def __init__(self):
        super(DeepNeo, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(5, 183), stride=1)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=10, kernel_size=(5, 183), stride=1)
        self.fc = nn.Linear(1*5*10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc(x))
        x = torch.sigmoid(x)
        return x
    
    def regularize(self, loss, device):
        l1_lambda = 0.0001
        l2_lambda = 0.001
        l2_reg = torch.tensor(0.).to(device)
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        loss += l2_lambda*l2_reg + l1_lambda*torch.norm(self.fc.weight, 1)
        return loss
#%%
class DeepNeo_2_Custom(nn.Module):
    def __init__(self):
        super(DeepNeo_2_Custom, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 133), stride=1)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=10, kernel_size=(8, 133), stride=1)
        self.fc = nn.Linear(1*5*10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc(x))
        x = torch.sigmoid(x)
        return x
    
    def regularize(self, loss, device):
        l1_lambda = 0.0001
        l2_lambda = 0.001
        l2_reg = torch.tensor(0.).to(device)
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        loss += l2_lambda*l2_reg + l1_lambda*torch.norm(self.fc.weight, 1)
        return loss
    
#%%
class HLA_Preproc(nn.Module):
    def __init__(self, hla_dim, epi_dim, nhead, dropout_rate):
        super(HLA_Preproc, self).__init__()
        self.hla_self_attn = nn.MultiheadAttention(embed_dim=hla_dim, num_heads=nhead, dropout=dropout_rate)
        self.hla_layer_norm = nn.LayerNorm(hla_dim)
        if hla_dim > epi_dim:
            self.proj = True
            self.hla_q_proj = nn.Linear(hla_dim, epi_dim)
            self.hla_k_proj = nn.Linear(hla_dim, epi_dim)
            self.hla_v_proj = nn.Linear(hla_dim, epi_dim)
        else:
            self.proj = False

    def forward(self, x_hla):
        # Input shape: (batch_size, seq_len, hla_dim)
        x_hla = x_hla.permute(1, 0, 2)  # (seq_len, batch_size, hla_dim)
        x_hla, _ = self.hla_self_attn(x_hla, x_hla, x_hla)
        x_hla = self.hla_layer_norm(x_hla)
        x_hla = x_hla.permute(1, 0, 2)  # (batch_size, seq_len, hla_dim)
        x_hla = x_hla.mean(dim=1)  # (batch_size, hla_dim)
        if self.proj:
            Q = self.hla_q_proj(x_hla).unsqueeze(0)  # (1, batch_size, epi_dim)
            K = self.hla_k_proj(x_hla).unsqueeze(0)
            V = self.hla_v_proj(x_hla).unsqueeze(0)
        else:
            Q = x_hla.unsqueeze(0)    # (1, batch_size, hla_dim)
            K = x_hla.unsqueeze(0)
            V = x_hla.unsqueeze(0)
        return Q, K, V

class Epi_Preproc(nn.Module):
    def __init__(self, hla_dim, epi_dim, nhead, dropout_rate):
        super(Epi_Preproc, self).__init__()
        self.epi_self_attn = nn.MultiheadAttention(embed_dim=epi_dim, num_heads=nhead, dropout=dropout_rate)

    def forward(self, x_epi):
        # Input shape: (batch_size, seq_len, epi_dim)
        x_epi = x_epi.permute(1, 0, 2)  # (seq_len, batch_size, epi_dim)
        x_epi, _ = self.epi_self_attn(x_epi, x_epi, x_epi)
        x_epi = x_epi.permute(1, 0, 2)  # (batch_size, seq_len, epi_dim)
        x_epi = x_epi.mean(dim=1)  # (batch_size, epi_dim)

        Q = x_epi.unsqueeze(0)     # (1, batch_size, epi_dim)
        K = x_epi.unsqueeze(0)
        V = x_epi.unsqueeze(0)
        return Q, K, V

class Cross_Attn_Demo(nn.Module):
    def __init__(self, hla_dim=1536, epi_dim=25, hla_nhead=8, epi_nhead=5, d_model=1536, dropout_rate=0.2):
        super(Cross_Attn_Demo, self).__init__()
        proj_dim = epi_dim
        self.hla_preprocessor = HLA_Preproc(hla_dim=hla_dim, epi_dim=epi_dim, nhead=hla_nhead, dropout_rate=dropout_rate)
        self.epi_preprocessor = Epi_Preproc(hla_dim=hla_dim, epi_dim=epi_dim, nhead=epi_nhead, dropout_rate=dropout_rate)
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=epi_nhead, dropout=dropout_rate)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=epi_nhead, dropout=dropout_rate)

        concat_dim = 2*proj_dim
        nhead = self.find_optimal_nhead(concat_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=concat_dim, num_heads=nhead, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(concat_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(concat_dim, 1)

    def forward(self, x_hla, x_epi):
        query1, key1, value1 = self.hla_preprocessor(x_hla)
        query2, key2, value2 = self.epi_preprocessor(x_epi)

        # Align dimensions for cross-attention
        cross_out_1, _ = self.cross_attn_1(query1, key2, value2)
        cross_out_2, _ = self.cross_attn_2(query2, key1, value1)
        
        # Concatenate outputs and apply self-attention
        x = torch.cat((cross_out_1, cross_out_2), dim=-1)  # (1, batch_size, concat_dim)
        x, _ = self.self_attn(x, x, x)
        x = self.layer_norm(x)
        x = x.squeeze(0)  # (batch_size, concat_dim)
        x = self.dropout(x)
        x = torch.sigmoid(self.output_layer(x))
        return x
    
    def find_optimal_nhead(self, concat_dim):
        for nhead in range(10, 0, -1):
            if concat_dim % nhead == 0:
                return nhead
        return 1
    
#%%
class Alex_Basic(nn.Module):
    def __init__(self, hla_dim=1280, epi_dim=1280, num_heads=4, num_layers=2, hidden_dim=512):
        super(Alex_Basic, self).__init__()
        input_dim = hla_dim + epi_dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 첫 번째 Linear Layer
            nn.BatchNorm1d(hidden_dim),         # Batch Normalization
            nn.ReLU(),                          # ReLU 활성화 함수
            nn.Dropout(0.5),                    # 50% 드롭아웃
    
            nn.Linear(hidden_dim, hidden_dim // 2), # 추가된 히든 레이어
            nn.BatchNorm1d(hidden_dim // 2),        # Batch Normalization
            nn.LeakyReLU(0.1),                      # LeakyReLU 활성화 함수
            nn.Dropout(0.5),                        # 50% 드롭아웃
    
            nn.Linear(hidden_dim // 2, 1),      # 출력 레이어
            nn.Sigmoid()                        # Sigmoid 활성화 함수
        )

    def forward(self, x_hla, x_epi):
        x_hla = x_hla.mean(dim=-2)              # HLA 시퀀스의 평균값
        x_epi = x_epi.mean(dim=-2)
        # print(x_hla.shape, x_epi.shape)
        x = torch.cat((x_hla, x_epi), dim=-1)   # HLA와 에피토프 시퀀스 연결
        x = self.transformer_encoder(x)         # 트랜스포머 인코더 통과
        x = self.classifier(x)                  # 분류기 통과
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIP_Text(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding               # 文本的嵌入层，将输入的文本 token 转换为向量表示
        self.positional_embedding = clip_model.positional_embedding     # 位置嵌入，用于为每个 token 添加位置信息
        self.transformer = clip_model.transformer                       # Transformer 编码器，用于处理序列信息
        self.ln_final = clip_model.ln_final                             # LayerNorm 层，用于对最终输出进行归一化
        self.text_projection = clip_model.text_projection               # 一个投影矩阵，用于将特征映射到目标向量空间
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).to(self.dtype)  # [batch_size, n_ctx, d_model]  n_ctx:文本序列的长度  d_model:每个 token 嵌入的维度

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot(End of Text) embedding (eot_token is the highest number in each sequence)
        # eot_token can be considered as the global representation of a sequence
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

import torch
import timm
import numpy as np
import math
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
#from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=256,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 num_hops = 3,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        #self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(num_hops, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        # self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head, drop_path=0.0) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        # (batch_size, 3, 32, 32) -> (batch_size, 192, 16, 16), 192: hidden, 16: image_size / patch_size
        #patches = self.patchify(img)
        # (batch_size, 192, 16, 16) -> (16*16, batch_size, 192), AX is here, (num_hops, batch_size, 192)
        #patches = rearrange(patches, 'b c h w -> (h w) b c')
        # pos emb size: (256, 1, 192), result size: (16*16, batch_size, 192)
        patches = rearrange(img, 'b h c -> h b c') # h -> num_hops
        #print(patches.shape)
        patches = patches + self.pos_embedding
        #patches = patches + self.pos_embedding
        # torch.Size([64, 2, 192]) torch.Size([256, 2]) torch.Size([256, 2])
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        #print(patches.shape, forward_indexes.shape, backward_indexes.shape)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        #print("concated patch size: ", patches.shape)

        patches = rearrange(patches, 't b c -> b t c')
        # t: number of tokens
        #print("rearrange concated patch size: ", patches.shape)
        features = self.layer_norm(self.transformer(patches))
        #print("features after transformer size: ", features.shape)

        features = rearrange(features, 'b t c -> t b c')
        #print("features rearrange size: ", features.shape)

        return features, backward_indexes
 

class MAE_Encoder_pretrain(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=256,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 num_hops = 3,
                 h_dim = 25,
                 t_dim =26,
                 r_dim = 1   
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        #self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(num_hops, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        # self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.h = torch.nn.Linear(h_dim, emb_dim)
        self.t = torch.nn.Linear(t_dim, emb_dim)
        self.r = torch.nn.Linear(r_dim, emb_dim)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head, drop_path=0.0) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, src_x, dst_x,  src_dst_x):
        # (batch_size, 3, 32, 32) -> (batch_size, 192, 16, 16), 192: hidden, 16: image_size / patch_size
        #patches = self.patchify(img)
        # (batch_size, 192, 16, 16) -> (16*16, batch_size, 192), AX is here, (num_hops, batch_size, 192)
        #patches = rearrange(patches, 'b c h w -> (h w) b c')
        # pos emb size: (256, 1, 192), result size: (16*16, batch_size, 192)
        h = self.h(src_x)
        t = self.t(dst_x)
        r = self.r(src_dst_x)

        h = h.unsqueeze(1) # bs x 1 x dim
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)

        conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim


        patches = rearrange(conv_input, 'b h c -> h b c') # h -> num_hops
        
        patches = patches + self.pos_embedding
        #patches = patches + self.pos_embedding
        # torch.Size([64, 2, 192]) torch.Size([256, 2]) torch.Size([256, 2])

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        #print("concated patch size: ", patches.shape)

        patches = rearrange(patches, 't b c -> b t c')
        # t: number of tokens
        #print("rearrange concated patch size: ", patches.shape)
        features = self.layer_norm(self.transformer(patches))
        #print("features after transformer size: ", features.shape)

        features = rearrange(features, 'b t c -> t b c')
        #print("features rearrange size: ", features.shape)

        return features, backward_indexes, conv_input     

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=256,
                 num_layer=4,
                 num_head=3,
                 num_hops = 3,                 
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        #self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(num_hops+1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head, drop_path=0.0) for _ in range(num_layer)])

        #self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.head = torch.nn.Linear(emb_dim, emb_dim)
        #self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)
        #(16*16) batchsize (3*2*2) -> batchsize 3 16*2 16*2
        self.patch2img = Rearrange('t b c -> b t c')


        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        #print(T)
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        #print(backward_indexes)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features) 
       
        mask = torch.zeros_like(patches)

        mask[T:] = 1

        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)


        return img, mask


class MAE_Decoder_pretrain(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=256,
                 num_layer=4,
                 num_head=3,
                 num_hops =3, 
                 h_dim = 25,
                 t_dim = 26,
                 r_dim = 1                                 
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        #self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(num_hops+1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head, drop_path=0.0) for _ in range(num_layer)])

        #self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.head = torch.nn.Linear(emb_dim, emb_dim)
        #self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)
        #(16*16) batchsize (3*2*2) -> batchsize 3 16*2 16*2
        self.patch2img = Rearrange('t b c -> b t c') 

        self.h = torch.nn.Linear(h_dim, emb_dim)
        self.t = torch.nn.Linear(t_dim, emb_dim)
        self.r = torch.nn.Linear(r_dim, emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        #print(T)
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        #print(backward_indexes)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features) #torch.Size([6->5, batch_size, feat_size])
       
        mask = torch.zeros_like(patches)
        #print(mask)
        mask[T-1:] = 1
        #print(mask.shape)
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)
        #print(mask)

        return img, mask

class MAE_Encoder_E2E(torch.nn.Module):
    def __init__(self,args ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, args.emb_dim))
        #self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros( args.num_hops, 1,  args.emb_dim))
        self.shuffle = PatchShuffle( args.mask_ratio)

        # self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.h = torch.nn.Linear( args.h_dim,  args.emb_dim)
        self.t = torch.nn.Linear( args.t_dim,  args.emb_dim)
        self.r = torch.nn.Linear( args.r_dim,  args.emb_dim)

        self.transformer = torch.nn.Sequential(*[Block( args.emb_dim,  args.num_head, drop_path=0.0) for _ in range( args.encoder_num_layer)])

        self.layer_norm = torch.nn.LayerNorm( args.emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, src_x, dst_x,  src_dst_x):
        # (batch_size, 3, 32, 32) -> (batch_size, 192, 16, 16), 192: hidden, 16: image_size / patch_size
        #patches = self.patchify(img)
        # (batch_size, 192, 16, 16) -> (16*16, batch_size, 192), AX is here, (num_hops, batch_size, 192)
        #patches = rearrange(patches, 'b c h w -> (h w) b c')
        # pos emb size: (256, 1, 192), result size: (16*16, batch_size, 192)
        h = self.h(src_x)
        t = self.t(dst_x)
        r = self.r(src_dst_x)

        h = h.unsqueeze(1) # bs x 1 x dim
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)

        conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim

        patches = rearrange(conv_input, 'b h c -> h b c') # h -> num_hops
        #print(patches.shape)
        patches = patches + self.pos_embedding
        #patches = patches + self.pos_embedding
        # torch.Size([64, 2, 192]) torch.Size([256, 2]) torch.Size([256, 2])
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        #print(patches.shape, forward_indexes.shape, backward_indexes.shape)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        #print("concated patch size: ", patches.shape)

        patches = rearrange(patches, 't b c -> b t c')
        # t: number of tokens
        #print("rearrange concated patch size: ", patches.shape)
        features = self.layer_norm(self.transformer(patches))
        #print("features after transformer size: ", features.shape)

        features = rearrange(features, 'b t c -> t b c')
        #print("features rearrange size: ", features.shape)

        return features, backward_indexes 

class MAE_Decoder_E2E(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, args.emb_dim))
        #self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(args.num_hops+1, 1, args.emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(args.emb_dim, args.num_head, drop_path=0.0) for _ in range(args.decoder_num_layer)])

        #self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.head = torch.nn.Linear(args.emb_dim, args.emb_dim)
        #self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)
        #(16*16) batchsize (3*2*2) -> batchsize 3 16*2 16*2
        

        self.h = torch.nn.Linear(args.emb_dim, args.h_dim)
        self.t = torch.nn.Linear(args.emb_dim, args.t_dim)
        self.r = torch.nn.Linear(args.emb_dim, args.r_dim)

        self.out_lins = torch.nn.ModuleList([self.h,self.t,self.r])


        self.total_output_dim = args.h_dim+args.t_dim+args.r_dim

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]-1
        B = features.shape[1]
        #print(T,B)
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        # print(backward_indexes)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        predicted_middle = self.head(features) #torch.Size([6->5, batch_size, feat_size])

        
        mask = torch.zeros([features.shape[0],B,1]).to(self.device)
        # print(mask)
        mask[T:] = 1
        
        mask = take_indexes(mask, backward_indexes[1:] - 1) #t b 1
        #print(mask)
        predicted_input_list = []
        for i in range(predicted_middle.shape[0]):
            predicted_input_list.append(self.out_lins[i](predicted_middle[i]))

        # predicted_input = torch.cat(predicted_input_list,dim=-1) #b total_dim

        return predicted_input_list,mask

class MAE_E2E(torch.nn.Module):
    def __init__(self,args) -> None:
        super().__init__()

        self.encoder = MAE_Encoder_E2E(args)
        self.decoder = MAE_Decoder_E2E(args)

    def forward(self, src_x,dst_x,src_dst_x):
        features, backward_indexes = self.encoder(src_x,dst_x,src_dst_x)
        predicted_input_list,mask = self.decoder(features,  backward_indexes)

        input_list = [src_x,dst_x,src_dst_x]
        
        loss = 0.0
        for i in range(len(input_list)):
            loss += torch.sum((predicted_input_list[i]-input_list[i]) ** 2 *mask[i])

        return loss

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 num_hops = 9,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio,num_hops=num_hops)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, num_hops=num_hops)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask

class MAE_ViT_pretrain(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=2,
                 encoder_head=3,
                 decoder_layer=1,
                 decoder_head=3,
                 mask_ratio=0.75,
                 num_hops = 9,
                 h_dim = 25,
                 r_dim = 1,
                 t_dim = 26
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder_pretrain(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio,num_hops=num_hops, h_dim=h_dim, r_dim=r_dim, t_dim=t_dim)
        self.decoder = MAE_Decoder_pretrain(image_size, patch_size, emb_dim, decoder_layer, decoder_head, num_hops=num_hops)

    def forward(self, src_x,dst_x,src_dst_x):
        features, backward_indexes, img = self.encoder(src_x,dst_x,src_dst_x)
        predicted_img, mask = self.decoder(features,  backward_indexes)

        return predicted_img, mask, img
        
class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        #patches = self.patchify(img)
        #patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = rearrange(img, 'b h c -> h b c')
        #print(patches.shape)
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.query = nn.Linear(dim, dim) # 输入768， 输出768
        self.key = nn.Linear(dim, dim) # 输入768， 输出768
        self.value = nn.Linear(dim, dim) # 输入768， 输出768

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dim = dim

    def forward(self, x): # x 维度是（L, 768）
        B, N, C = x.shape
        #print(B, N, C)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        x = torch.matmul(attention_probs, V)
        x = self.proj(x)
        x = self.proj_drop(x)
        #print(x.shape)

        return x

class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim) # 输入768， 输出768
        self.key = nn.Linear(dim, dim) # 输入768， 输出768
        self.value = nn.Linear(dim, dim) # 输入768， 输出768

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dim = dim

    def forward(self, x): # x 维度是（L, 768）
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #print(B, N, C)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        x = torch.matmul(attention_probs, V).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #print(x.shape)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention1(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    #print(b.shape)

    img = torch.rand(5, 3, 2)
    encoder = MAE_Encoder(emb_dim=2,mask_ratio=0.2)
    decoder = MAE_Decoder(emb_dim=2)
    features, backward_indexes = encoder(img)
    patches = rearrange(img, 'b h c -> h b c')
    print("patches size: ", patches.shape)
    patches = patches + encoder.pos_embedding
    patches = torch.cat([encoder.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
    patches = rearrange(patches, 't b c -> b t c')
    features = encoder.layer_norm(encoder.transformer(patches))
    features = rearrange(features, 'b t c -> t b c')
    print(features.shape)

    #print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print("predicted_img.shape: ", predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)

__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import to_dense_adj
import numpy as np

from collections import OrderedDict
from ..models.layers.pos_encoding import *
from ..models.layers.basics import *
from ..models.layers.attention import *

            
# Cell
class PatchTST(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int,
                 n_layers:int=3, ci=1, graph=0,d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "prediction", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'
        # Backbone
        self.backbone = PatchTSTEncoder(c_in, num_patch=num_patch, patch_len=patch_len,
                                n_layers=n_layers, ci=ci, graph=graph, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.n_vars = c_in
        self.head_type = head_type

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)


    def forward(self, z, hyperedge_index=None, device='cpu'):                             
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        hyperedge_index: tensor [2 * hype_edge_nums]
        """   
        z = self.backbone(z, hyperedge_index, device)                                                # z: [bs x nvars x d_model x num_patch]
        z = self.head(z)                                                                    
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                    # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x


class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len,
                 n_layers=3, ci=1, graph=0, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.ci = ci
        self.graph = graph
        self.d_model = d_model
        self.shared_embedding = shared_embedding        

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if self.ci:
            if not shared_embedding: 
                self.W_P = nn.ModuleList()
                for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
            else:
                self.W_P = nn.Linear(patch_len, d_model)    
        else:
            self.W_P = nn.Linear(c_in*patch_len, d_model)
            self.W_P_ = nn.Linear(d_model, c_in*d_model)  

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)
        
        # HGAT
        if self.graph:
            self.hgat = HGAT(num_patch*d_model,d_model,dropout)

    def forward(self, x, hyperedge_index=None, device='cpu') -> Tensor:          
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        hyperedge_index: tensor [2 * hype_edge_nums]
        """
        bs, num_patch, n_vars, patch_len = x.shape
        
        if self.ci:
            # channel independence
            # Input encoding
            if not self.shared_embedding:
                x_out = []
                for i in range(n_vars): 
                    z = self.W_P[i](x[:,:,i,:])
                    x_out.append(z)
                x = torch.stack(x_out, dim=2)
            else:
                x = self.W_P(x)                                                      # x: [bs x num_patch x nvars x d_model]
            x = x.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        

            # bs*n_vars == B*M : channel-independent
            u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
            u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]

            # Encoder
            z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
            z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]

            # HyperGraph
            if self.graph:
                x_out = []
                for i in range(n_vars):
                    x = self.hgat(z[:,i,:,:],hyperedge_index[i],device)              # x: [bs x num_patch x d_model]
                    x_out.append(x)
                z = torch.stack(x_out, dim=1)                                        # z: [bs x nvars x num_patch x d_model]

            z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x num_patch]
        else:
            # channel mixing
            # Input encoding
            x = torch.reshape(x, (x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))      # x: [bs x num_patch x nvars * patch_len]
            x = self.W_P(x)                                                          # x: [bs x num_patch x d_model]

            u = self.dropout(x + self.W_pos)                                         # u: [bs x num_patch x d_model]

            # Encoder
            z = self.encoder(u)                                                      # z: [bs x num_patch x d_model]

            # HyperGraph
            if self.graph:
                z = self.hgat(z, hyperedge_index, device)                            # z: [bs x num_patch x d_model]

            z = self.W_P_(z)                                                         # z: [bs x num_patch x nvars * d_model]
            z = torch.reshape(z, (z.shape[0],z.shape[1],n_vars,-1))                  # z: [bs x num_patch x nvars x d_model]
            z = z.permute(0,2,3,1)                                                   # z: [bs x nvars x d_model x num_patch]

        return z
    
    
# Cell
class HGAT(nn.Module):
    def __init__(self, d_in, d_model, dropout):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model

        self.linear1 = nn.Linear(d_in, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(d_model)
        
        self.hatt1 = HypergraphConv(d_model, d_model, use_attention=True, heads=4, concat=False, negative_slope=0.2, dropout=0.2, bias=True)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(d_model)

        self.hatt2 = HypergraphConv(d_model, d_model, use_attention=True, heads=1, concat=False, negative_slope=0.2, dropout=0.2, bias=True)
        self.dropout3 = nn.Dropout(dropout)
        self.bn3 = nn.BatchNorm1d(d_model)

        self.linear2 = nn.Linear(d_model, d_in)
        self.dropout4 = nn.Dropout(dropout)
        self.bn4 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))

    def forward(self, x, hyperedge_index, device):                      # x: [bs x patch_num x d_model] hyperedge_index: [2 x hyperedge_nums]
        src = x                                                                                   # src: [bs x patch_num x d_model]
        self.device = device

        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))                                 # x: [bs x patch_num * d_model]
        x = F.leaky_relu(self.linear1(x),0.2)                                                     
        x = self.dropout1(x)
        x = self.bn1(x)                                                                           # x: [bs x d_model]

        num_nodes = x.shape[0]
        num_edges = hyperedge_index[1].max().item() + 1

        a = to_dense_adj(hyperedge_index)[0].to(self.device)                                      # a: [bs x num_edges]
        if num_nodes > num_edges:
            a = a[:,:num_edges]
        else:
            a = a[:num_nodes]
        hyperedge_weight = torch.ones(num_edges).to(self.device)                                  # hyperedge_weight: [num_edges]
        hyperedge_attr = torch.matmul(a.T, x)                                                     # hyperedge_attr: [num_edges x d_model]

        x2 = self.hatt1(x, hyperedge_index, hyperedge_weight, hyperedge_attr)                     # x2: [bs x d_model]
        # Add & Norm
        x = x + self.dropout2(x2) # Add: residual connection with residual dropout
        x = self.bn2(x)

        hyperedge_attr = torch.matmul(a.T, x)                                                     # hyperedge_attr: [num_edges x d_model]
        x2 = self.hatt2(x, hyperedge_index, hyperedge_weight, hyperedge_attr)                     # x2: [bs x d_model]
        # Add & Norm
        x = x + self.dropout3(x2) # Add: residual connection with residual dropout
        x = self.bn3(x)

        x = F.leaky_relu(self.linear2(x), 0.2)                                                    # x: [bs x patch_num * d_model]
        x = torch.reshape(x, (x.shape[0], -1, self.d_model))                                      # x: [bs x patch_num x d_model]
        # Add & Norm
        x = src + self.dropout4(x) # Add: residual connection with residual dropout
        x = self.bn4(x)                                                                           # x: [bs x patch_num x d_model]
        return x



class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention
        self.store_attn = store_attn

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            if self.store_attn:
                print('save encoder attn')
                np.save('encoder_attn.npy',mod.attn.detach().cpu().numpy())
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




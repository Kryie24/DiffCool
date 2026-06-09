import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.nn.functional as F
import os
import pandas as pd

from torchvision.utils import make_grid
from tqdm import tqdm
import time

blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
)

blku = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(oc, oc, 2, stride=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
)

def generate_random_gaussian_heatmap(size=(64, 64), 
                                   min_sources=5, 
                                   max_sources=10,
                                   min_intensity=150,
                                   max_intensity=255,
                                   min_sigma=10,
                                   max_sigma=20,
                                   add_noise=True,
                                   noise_level=0.05):
    """
    生成带有随机高斯热源的热区图
    
    参数:
    - size: 热区图尺寸 (height, width)
    - min_sources: 最小热源数量
    - max_sources: 最大热源数量
    - min_intensity: 热源最小强度
    - max_intensity: 热源最大强度
    - min_sigma: 高斯分布最小标准差(控制热源扩散范围)
    - max_sigma: 高斯分布最大标准差
    - add_noise: 是否添加噪声
    - noise_level: 噪声水平
    
    返回:
    - heatmap: 生成的热区图数据
    """
    # 初始化热区图
    heatmap = np.zeros(size)
    
    # 随机确定热源数量
    num_sources = np.random.randint(min_sources, max_sources + 1)
    
    # 生成每个热源
    for _ in range(num_sources):
        # 随机热源位置
        x0 = np.random.randint(0, size[0])
        y0 = np.random.randint(0, size[1])
        
        # 随机强度
        intensity = min_intensity + (max_intensity - min_intensity) * np.random.rand()
        
        # 随机sigma值
        sigma = min_sigma + (max_sigma - min_sigma) * np.random.rand()
        
        # 创建网格
        x = np.arange(0, size[0], 1)
        y = np.arange(0, size[1], 1)
        xx, yy = np.meshgrid(x, y)
        
        # 添加高斯热源
        heatmap += intensity * np.exp(-((xx-x0)**2 + (yy-y0)**2)/(2*sigma**2))
    
    # 可选: 添加噪声
    if add_noise:
        heatmap += noise_level * np.random.randn(*size)
    
    # 确保没有负值
    heatmap = np.clip(heatmap, 0, None)
    
    return heatmap


############################# classes for the surrogate loss fn ##############################
class SimpleCentrosymmetricCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 单层中心对称卷积
        self.conv = CentrosymmetricConv2d(2, 1, kernel_size=5, padding=2)
        
        # 输出缩放参数
        self.scale = nn.Parameter(torch.tensor(1.0))

        self.dropout=nn.Dropout(0.5)
        
    def forward(self, binary_channel, int_channel):
        # 分离通道
        # binary_channel = x[:, 0:1, :, :]  # 0/1通道
        # int_channel = x[:, 1:2, :, :].float()  # 正整数通道
        
        # 对整数通道做归一化(保持比例)
        max_val = int_channel.max().clamp(min=1)  # 避免除以0
        int_channel = int_channel / max_val
        
        # 合并通道
        x_norm = torch.cat([binary_channel, int_channel], dim=1)
        
        # 卷积处理
        x = self.conv(x_norm)
        
        # x=self.dropout(x)

        x = F.relu(x) * self.scale * max_val  # 缩放回原范围
       
        return x

class CentrosymmetricConv2d(nn.Module):
    """严格中心对称的卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        assert kernel_size % 2 == 1, "内核大小必须是奇数"
        self.kernel_size = kernel_size
        self.padding = padding
        
        # 只初始化核的1/4部分(包括中心行列)
        half = (kernel_size + 1) // 2
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, half, half))
        self.bias = nn.Parameter(torch.randn(out_channels))
    
    def _build_kernel(self):
        """构建严格中心对称的完整卷积核"""
        k = self.kernel_size
        full_kernel = torch.zeros(self.weight.shape[0], self.weight.shape[1], k, k,
                                device=self.weight.device)
        
        half = (k + 1) // 2
        full_kernel[:, :, :half, :half] = self.weight  # 第一象限
        # 对称填充其他三个象限
        for i in range(half):
            for j in range(half):
                full_kernel[:, :, half-1+i, j]=full_kernel[:, :, half-1-i, j] 

                full_kernel[:, :, i, half-1+j]=full_kernel[:, :, i, half-1-j]

                full_kernel[:, :, half-1+i, half-1+j]=full_kernel[:, :, half-1-i, half-1-j]
        return full_kernel
    
    def forward(self, x):
        kernel = self._build_kernel()
        # import pdb; pdb.set_trace()
        return F.conv2d(x, kernel, bias=self.bias, padding=self.padding)
#########################################################################################




# this loss has no gradient
# def thermal_loss(CUR_LAYER_DESIGN, PREVIOUS_HEATMAP):
#     flatten_design=CUR_LAYER_DESIGN.flatten()
#     flatten_heatmap=PREVIOUS_HEATMAP.flatten()
#     np.savetxt('input_bool.txt', flatten_design, fmt='%d')  
#     np.savetxt('input_heatmap.txt', flatten_heatmap, fmt='%d')  
#     return_code = os.system('./therminator2 -d heatsink.xml -p heatsink.trace -o calloss')
#     new_heatmap_1d = np.loadtxt('output_heatmap.txt')
#     new_heatmap_2d = new_heatmap_1d.reshape(PREVIOUS_HEATMAP.shape[0],PREVIOUS_HEATMAP.shape[1])  

#     # std_previous=np.std(PREVIOUS_HEATMAP)
#     # std_new=np.std(new_heatmap_2d)

#     # np.max(PREVIOUS_HEATMAP)-np.max(new_heatmap_2d)
#     delta=PREVIOUS_HEATMAP-new_heatmap_2d
#     flux=np.sum(delta)

#     return -flux





class DummyX0Model(nn.Module):
    def __init__(self, n_channel: int, N: int = 16) -> None:
        super(DummyX0Model, self).__init__()
        self.down1 = blk(n_channel, 16)
        self.down2 = blk(16, 32)
        self.down3 = blk(32, 64)
        self.down4 = blk(64, 512)
        self.down5 = blk(512, 512)
        self.up1 = blku(512, 512)
        self.up2 = blku(512 + 512, 64)
        self.up3 = blku(64, 32)
        self.up4 = blku(32, 16)
        self.convlast = blk(16, 16)
        self.final = nn.Conv2d(16, N * n_channel, 1, bias=False)

        self.tr1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr3 = nn.TransformerEncoderLayer(d_model=64, nhead=8)

        # convolutional encoder for conditioning image
        self.cond_down1 = blk(n_channel, 16)
        self.cond_down2 = blk(16, 32)
        self.cond_down3 = blk(32, 64)
        self.cond_down4 = blk(64, 512)
        self.cond_down5 = blk(512, 512)

        self.temb_1 = nn.Linear(32, 16)
        self.temb_2 = nn.Linear(32, 32)
        self.temb_3 = nn.Linear(32, 64)
        self.temb_4 = nn.Linear(32, 512)
        self.N = N
        self.tau=1.0
        self.eps=1e-10

    def gumbel_hard_softmax(self, logits):
        # 生成 Gumbel 噪声
        U = torch.rand_like(logits)
        G = -torch.log(-torch.log(U + self.eps) + self.eps)  # Gumbel(0,1)

        # Gumbel-Softmax
        y = F.softmax((logits + G) / self.tau, dim=-1)

        y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(-1, keepdim=True), 1.0)
        # 反向：用 y 的梯度绕过 argmax
        y = (y_hard - y).detach() + y  # STE
        return y

    def forward(self, x, t, cond) -> torch.Tensor:
        x = (2 * x.float() / self.N) - 1.0
        cond = cond.float() * 2.0 - 1.0
        t = t.float().reshape(-1, 1) / 1000
        t_features = [torch.sin(t * 3.1415 * 2**i) for i in range(16)] + [
            torch.cos(t * 3.1415 * 2**i) for i in range(16)
        ]
        tx = torch.cat(t_features, dim=1).to(x.device)

        t_emb_1 = self.temb_1(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_2 = self.temb_2(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_3 = self.temb_3(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_4 = self.temb_4(tx).unsqueeze(-1).unsqueeze(-1)

        # encode conditioning image through convolutional tower
        cond1 = self.cond_down1(cond)
        cond2 = self.cond_down2(F.avg_pool2d(cond1, 2))
        cond3 = self.cond_down3(F.avg_pool2d(cond2, 2))
        cond4 = self.cond_down4(F.avg_pool2d(cond3, 2))
        cond5 = self.cond_down5(F.avg_pool2d(cond4, 2))

        x1 = self.down1(x) + t_emb_1 + cond1
        x2 = self.down2(F.avg_pool2d(x1, 2)) + t_emb_2 + cond2
        x3 = self.down3(F.avg_pool2d(x2, 2)) + t_emb_3 + cond3
        x4 = self.down4(F.avg_pool2d(x3, 2)) + t_emb_4 + cond4
        x5 = self.down5(F.avg_pool2d(x4, 2)) + cond5

        x5 = (
            self.tr1(x5.reshape(x5.shape[0], x5.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(x5.shape)
        )

        y = self.up1(x5) + cond4

        y = (
            self.tr2(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )

        y = self.up2(torch.cat([x4, y], dim=1)) + cond3

        y = (
            self.tr3(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )
        y = self.up3(y)
        y = self.up4(y)
        y = self.convlast(y)
        y = self.final(y)

        # reshape to B, C, H, W, N
        y = (
            y.reshape(y.shape[0], -1, self.N, *x.shape[2:])
            .transpose(2, -1)
            .contiguous()
        )

        # softmax
        y = self.gumbel_hard_softmax(y)

        return y


class D3PM(nn.Module):
    def __init__(
        self,
        x0_model: nn.Module,
        n_T: int,
        num_classes: int = 10,
        forward_type="uniform",
        hybrid_loss_coeff=0.001,
    ) -> None:
        super(D3PM, self).__init__()
        self.x0_model = x0_model
        self.forward_type = forward_type

        self.n_T = n_T
        self.hybrid_loss_coeff = hybrid_loss_coeff

        steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )

        # self.beta_t = [1 / (self.n_T - t + 1) for t in range(1, self.n_T + 1)]
        self.eps = 1e-6
        self.num_classses = num_classes
        q_onestep_mats = []
        q_mats = []  # these are cumulative
        # self.ADJ=torch.tensor(np.loadtxt('../data/ADJ.csv', delimiter=','),device=device).float()

        self.g_amb=100*0.000001
        self.g_al=237*0.000001/0.0005 # half the cell
        self.g_air=1*0.000001/0.0005 # half the cell
        self.T_amb=293.15 # should be 298.15

        for beta in self.beta_t:

            if forward_type == "uniform":
                mat = torch.ones(num_classes, num_classes) * beta / num_classes
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
                q_onestep_mats.append(mat)

            elif forward_type == "absorb":
                mat = torch.eye(num_classes)
                if num_classes >= 2:
                    mat[0, 0] = 1 - beta
                    mat[0, 1] = beta
                q_onestep_mats.append(mat)
            else:
                raise NotImplementedError
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

        q_one_step_transposed = q_one_step_mats.transpose(
            1, 2
        )  # this will be used for q_posterior_logits

        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.logit_type = "logit"

        # register
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (
            self.n_T,
            num_classes,
            num_classes,
        ), self.q_mats.shape

        self.surrogatemodel=torch.load('./surrogate.pth')

    def _at(self, a, t, x):
        # t is 1-d, x is integer value of 0 to num_classes - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        return a[t - 1, x, :]

    def q_posterior_logits(self, x_0, x_t, t):
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classses) + self.eps
            )
        else:
            x_0_logits = x_0.clone()

        assert x_0_logits.shape == x_t.shape + (self.num_classses,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        # Here, we caclulate equation (3) of the paper. Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.

        # fact1 is "guess of x_{t-1}" from x_t
        # fact2 is "guess of x_{t-1}" from x_0

        fact1 = self._at(self.q_one_step_transposed, t, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        # bs, num_classes, num_classes
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))

        bc = torch.where(t_broadcast == 1, x_0_logits, out)

        return bc

    def vb(self, dist1, dist2):

        # flatten dist1 and dist2
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1)
            - torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1).mean()

    def q_sample(self, x_0, t, noise):
        # forward process, x_0 is the clean input.
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def model_predict(self, x_0, t, cond):
        # this part exists because in general, manipulation of logits from model's logit
        # so they are in form of x_0's logit might be independent to model choice.
        # for example, you can convert 2 * N channel output of model output to logit via get_logits_from_logistic_pars
        # they introduce at appendix A.8.

        predicted_x0_logits = self.x0_model(x_0, t, cond)

        return predicted_x0_logits

    ## too much memory cost
    # def simplified_thermal_loss(self, design, heatmap):
    #     # input are batched 
    #     batch_size=design.shape[0]
    #     size=self.ADJ.shape[0]-1

    #     A_batch=[]
    #     b_batch=[]
    #     g_al_or_air_batch=[]
    #     # x [T_next T_inter T_before]
    #     for idx_linear in range(batch_size):

    #         print(idx_linear,3*size)
    #         A_tmp=torch.zeros(3*size,3*size,device=device)
    #         b_tmp=torch.zeros(3*size,device=device)
    #         g_al_or_air=torch.flatten(torch.mul(design[idx_linear,:,:,:,1],self.g_al)+torch.mul((1-design[idx_linear,:,:,:,0]),self.g_air)).to(device)
    #         g_al_or_air_batch.append(g_al_or_air)

    #         # first part
    #         A_tmp[:size,:size]=torch.matmul(torch.eye(size,device=device),g_al_or_air)+torch.mul(torch.eye(size,device=device),self.g_amb)
    #         A_tmp[:size,size:2*size]=torch.matmul(torch.mul(torch.eye(size,device=device),(-1)),g_al_or_air)

    #         b_tmp[:size]=torch.mul(torch.ones(size,device=device),self.T_amb*self.g_amb)

    #         # second part
    #         # thermal conductance
    #         up=torch.matmul(torch.eye(size,device=device),g_al_or_air)
    #         down=torch.matmul(torch.eye(size,device=device),g_al_or_air)
    #         out=torch.mul(self.ADJ[-1,:size],torch.mul(torch.ones(size,device=device),self.g_amb))
    #         intra_connection=torch.mul(self.ADJ[:size,:size],g_al_or_air.unsqueeze(1))+torch.matmul(self.ADJ[:size,:size],g_al_or_air)            
    #         intra=torch.matmul(torch.eye(size,device=device),intra_connection.sum(dim=1))-intra_connection

    #         A_tmp[size:2*size,:size]=torch.mul(-1,up)

    #         A_tmp[size:2*size,size:2*size]=torch.matmul(torch.eye(size,device=device),out)+up+down+intra

    #         A_tmp[size:2*size,2*size:3*size]=torch.mul(-1,down)

    #         b_tmp[size:2*size]=torch.mul(self.T_amb,out)

    #         # third part
    #         A_tmp[2*size:3*size,2*size:3*size]=torch.eye(size,device=device)
    #         b_tmp[2*size:3*size]=torch.flatten(heatmap[idx_linear])

    #         A_batch.append(A_tmp)
    #         b_batch.append(b_tmp)


    #     A_batch=torch.stack(A_batch)
    #     b_tmp=torch.stack(b_tmp)
    #     g_al_or_air_batch=torch.stack(g_al_or_air_batch)

    #     x_batch=torch.linalg.solve(A_batch, b_batch) # x_batch shape: B [T_next, T_intra, T_cur]

    #     losses=torch.mul(-1,torch.mul(g_al_or_air_batch,x_batch[:,:size])-torch.mul(torch.ones(size,device=device),self.T_amb)) # tensor broadcasting

    #     return losses


    def surrogate_loss(self, DESIGNS, HEATMAPS):
        # two parts: flow+hotspot_reduction
        self.surrogatemodel.eval()

        start=time.time()

        predicted_heatmap=self.surrogatemodel(DESIGNS, HEATMAPS) # here, all heatmap uses the delta temperature comparing to the room temperature
        
        inter_heatmap=(predicted_heatmap+HEATMAPS)/2

        ## 1st part, flow
        # neg_up_flow = -(predicted_heatmap)*self.g_amb
        # neg_side_flow = -inter_heatmap[:,:,:,0]*self.g_amb -inter_heatmap[:,:,:,-1]*self.g_amb -inter_heatmap[:,:,0,:]*self.g_amb -inter_heatmap[:,:,-1,:]*self.g_amb

        neg_up_flow = -(predicted_heatmap)
        neg_side_flow = -inter_heatmap[:,:,:,0] -inter_heatmap[:,:,:,-1] -inter_heatmap[:,:,0,:] -inter_heatmap[:,:,-1,:]

        flow_loss = neg_up_flow.sum(dim=(1,2,3)) + neg_side_flow.sum(dim=(1,2)) # -(heat flowing through the heatsink layer)

        ## 2nd part, hotspot_reduction

        hotspot_previous = HEATMAPS.amax(dim=(1,2,3))
        hotspot_now = predicted_heatmap.amax(dim=(1,2,3))

        hotspot_loss = hotspot_now
        # import pdb; pdb.set_trace()

        # standardize
        num_cells = DESIGNS.shape[2]*DESIGNS.shape[3] + 2*DESIGNS.shape[2] + 2*DESIGNS.shape[2] # for flow loss, the upper face and side faces
        flow_loss_std = (flow_loss)/hotspot_previous/num_cells

        hotspot_loss_std = hotspot_loss/hotspot_previous

        # import pdb; pdb.set_trace()
        losses = flow_loss_std + hotspot_loss_std

        end=time.time()
        print('surrogate loss time:',end-start)

        return losses.mean() # need to be scalar to backward()

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        Makes forward diffusion x_t from x_0, and tries to guess x_0 value from x_t using x0_model.
        x is one-hot of dim (bs, ...), with int values of 0 to num_classes - 1
        """
        # t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        # x_t = self.q_sample(
        #     x, t, torch.rand((*x.shape, self.num_classses), device=x.device)
        # )
        ############# needs a new way for x_t #################

        batchsize=x.shape[0]
        losses=[]
        x_t=[]
        cond_case=[]
        for idx_case in range(batchsize):
            idx_sample=torch.randint(0,self.n_T-1,(1,))
            # import pdb; pdb.set_trace()
            x_t.append(x[idx_case][idx_sample])
            cond_case.append(cond[idx_case][0].unsqueeze(0))
        x_t=torch.stack(x_t)
        cond_case=torch.stack(cond_case)



        # print(x_t.shape,cond_case.shape)

            # x_t is same shape as x
            # assert x_t.shape == x.shape, print(
            #     f"x_t.shape: {x_t.shape}, x.shape: {x.shape}"
            # )



        # is this the new x_(t-1)??????
        # this function requires batched data
        predicted_x0 = self.model_predict(x_t, idx_sample, cond_case) # cond contains all heatmaps for training efficiency, cond[0] is the initial heatmap, true cond
        # B C H W [0,1]


        ############ 7.11 needs a new thermal-related loss ###############
        ## we use hybrid loss.
        # predicted_x0_logits = self.model_predict(x_t, t, cond)

        ## based on this, we first do vb loss.
        # true_q_posterior_logits = self.q_posterior_logits(x, x_t, t)
        # pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t)

        # vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits)

        # predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        # x = x.flatten(start_dim=0, end_dim=-1)

        # ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        # return self.hybrid_loss_coeff * vb_loss + ce_loss, {
        #     "vb_loss": vb_loss.detach().item(),
        #     "ce_loss": ce_loss.detach().item(),
        # }
        #########################################################################
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        losses=self.surrogate_loss(predicted_x0[:,:,:,:,1], cond_case)

        return losses,[]

    def p_sample(self, x, t, cond, noise):

        predicted_x0_logits = self.model_predict(x, t, cond)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)

        noise = torch.clip(noise, self.eps, 1.0)

        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample

    def _init_image(self, shape, device):
        if self.forward_type == "absorb":
            return torch.ones(shape, dtype=torch.long, device=device)
        else:
            return torch.randint(0, self.num_classses, shape, device=device)

    def sample(self, x=None, cond=None):
        if x is None:
            if cond is None:
                raise ValueError("cond must be provided when x is None")
            x = self._init_image(cond.shape, cond.device)
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device)
            )

        return x

    def sample_with_image_sequence(self, x=None, cond=None, stride=10):
        if x is None:
            if cond is None:
                raise ValueError("cond must be provided when x is None")
            x = self._init_image(cond.shape, cond.device)
        steps = 0
        images = []
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device)
            )
            steps += 1
            if steps % stride == 0:
                images.append(x)
                
            # import pdb; pdb.set_trace()
            
        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images


if __name__ == "__main__":

    n_epoch = 1000
    num_data = 1000
    # device = "cuda:1"
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Num_batch=16
    l_r=0.00001
    layer_design=25
    
    
    N = 2  # number of classes for discretized state per pixel
    d3pm = D3PM(DummyX0Model(1, N), layer_design, num_classes=N, hybrid_loss_coeff=0.0, forward_type='absorb').to(device=device)


    logs={"batch_size":Num_batch, "learning_rate": l_r, "epoch": [], "loss":[], "norm": [], "param_norm":[]}
    for idx_folder in range(100):
        if not os.path.exists("./trained_models/"+str(idx_folder)):
            os.makedirs("./trained_models/"+str(idx_folder))
            break


    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")
    # dataset = MNIST(
    #     "./data",
    #     train=True,
    #     download=True,
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize(128),
    #             transforms.ToTensor(),
    #         ]
    #     ),
    # )

    ############################ load data ################################
    design=[]
    heatmap=[]

    for idx in range(num_data):
        tmp_design=[]
        tmp_heatmap=[]
        for j in range(layer_design):
            # each 0.csv design is full 1 
            tmp_design.append(np.loadtxt('../data/design/'+str(idx)+'/'+str(j)+'.csv',delimiter=','))
            tmp_heatmap.append(np.loadtxt('../data/heatmap/'+str(idx)+'/'+str(j)+'.csv',delimiter=','))
        design.append(tmp_design)
        heatmap.append(tmp_heatmap)
        
    # import pdb; pdb.set_trace()

    dataset=TensorDataset(torch.tensor(np.array(design)).float(), torch.tensor(np.array(heatmap)).float())
    ########################################################################

    dataloader = DataLoader(dataset, batch_size=Num_batch, shuffle=True, num_workers=8)

    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=l_r)
    d3pm.train()


    global_step = 0
    best_loss=999999999999
    for i in range(n_epoch):
        # t1=time.time()

        pbar = tqdm(dataloader)
        loss_ema = None
        # for x, _ in pbar:  B CC H W   
        #     optim.zero_grad()
        #     x = x.to(device)
        #     cond = torch.rand_like(x)

        #     # discritize x to N bins
        #     x = (x * (N - 1)).round().long().clamp(0, N - 1)

    ######################### for the updated dataset ######################
        for sample, condition in pbar:
            optim.zero_grad()
            x=sample.to(device)
            cond=condition.to(device)
            # import pdb; pdb.set_trace()
            # x = sample[0].to(device)
            # cond = sample[1].to(device)

            # # discritize x to N bins
            # x = (x * (N - 1)).round().long().clamp(0, N - 1)
    ########################################################################








            loss, info = d3pm(x, cond)  # loss is acquired here




            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 0.1)

            with torch.no_grad():
                param_norm = sum([torch.norm(p) for p in d3pm.x0_model.parameters()])

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()


                
            pbar.set_description(
                # f"loss: {loss_ema:.4f}, norm: {norm:.4f}, param_norm: {param_norm:.4f}, vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}"
                f"loss: {loss_ema:.4f}, norm: {norm:.4f}, param_norm: {param_norm:.4f}"
            )
            logs["epoch"].append(i)
            logs["loss"].append(loss_ema)            
            logs["norm"].append(norm.item())    
            logs["param_norm"].append(param_norm.item())    
            pd.DataFrame(logs).to_csv("./trained_models/"+str(idx_folder)+"/training_log.csv", index=False)
    

            optim.step()
            global_step += 1

            if global_step % 1000 == 1:
                d3pm.eval()
                if loss_ema<=best_loss:
                    torch.save(d3pm,"./trained_models/"+str(idx_folder)+"/"+str(int(i))+".pth")
                    best_loss=loss_ema

                with torch.no_grad():
                    # use random noise for conditioning during evaluation
                    # cond_eval = torch.rand(4, 1, 128, 128, device=device)
                    
                    # cond_eval = torch.rand(4, 1, 128, 128, device=device)*300 # using hotspot=300 K as testcase
                    cond_eval = []
                    for _ in range(4):
                        cond_eval.append([generate_random_gaussian_heatmap(size=(64, 64), 
                                                min_sources=4, 
                                                max_sources=12,
                                                min_intensity=8,
                                                max_intensity=32,
                                                min_sigma=8,
                                                max_sigma=20,
                                                add_noise=True,
                                                noise_level=0.05)])
                        
                    cond_eval=torch.tensor(np.array(cond_eval),device=device).float()

                    images = d3pm.sample_with_image_sequence(
                        cond=cond_eval, stride=5
                    )
                    # image sequences to gif
                    gif = []
                    for image in images:
                        x_as_image = make_grid(image.float() / (N - 1), nrow=2)
                        img = x_as_image.permute(1, 2, 0).cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        gif.append(Image.fromarray(img))

                    gif[0].save(
                        f"./trained_models/"+str(idx_folder)+"/sample_heatsink.gif",
                        save_all=True,
                        append_images=gif[1:],
                        duration=100,
                        loop=0,
                    )

                    last_img = gif[-1]
                    last_img.save(f"./trained_models/"+str(idx_folder)+"/sample_heatsink_last.png")

                d3pm.train()
        # t2=time.time()
        # print(t2-t1)

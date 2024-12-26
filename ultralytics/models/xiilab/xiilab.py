import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import time
import random 
import numpy as np 

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DetectionModel

from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)

# from src.core import register
import cv2
import os 
__all__ = ['RTDETR', ]
import torch

def generate_overlap_bbox_mask(batch_size, height, width, bboxes, original_size, device=torch.device("cuda")):
    """
    Args:
        batch_size (int): 배치 크기.
        height (int): 출력 마스크 높이 (새로운 해상도).
        width (int): 출력 마스크 너비 (새로운 해상도).
        bboxes (list of list): 각 이미지에 대한 바운딩박스 리스트.
                               예: [[(cx1, cy1, w1, h1), (cx2, cy2, w2, h2)], ...]
                               좌표는 [0, 1] 사이의 값 (정규화된 값).
        original_size (tuple): 원본 이미지 크기 (width, height).
        device (torch.device): 마스크가 생성될 디바이스.

    Returns:
        torch.Tensor: 배치 마스크 텐서. (B, 1, H, W)
                      바운딩박스 영역은 1, 바운딩박스 외부는 0.5로 구성.
    """
    orig_w, orig_h = original_size

    # 원본 이미지 크기와 새로운 이미지 크기 비율 계산
    x_scale = width / orig_w
    y_scale = height / orig_h

    # 초기 마스크 생성 (배경 값 0.5)
    mask = torch.ones((batch_size, 1, height, width), device=device) * 0.5

    for i, image_bboxes in enumerate(bboxes):
        for bbox in image_bboxes:
            cx, cy, w, h = bbox

            # 중심 좌표와 크기를 사용하여 좌상단 및 우하단 좌표 계산
            x_min = int((cx - w / 2) * orig_w)
            x_max = int((cx + w / 2) * orig_w)
            y_min = int((cy - h / 2) * orig_h)
            y_max = int((cy + h / 2) * orig_h)

            # 변환된 좌표를 새로운 이미지 크기에 맞게 조정
            x_min_new = int(x_min * x_scale)
            x_max_new = int(x_max * x_scale)
            y_min_new = int(y_min * y_scale)
            y_max_new = int(y_max * y_scale)

            # 바운딩박스 영역을 1로 설정
            mask[i, 0, y_min_new:y_max_new, x_min_new:x_max_new] = 1

    return mask


class SelfAttention(nn.Module):
    def __init__(self, unified_channels, num_heads=8):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(unified_channels, unified_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(unified_channels, unified_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(unified_channels, unified_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.size()
        
        query = self.query_conv(x).view(batch, self.num_heads, channels // self.num_heads, height * width)
        key = self.key_conv(x).view(batch, self.num_heads, channels // self.num_heads, height * width)
        value = self.value_conv(x).view(batch, self.num_heads, channels // self.num_heads, height * width)
        
        key = key.permute(0, 1, 3, 2)
        
        attention = torch.matmul(query, key) / (channels // self.num_heads) ** 0.5
        attention = self.softmax(attention)
        
        out = torch.matmul(attention, value)
        out = out.view(batch, channels, height, width)        
        # return out + x
        return out

 



class Generator(nn.Module):    
    def __init__(self, input_channels1, input_channels2, input_channels3, output_channels=1, unified_channels=128, target_size=(320, 320), num_heads=8):
        """
        마스크 생성 모듈의 생성자
        Args:
            input_channels1 (int): Low Feature Map의 채널 사이즈(ex)512)
            input_channels2 (int): Mid Feature Map의 채널 사이즈(ex)1024)
            input_channels3 (int): High Feature Map의 채널 사이즈(ex)2048)
            unified_channels : 기준이 되는 채널 수(128 사용)
            target_size : 마스크 사이즈 == 이미지 사이즈동일하게 부여
            num_heads : 셀프어텐션 num_head (8 사용)
        Returns:
            -               
        """
        super(Generator, self).__init__()
        
        self.target_size = target_size
        # 해상도 통합 레이어
        self.resize1 = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
        self.resize2 = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
        self.resize3 = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
        
        # 채널 정렬 레이어
        self.align1_1 = nn.Conv2d(input_channels1, unified_channels, kernel_size=1)
        self.align2_1 = nn.Conv2d(input_channels2, unified_channels, kernel_size=1)
        self.align3_1 = nn.Conv2d(input_channels3, unified_channels, kernel_size=1)
        
        # 채널 정렬 레이어
        self.align1_2 = nn.Conv2d(unified_channels, int(unified_channels/2), kernel_size=1)
        self.align2_2 = nn.Conv2d(unified_channels, int(unified_channels/2), kernel_size=1)
        self.align3_2 = nn.Conv2d(unified_channels, int(unified_channels/2), kernel_size=1)
              
        
        # Self-Attention 레이어
        self.attention1 = SelfAttention(512, num_heads=num_heads)
        self.attention2 = SelfAttention(1024, num_heads=num_heads)
        self.attention3 = SelfAttention(2048, num_heads=num_heads)
        
        # 채널 정렬 레이어
        self.align1_3 = nn.Conv2d(int(unified_channels/2), int(unified_channels/4), kernel_size=1)
        self.align2_3 = nn.Conv2d(int(unified_channels/2), int(unified_channels/4), kernel_size=1)
        self.align3_3 = nn.Conv2d(int(unified_channels/2), int(unified_channels/4), kernel_size=1)
        
        
        # Residual 연결을 위한 Conv
        self.residual_conv = nn.Conv2d(int(unified_channels/4 * 3), int(unified_channels/4 * 3), kernel_size=3, padding=1)
        # 배치노멀라이즈 추가
        self.residual_bn = nn.BatchNorm2d(int(unified_channels/4 * 3))  # BatchNorm 추가
        # 결합 레이어
        self.combine_conv = nn.Conv2d(int(unified_channels/4 * 3), int(unified_channels/4), kernel_size=1)
        
        # Refinement Network
        self.refinement1 = nn.Conv2d(int(unified_channels/4), int(unified_channels/4/2), kernel_size=3, padding=1)
        self.refinement2 = nn.Conv2d(int(unified_channels/4/2), int(unified_channels/4/2/2), kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(int(unified_channels/4/2/2), output_channels, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=False)
        self.activation = nn.Sigmoid()



    def forward(self, x1, x2, x3,targets): 
        """
        마스크 생성 모듈의 Forward 함수
        Args:
            x1 : Low Feature Map
            x2 : Mid Feature Map
            x3 : High Feature Map
            targets : 정답지(바운딩박스) - 일단 배제하고 진행해도됨
        Returns:
            torch.Tensor: 배치 마스크 텐서. (B, 1, H, W)                  
        """       
        x1 = self.attention1(x1)
        x2 = self.attention2(x2)
        x3 = self.attention3(x3)      
        
        # 해상도 정렬
        x1 = self.resize1(x1)
        x2 = self.resize2(x2)
        x3 = self.resize3(x3)        
        
        # 채널 정렬
        x1 = self.align1_1(x1)
        x1 = self.relu(x1)
        x2 = self.align2_1(x2)
        x2 = self.relu(x2)
        x3 = self.align3_1(x3)     
        x3 = self.relu(x3)
        
        # 채널 정렬
        x1 = self.align1_2(x1)
        x1 = self.relu(x1)
        x2 = self.align2_2(x2)
        x2 = self.relu(x2)
        x3 = self.align3_2(x3)     
        x3 = self.relu(x3)
        
        ## 바운딩 박스 기반의 가중치 마스크를 활용(정답지 가져오기 어려우면 배제)
        if targets != None and random.choice([True, False]):
            g_batch_size = x1.shape[0]
            g_height = self.target_size[1]
            g_width = self.target_size[0]
            g_bboxes = list()
            for t in targets:
                g_bboxes.append(t["boxes"])        
            g_original_size = self.target_size
            weight_mask = generate_overlap_bbox_mask(g_batch_size, g_height, g_width, g_bboxes, g_original_size)

            x1 = x1 * weight_mask
            x2 = x2 * weight_mask
            x3 = x3 * weight_mask
        
        # 채널 정렬
        attn_out1 = self.align1_3(x1)
        attn_out2 = self.align2_3(x2)
        attn_out3 = self.align3_3(x3)     
        
        
        
        # Residual 연결        
        combined = torch.cat([attn_out1, attn_out2, attn_out3], dim=1)
        combined = self.residual_conv(combined) + combined
        ## 배치노멀라이즈 
        combined = self.residual_bn(combined)  # BatchNorm 적용
        
        # 결합 및 Refinement
        refined = self.relu(self.combine_conv(combined))
        refined = self.relu(self.refinement1(refined))
        refined = self.refinement2(refined)        
        
        # 최종 마스크 생성
        mask = self.activation(self.final_conv(refined))
        return mask
  
 
 

# @register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
        ### 어텐션 맵 기반의 마스크 생성 모듈
        self.gen = Generator(512,1024,2048)        
    
    def forward(self, x, targets=None, option=True):

        if self.multi_scale and self.training and option:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        ## 백본에 이미지를 넣어서 Feature Map 추출
        backbone_x_all = self.backbone(x)  
        ## Feature Map Tensor Shape : [torch.Size([2, 256, 144, 144]), torch.Size([2, 512, 72, 72]), torch.Size([2, 1024, 36, 36]), torch.Size([2, 2048, 18, 18])]

        ## 여기서는 4개 추출 후 2~4번 Feature Map 사용
        backbone_x = backbone_x_all[1:] 


        ## Feature Map 3개 준비
        feature_map1 = backbone_x[0]
        feature_map2 = backbone_x[1]
        feature_map3 = backbone_x[2]
        
        ## option값이 True일때만 마스크 이미지를 구성
        if option :            
            ## Feature Map을 통해서 마스크를 생성
            mask = self.gen(feature_map1, feature_map2, feature_map3,targets)
            ## 해상도 맞춤
            upscaled_mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)
            ## 마스크값이 0.5이상일때만 활성화하고 나머지는 0으로. => 전경배경 분리된 이미지를 생성
            masked_image = x * (upscaled_mask >= 0.5).float()                        
        else:
            masked_image = None        
        
        # encoder
        neck_x = self.encoder(backbone_x)                
        
        # decoder
        x = self.decoder(neck_x, targets)

        ## 디코더 결과와, 마스크 이미지 반환
        return x, masked_image

    


# # Define backbone
# class Backbone(nn.Module):
#     def __init__(self, layers):
#         super(Backbone, self).__init__()
#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.layers(x)

# # Define neck
# class Neck(nn.Module):
#     def __init__(self, layers):
#         super(Neck, self).__init__()
#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.layers(x)

# # Define head
# class Head(nn.Module):
#     def __init__(self, layers):
#         super(Head, self).__init__()
#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.layers(x)

# # Define full YOLO model
# class DevidedModel(DetectionModel):
#     def __init__(self, backbone, neck, head):
#         super().__init__()
#         self.backbone = backbone
#         self.neck = neck
#         self.head = head

#         self._layers = []  # _layers 속성 추가

#         ### 어텐션 맵 기반의 마스크 생성 모듈
#         self.gen = Generator(512,1024,2048)   

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.neck(x)
#         x = self.head(x)
#         return x

#     def __getitem__(self, idx):
#         """
#         인덱스 접근을 지원하여 기존 YOLO 모델과의 호환성 유지.
#         """
#         if idx >= len(self._layers):
#             raise IndexError(f"Invalid index {idx}. DevidedModel supports 0 to {len(self._layers) - 1}.")
#         return self._layers[idx]


# Backbone, Neck, Head를 나누기 위한 클래스 정의
class Backbone(nn.Module):
    def __init__(self, model, backbone_last_idx):
        super(Backbone, self).__init__()
        self.model = nn.Sequential(*[m for i, m in enumerate(model) if i <= backbone_last_idx])  # Backbone 구성

    def forward(self, x):
        y = []
        for m in self.model:
            x = m(x)
            y.append(x)
        return x, y  # Backbone의 출력과 모든 저장된 출력


class Neck(nn.Module):
    def __init__(self, model, backbone_last_idx, neck_last_idx):
        super(Neck, self).__init__()
        self.model = nn.Sequential(*[m for i, m in enumerate(model) if backbone_last_idx < i <= neck_last_idx])  # Neck 구성

    def forward(self, x, y):
        for m in self.model:
            if isinstance(m.f, int):  # 이전 레이어에서 가져오기
                x = y[m.f]
            else:  # 여러 레이어에서 가져오기
                x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return x, y


class Head(nn.Module):
    def __init__(self, model, neck_last_idx):
        super(Head, self).__init__()
        self.model = nn.Sequential(*[m for i, m in enumerate(model) if i > neck_last_idx])  # Head 구성

    def forward(self, x, y, visualize=False, embed=None):
        embeddings = []
        for m in self.model:
            if isinstance(m.f, int):  # 이전 레이어에서 가져오기
                x = y[m.f]
            else:  # 여러 레이어에서 가져오기
                x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x


class XiilabModel(DetectionModel):
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.cfg = cfg
        self.is_split = False
        self.gen = Generator(512,1024,2048) 

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")
        
        self.split_yolo_model()

    def split_yolo_model(self):
        self.is_split = True

        backbone_last_idx = 11  # Backbone의 마지막 인덱스
        neck_last_idx = 23  # Neck의 마지막 인덱스
        self.backbone = Backbone(self.model, backbone_last_idx)
        self.neck = Neck(self.model, backbone_last_idx, neck_last_idx)
        self.head = Head(self.model, neck_last_idx)

        # 새 YOLO 모델 생성
        # model = DevidedModel(backbone, neck, head)
        # model._layers = layers  # _layers 속성 설정

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs

        if hasattr(self, "is_split") and self.is_split:
        # if False:
            # Backbone 실행
            x, y = self.backbone(x)

            for idx, y_i in enumerate(y):
                print(idx, y_i.shape)

            low_feature = y[4]
            mid_feature = y[6]

            # Neck 실행
            x, y = self.neck(x, y)

            # Head 실행
            x = self.head(x, y, visualize=visualize, embed=embed)

        else:
            for m in self.model:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)
                if embed and m.i in embed:
                    embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                    if m.i == max(embed):
                        return torch.unbind(torch.cat(embeddings, 1), dim=0)

        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)
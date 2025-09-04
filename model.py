import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet3d import resnet3d50,resnet3d36,resnet3d18
import torchvision.models as models # Import torchvision models


class TeacherClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # --- CT图像特征提取 (3D ResNet) ---

        self.ct_encoder = resnet3d18()
        self.fc_ct = nn.Linear(512, 128)  # 输出为 128 dim 特征

        # --- 音频模态 (VGG16 预训练) ---
        def make_audio_encoder():
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            features = vgg.features
            features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 修改首层支持单通道
            return features

        self.audio_encoder_stft = make_audio_encoder()
        self.audio_encoder_wave = make_audio_encoder()
        self.audio_encoder_mel = make_audio_encoder()
        self.fc_audio = nn.Linear(512, 64)  # 音频特征变换

        # --- 结构化数据（生理指标：如血压、体温等） ---
        self.fc_index = nn.Linear(6, 16)  # 输入6维，输出16维

        # --- 文本报告（BERT/LLM抽取后输出768维CLS） ---
        self.fc_rep = nn.Linear(768, 64)

        # --- 最终融合分类 ---
        total_feat_dim = 128 + 3 * 64 + 16 + 64
        self.classifier = nn.Sequential(
            nn.Linear(total_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.proj = nn.Linear(400,384) 
    def encode_audio(self, x, encoder):
        x = encoder(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)

    def forward(self, x_stft, x_wave, x_mel, x_index, x_ct, x_rep):
        # CT特征
        feat_ct = self.ct_encoder(x_ct)  # -> (B, 512, 1, 1, 1)
        feat_ct = feat_ct.view(feat_ct.size(0), -1)  # -> (B, 512)
        feat_ct = self.fc_ct(feat_ct)  # -> (B, 128)

        # 音频特征
        feat_stft = self.fc_audio(self.encode_audio(x_stft, self.audio_encoder_stft))  # -> (B, 64)
        feat_wave = self.fc_audio(self.encode_audio(x_wave, self.audio_encoder_wave))  # -> (B, 64)
        feat_mel = self.fc_audio(self.encode_audio(x_mel, self.audio_encoder_mel))     # -> (B, 64)

        # 结构化特征
        feat_index = self.fc_index(x_index)  # -> (B, 16)

        # 文本特征（CLS）
        feat_rep = self.fc_rep(x_rep)  # -> (B, 64)

        # 拼接所有模态
        feat_all = torch.cat([feat_ct, feat_stft, feat_wave, feat_mel, feat_index, feat_rep], dim=1)
        out = self.classifier(feat_all)
        return out,self.proj(feat_all) 
    
class AudioStudent_vgg(nn.Module): # Renamed to match your traceback
    def __init__(self, num_classes=2):
        super().__init__()

        vgg16_base = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        original_first_conv = vgg16_base.features[0]

        # Determine if the original layer had a bias
        has_bias = original_first_conv.bias is not None

        self.first_conv_adapted = nn.Conv2d(
            in_channels=1,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=has_bias # <--- Pass a boolean here
        )

        with torch.no_grad():
            # Copy weights - keep the unsqueeze(1) as the original code
            self.first_conv_adapted.weight.copy_(original_first_conv.weight[:, 0, :, :].unsqueeze(1))
            
            # Copy bias if it exists
            if has_bias: # Check if bias exists before copying
                self.first_conv_adapted.bias.copy_(original_first_conv.bias)


        self.vgg_features_base = vgg16_base.features[1:]
        
        self.encoder_stft = nn.Sequential(self.first_conv_adapted, self.vgg_features_base)
        self.encoder_wave = nn.Sequential(self.first_conv_adapted, self.vgg_features_base)
        self.encoder_mel = nn.Sequential(self.first_conv_adapted, self.vgg_features_base)

        self.fc_audio = nn.Linear(512, 128)
        
        self.classifier = nn.Sequential(
            nn.Linear(3 * 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_stft, x_wave, x_mel):
        def encode(x, encoder):
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            x = encoder(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            return x.view(x.size(0), -1)

        feat_stft = self.fc_audio(encode(x_stft, self.encoder_stft))
        feat_wave = self.fc_audio(encode(x_wave, self.encoder_wave))
        feat_mel = self.fc_audio(encode(x_mel, self.encoder_mel))

        feat_all = torch.cat([feat_stft, feat_wave, feat_mel], dim=1)
        out = self.classifier(feat_all)
        return out, feat_all 

class AudioStudent_vgg_2mel(nn.Module): # Renamed to match your traceback
    def __init__(self, num_classes=2):
        super().__init__()

        vgg16_base = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        original_first_conv = vgg16_base.features[0]

        # Determine if the original layer had a bias
        has_bias = original_first_conv.bias is not None

        self.first_conv_adapted = nn.Conv2d(
            in_channels=1,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=has_bias # <--- Pass a boolean here
        )

        with torch.no_grad():
            # Copy weights - keep the unsqueeze(1) as the original code
            self.first_conv_adapted.weight.copy_(original_first_conv.weight[:, 0, :, :].unsqueeze(1))
            
            # Copy bias if it exists
            if has_bias: # Check if bias exists before copying
                self.first_conv_adapted.bias.copy_(original_first_conv.bias)


        self.vgg_features_base = vgg16_base.features[1:]
        
        self.encoder_stft = nn.Sequential(self.first_conv_adapted, self.vgg_features_base)
        self.encoder_wave = nn.Sequential(self.first_conv_adapted, self.vgg_features_base)
        

        self.fc_audio = nn.Linear(512, 128)
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_stft, x_wave):
        def encode(x, encoder):
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            x = encoder(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            return x.view(x.size(0), -1)

        feat_stft = self.fc_audio(encode(x_stft, self.encoder_stft))
        feat_wave = self.fc_audio(encode(x_wave, self.encoder_wave))


        feat_all = torch.cat([feat_stft, feat_wave], dim=1)
        out = self.classifier(feat_all)
        return out

class AudioStudent_vgg_2melwave(nn.Module): # Renamed to match your traceback
    def __init__(self, num_classes=2):
        super().__init__()

        vgg16_base = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        original_first_conv = vgg16_base.features[0]

        # Determine if the original layer had a bias
        has_bias = original_first_conv.bias is not None

        self.first_conv_adapted = nn.Conv2d(
            in_channels=1,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=has_bias # <--- Pass a boolean here
        )

        with torch.no_grad():
            # Copy weights - keep the unsqueeze(1) as the original code
            self.first_conv_adapted.weight.copy_(original_first_conv.weight[:, 0, :, :].unsqueeze(1))
            
            # Copy bias if it exists
            if has_bias: # Check if bias exists before copying
                self.first_conv_adapted.bias.copy_(original_first_conv.bias)


        self.vgg_features_base = vgg16_base.features[1:]
        
        self.encoder_stft = nn.Sequential(self.first_conv_adapted, self.vgg_features_base)

        self.fc_audio = nn.Linear(512, 128)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_stft):
        def encode(x, encoder):
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            x = encoder(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            return x.view(x.size(0), -1)

        feat_stft = self.fc_audio(encode(x_stft, self.encoder_stft))


        feat_all = torch.cat([feat_stft], dim=1)
        out = self.classifier(feat_all)
        return out

class AudioStudent_resnet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # 加载预训练的ResNet50模型
        resnet50_base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # ResNet50的第一个卷积层
        original_first_conv = resnet50_base.conv1

        # 确定原始层是否有偏置
        has_bias = original_first_conv.bias is not None

        # 调整第一个卷积层以接受单通道输入
        self.first_conv_adapted = nn.Conv2d(
            in_channels=1,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=has_bias
        )

        # 复制权重。对于单通道输入，我们通常会复制原始RGB通道中的一个（例如，绿色通道）
        # 或者对RGB通道进行平均。这里我们选择复制绿色通道的权重。
        with torch.no_grad():
            self.first_conv_adapted.weight.copy_(original_first_conv.weight[:, 1, :, :].unsqueeze(1)) # 复制绿色通道的权重

            # 如果存在偏置，复制偏置
            if has_bias:
                self.first_conv_adapted.bias.copy_(original_first_conv.bias)

        # ResNet50的特征提取部分 (去除原始的conv1层)
        # ResNet的结构与VGG不同，它不是一个简单的features序列
        # 我们需要单独处理第一层，然后将剩余部分作为encoder
        self.resnet_encoder_base = nn.Sequential(
            self.first_conv_adapted,
            resnet50_base.bn1,
            resnet50_base.relu,
            resnet50_base.maxpool,
            resnet50_base.layer1,
            resnet50_base.layer2,
            resnet50_base.layer3,
            resnet50_base.layer4
        )

        # 为不同的音频表示创建独立的编码器
        # 尽管这里它们结构相同，但在实际应用中可能需要不同的权重或微调策略
        self.encoder_stft = self.resnet_encoder_base
        self.encoder_wave = self.resnet_encoder_base
        self.encoder_mel = self.resnet_encoder_base

        # ResNet50的输出特征维度通常是2048（在最后一个avgpool之前）
        # 调整fc_audio层的输入维度
        self.fc_audio = nn.Linear(2048, 128) # ResNet50的输出特征维度

        self.classifier = nn.Sequential(
            nn.Linear(3 * 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_stft, x_wave, x_mel):
        def encode(x, encoder):
            if x.dim() == 3:
                x = x.unsqueeze(1) # 增加通道维度，从(B, H, W) -> (B, 1, H, W)
            
            x = encoder(x)
            # ResNet的全局平均池化在最后
            x = F.adaptive_avg_pool2d(x, (1, 1))
            return x.view(x.size(0), -1)

        feat_stft = self.fc_audio(encode(x_stft, self.encoder_stft))
        feat_wave = self.fc_audio(encode(x_wave, self.encoder_wave))
        feat_mel = self.fc_audio(encode(x_mel, self.encoder_mel))

        feat_all = torch.cat([feat_stft, feat_wave, feat_mel], dim=1)
        out = self.classifier(feat_all)
        return out

class AudioStudent_resnet18(nn.Module): # Renamed for clarity to reflect ResNet18 usage
    def __init__(self, num_classes=2):
        super().__init__()

        # 加载预训练的ResNet18模型
        # 注意: 使用 models.ResNet18_Weights.IMAGENET1K_V1 获取预训练权重
        resnet18_base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # ResNet18的第一个卷积层
        original_first_conv = resnet18_base.conv1

        # 确定原始层是否有偏置
        has_bias = original_first_conv.bias is not None

        # 调整第一个卷积层以接受单通道输入
        self.first_conv_adapted = nn.Conv2d(
            in_channels=1,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=has_bias
        )

        # 复制权重。对于单通道输入，我们通常会复制原始RGB通道中的一个（例如，绿色通道）
        # 或者对RGB通道进行平均。这里我们选择复制绿色通道的权重。
        with torch.no_grad():
            self.first_conv_adapted.weight.copy_(original_first_conv.weight[:, 1, :, :].unsqueeze(1)) # 复制绿色通道的权重
            
            # 如果存在偏置，复制偏置
            if has_bias:
                self.first_conv_adapted.bias.copy_(original_first_conv.bias)

        # ResNet18的特征提取部分 (去除原始的conv1层)
        # ResNet的结构与VGG不同，它不是一个简单的features序列
        # 我们需要单独处理第一层，然后将剩余部分作为encoder
        self.resnet_encoder_base = nn.Sequential(
            self.first_conv_adapted,
            resnet18_base.bn1,
            resnet18_base.relu,
            resnet18_base.maxpool,
            resnet18_base.layer1,
            resnet18_base.layer2,
            resnet18_base.layer3,
            resnet18_base.layer4
        )

        # 为不同的音频表示创建独立的编码器
        # 尽管这里它们结构相同，但在实际应用中可能需要不同的权重或微调策略
        self.encoder_stft = self.resnet_encoder_base
        self.encoder_wave = self.resnet_encoder_base
        self.encoder_mel = self.resnet_encoder_base

        # ResNet18的输出特征维度通常是512（在最后一个avgpool之前）
        # 调整fc_audio层的输入维度
        self.fc_audio = nn.Linear(512, 128) # ResNet18的输出特征维度是512

        self.classifier = nn.Sequential(
            nn.Linear(3 * 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_stft, x_wave, x_mel):
        def encode(x, encoder):
            if x.dim() == 3:
                x = x.unsqueeze(1) # 增加通道维度，从(B, H, W) -> (B, 1, H, W)
            
            x = encoder(x)
            # ResNet的全局平均池化在最后
            x = F.adaptive_avg_pool2d(x, (1, 1))
            return x.view(x.size(0), -1)

        feat_stft = self.fc_audio(encode(x_stft, self.encoder_stft))
        feat_wave = self.fc_audio(encode(x_wave, self.encoder_wave))
        feat_mel = self.fc_audio(encode(x_mel, self.encoder_mel))

        feat_all = torch.cat([feat_stft, feat_wave, feat_mel], dim=1)
        out = self.classifier(feat_all)
        return out

class AudioStudent_MLPEmbed(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        def conv_block():
            return nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten()
            )
        self.stft_branch = conv_block()
        self.wave_branch = conv_block()
        self.mel_branch = conv_block()
        self.classifier = nn.Sequential(
            nn.Linear(393216, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, stft, wave, mel):
        stft_feat = self.stft_branch(stft)
        wave_feat = self.wave_branch(wave)
        mel_feat = self.mel_branch(mel)
        x = torch.cat([stft_feat, wave_feat, mel_feat], dim=1)
        # print(f"Concatenated feature shape: {x.shape}")  # Debugging line
        return self.classifier(x)

      
if __name__ == "__main__":
    model = AudioStudent_vgg()
    model2 = TeacherClassifier()
    # 假设输入数据
    x_rep = torch.randn(2, 768)
    x_stft = torch.randn(2, 1, 256, 256)
    x_wave = torch.randn(2, 1, 256, 256)  
    x_mel = torch.randn(2, 1, 256, 256)  
    x_index = torch.randn(2, 6) 
    x_ct = torch.randn(2, 1, 256, 256, 256)  
    output = model(x_stft, x_wave, x_mel)
    output2 = model2(x_stft, x_wave, x_mel, x_index, x_ct, x_rep)
    print( output[0].shape)
    print( output[1].shape)
    print( output2[0].shape)
    print( output2[1].shape)

#%%

#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
#from PIL import ImageEnhance 
from torch.utils.data import DataLoader
from PIL import Image,ImageOps
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

# %%
affine_par = True
class Convblock(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes1, planes2, stride=1, dilation=1, padding=1):
        super(Convblock, self).__init__()
        
        self.inplanes = inplanes
        self.planes1 = planes1
        self.planes2 = planes2
        self.dilation = dilation
        self.padding = padding
        conv1 = nn.Conv2d(int(self.inplanes), int(self.planes1), kernel_size=3, stride=1, padding=int(self.padding), bias=False)
        bn1 = nn.BatchNorm2d(int(self.planes1), affine=affine_par)

        print(f"Creating Convblock: inplanes={inplanes}, planes1={planes1}, planes2={planes2}")
        
        conv2 = nn.Conv2d(int(self.planes1), int(self.planes2), kernel_size=3, stride=1, padding=int(self.padding), 
                               bias=False, dilation=int(self.dilation))
        bn2 = nn.BatchNorm2d(int(self.planes2), affine=affine_par)


        conv3 = nn.Conv2d(int(self.planes2), int(self.planes2), kernel_size=3, stride=1, padding=int(self.padding),
                               bias=False, dilation=int(self.dilation))
        bn3 = nn.BatchNorm2d(int(self.planes2), affine=affine_par)
        
        relu = nn.ReLU(inplace=True)
        
        conv_block_list = list()
        conv_block_list.append(conv1)
        conv_block_list.append(bn1)
        conv_block_list.append(relu)
        conv_block_list.append(conv2)
        conv_block_list.append(bn2)
        conv_block_list.append(relu)
        conv_block_list.append(conv3)
        conv_block_list.append(bn3)
        conv_block_list.append(relu)
        self.net = nn.Sequential(*conv_block_list)
        
        
    def forward(self, x):
        for layer in self.net:
            x = layer(x)        
        return x
    
#%%
class _2D_Unet(nn.Module):
    def __init__(self, block, depth, num_classes, dilation=1, gray_scale = True, base=16):
        super(_2D_Unet, self).__init__()
        self.depth = depth
        self.num_classes = num_classes
        if gray_scale:
            self.input_channel = 1
        else:
            self.input_channel = 3
        self.base = base
        
        
        
        self.initial = nn.Conv2d(self.input_channel, self.base, kernel_size=3, stride=1, padding=1, bias=False)
        
        down_part = []
        for i in range(int(self.depth)-1):
            if i == 0:
                down_part.extend(self.down_stream_block(block, self.base*2**(i), self.base*2**(i) , self.base*2**(i)))
            else:
                down_part.extend(self.down_stream_block(block, self.base*2**(i-1), self.base*2**(i), self.base*2**(i)))
        self.down_part = nn.Sequential(*down_part)
        
        
        
        bottom_input_channel = self.base*2**(int(self.depth)-1)
        bottom_block_1 = Convblock(bottom_input_channel//2, bottom_input_channel, bottom_input_channel)
        bottom_block_2 = Convblock(bottom_input_channel, bottom_input_channel, bottom_input_channel//2)
        self.bottom_block = nn.Sequential(bottom_block_1, bottom_block_2)
        
        print("bottom_input_channel:", bottom_input_channel)
        print("bottom_input_channel//2:", bottom_input_channel//2)

        up_part = []
        for u in range(int(self.depth)-2, -1, -1):
            if u >= 1:
                up_part.extend(self.up_stream_block(block, self.base*2**(u+1), self.base*2**(u), self.base*2**(u-1), int(self.depth)))
            else:
                up_part.extend(self.up_stream_block(block, self.base*2**(u+1), self.base*2**(u), self.base*2**(u), int(self.depth)))
        self.up_part = nn.Sequential(*up_part)
        
        self.classifer = nn.Conv2d(self.base, int(self.num_classes), kernel_size=3, stride=1, padding=1, bias=False)
                
        
    
    def down_stream_block(self, block, inplanes, planes1, planes2, stride=1, dilation=1):
        layers = []
        layers.append(block(inplanes, planes1, planes2, stride=stride, dilation=dilation))
        downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        layers.append(downsample)
        return layers
    
    def up_stream_block(self, block, inplanes, planes1, planes2, depth, stride=1, dilation=1):
        layers = []
        upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners= False)
        layers.append(upsample)
        layers.append(block(inplanes, planes1, planes2, stride=stride, dilation=dilation))
        return layers
    
    
    
    def forward(self, x):
        x = self.initial(x)
        middle_module = self.down_part
        middle_products = []
        for k in range(1, (int(self.depth)-1)*2, 2):
            l = middle_module[k-1:k]
            x = l(x)
            middle_products.append(x)
            x = middle_module[k](x)
            
            
        x = self.bottom_block(x)
        
        upper_module = self.up_part
        for k in range(1, (int(self.depth)-1)*2, 2):
            middle_product = middle_products.pop()
            l = upper_module[k-1]
            x = l(x)
            diffY = x.size()[2] - middle_product.size()[2]
            diffX = x.size()[3] - middle_product.size()[3]
            middle_product = F.pad(middle_product, [diffX//2, diffX-diffX//2,
                                                    diffY//2, diffY-diffY//2])
            x = torch.cat((x, middle_product), 1)
            m = upper_module[k]
            x = m(x)
                   
        x = self.classifer(x)
        
        return x
        
    
def _2D_unet(n_class=1, gray_scale = True, base=16):
    _2d_unet = _2D_Unet(Convblock, 4, n_class, gray_scale = gray_scale, base = base)
    return _2d_unet

#%%
config = {
    "image_size": (256, 256),
    "batch_size": 4,
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "num_classes": 1,
    "gray_scale": False,
    "base_channels": 32,
    "device": "cuda",
    "train_image_dir": "Fold5/train",
    "train_mask_dir": "Fold5/trainannot",
    "val_image_dir": "Fold5/val",
    "val_mask_dir": "Fold5/valannot",
    "test_image_dir": "Fold5/test",
    "test_mask_dir": "Fold5/testannot",
}
print(f"Using device: {config['device']}")
print("num_classes:", config["num_classes"], type(config["num_classes"]))
print("gray_scale:", config["gray_scale"], type(config["gray_scale"]))
print("base_channels:", config["base_channels"], type(config["base_channels"]))

#%%
#Dataset
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, gray_scale=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.gray_scale = gray_scale
        
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        mask_name = self.mask_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        

        img = img.convert("L") if self.gray_scale else img.convert("RGB")
        mask = mask.convert("L")

        

        target_size = (256, 256)  # 與圖像相同尺寸
        mask = mask.resize(target_size, Image.NEAREST)

        if self.transform is not None:
            img = self.transform(img)
            #mask = self.transform(mask)
            #mask = torch.as_tensor(np.array(mask), dtype=torch.long)
            mask = np.array(mask, dtype=np.uint8)
            mask = torch.from_numpy(mask).float() / 255.0
        #print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
        
        

        return img, mask


#%%
input_img_paths = sorted(
    [
        os.path.join("Fold5/train", fname)
        for fname in os.listdir("Fold5/train")
        if fname.endswith(".jpg")
    ]
)

# 取得目標圖遮罩的檔案路徑
target_img_paths = sorted(
    [
        os.path.join("Fold5/trainannot", fname)
        for fname in os.listdir("Fold5/trainannot")
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)
input_image = Image.open(input_img_paths[9]).resize((256, 256))
target_image = Image.open(target_img_paths[9]).resize((256, 256))
target_image = ImageOps.autocontrast(target_image)
input_image = ImageOps.autocontrast(input_image)

plt.figure(figsize=(6, 3))  # 調整整體圖表尺寸

# 原圖
plt.subplot(1, 2, 1)
plt.title("Input")
plt.imshow(input_image)
plt.axis("off")

# 遮罩圖
plt.subplot(1, 2, 2)
plt.title("Mask")
plt.imshow(target_image)
plt.axis("off")

plt.tight_layout()
plt.show()
#%%
#DataLoader 
transform = transforms.Compose([
    transforms.Resize(config["image_size"]),
    transforms.ToTensor()
])

train_dataset = SegmentationDataset(config["train_image_dir"], config["train_mask_dir"], transform=transform, gray_scale=config["gray_scale"])
val_dataset   = SegmentationDataset(config["val_image_dir"],   config["val_mask_dir"],   transform=transform, gray_scale=config["gray_scale"])
test_dataset   = SegmentationDataset(config["test_image_dir"],   config["test_mask_dir"],   transform=transform, gray_scale=config["gray_scale"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)


model = _2D_unet(n_class=config["num_classes"], gray_scale=config["gray_scale"], base=config["base_channels"]).to(config["device"])
#criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 展平預測和目標
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice
criterion = DiceLoss()

#%%
#計算IOU
#def compute_iou(pred, target):
#    pred = (pred > 0.5).float()
#    target = (target > 0.5).float()
#    
#    # 展平張量
#    pred = pred.view(-1)
#    target = target.view(-1)
#    
#    # 計算交集和聯集
#    intersection = (pred * target).sum().item()
#    union = (pred + target).sum().item() - intersection
#    
#    return intersection / union if union > 0 else 0.0
#
def compute_iou(a=None, b=None, model=None, data_loader=None, device=None, return_first_sample=False):
    """
    通用IoU計算函數，支持兩種模式：
    1. 單個預測與目標之間的IoU: compute_iou(pred, target)
    2. 整個數據集的平均IoU: compute_iou(model=model, data_loader=data_loader, device=device)
    """
    # 模式1：計算單對預測與目標之間的IoU
    if a is not None and b is not None and model is None and data_loader is None:
        pred, target = a, b
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        
        # 展平張量
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 計算交集和聯集
        intersection = (pred * target).sum().item()
        union = (pred + target).sum().item() - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # 模式2：計算整個數據集的平均IoU
    elif model is not None and data_loader is not None and device is not None:
        model.eval()
        total_iou = 0.0
        count = 0
        first_sample = None
        
        with torch.no_grad():
            for img, mask in tqdm(data_loader, desc="Computing IoU", leave=False):
                img = img.to(device)
                mask = mask.to(device)
                if len(mask.shape) == 3:
                    mask = mask.unsqueeze(1)
                
                output = model(img)
                pred = (torch.sigmoid(output) > 0.5).float()
                
                # 遞歸調用自身的模式1來計算單個IoU
                iou = compute_iou(pred.cpu(), mask.cpu())
                total_iou += iou
                count += 1
                
                # 保存第一個樣本用於可視化
                if return_first_sample and first_sample is None:
                    first_sample = (img.cpu(), mask.cpu(), pred.cpu())
        
        avg_iou = total_iou / count if count > 0 else 0
        
        if return_first_sample:
            return avg_iou, first_sample
        else:
            return avg_iou
    
    else:
        raise ValueError("Invalid arguments combination")
#%%
#訓練+驗證
# 初始化IoU追蹤列表
train_ious = []
val_ious = []
test_ious = []
for epoch in range(config["num_epochs"]):
    model.train()
    total_loss = 0

    
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
        images = images.to(config["device"])
        if len(masks.shape) == 3:  # 如果掩碼沒有通道維度
            masks = masks.unsqueeze(1).to(config["device"])

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

    

    val_iou, first_val_sample = compute_iou(
        model=model, data_loader=val_loader, device=config["device"], return_first_sample=True
    )
    first_val_img, first_val_mask, first_pred_mask = first_val_sample
    
    # 計算訓練集和測試集的IoU
    train_iou = compute_iou(model=model, data_loader=train_loader, device=config["device"])
    test_iou = compute_iou(model=model, data_loader=test_loader, device=config["device"])
    
    # 打印所有IoU結果
    print(f"[Epoch {epoch+1}] IoU Metrics - Train: {train_iou:.4f}, Val: {val_iou:.4f}, Test: {test_iou:.4f}")
    
    # 將IoU添加到追蹤列表
    train_ious.append(train_iou)
    val_ious.append(val_iou)
    test_ious.append(test_iou)

    # 遮罩對比圖
    if first_val_img is not None and first_val_mask is not None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        # 顯示原始彩色影像
        if config["gray_scale"]:
            # 灰階模式
            img_array = first_val_img.squeeze().cpu().numpy()
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            img_pil = ImageOps.autocontrast(img_pil)
            axs[0].imshow(np.array(img_pil), cmap="gray")
        else:
            # 彩色模式 - 注意通道順序轉換：從 [C,H,W] 到 [H,W,C]
            img_array = first_val_img.squeeze().permute(1, 2, 0).cpu().numpy()
            # 確保值在0-1範圍內
            img_array = np.clip(img_array, 0, 1)
            # 轉換為PIL Image (需要將值範圍從0-1轉為0-255)
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            img_pil = ImageOps.autocontrast(img_pil)
            axs[0].imshow(np.array(img_pil))
        
        axs[0].set_title("Original Image")
        
        # 顯示真實掩碼
        mask_array = first_val_mask.squeeze().cpu().numpy()
        #轉換為PIL Image (需要將值範圍從0-1轉為0-255)
        mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8))
        mask_pil = ImageOps.autocontrast(mask_pil)
        axs[1].imshow(np.array(mask_pil))
        axs[1].set_title("Ground Truth")
        
        # 顯示預測掩碼
        pred_array = first_pred_mask.squeeze().cpu().numpy()
        # 將預測掩碼轉換為PIL Image
        pred_pil = Image.fromarray((pred_array * 255).astype(np.uint8))
        pred_pil = ImageOps.autocontrast(pred_pil)
        axs[2].imshow(np.array(pred_pil))
        axs[2].set_title("Prediction")
        
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()


# %%
#計算誤差指標
# 計算測試集上的Y座標誤差



model.eval()
total_y_error_pixels = 0.0
valid_mask_count = 0

# 定義兩個閾值和對應的計數器
threshold_cm_1 = 0.5  # 第一個閾值：0.5公分
threshold_pixels_1 = threshold_cm_1 * 72.0  # 轉換為像素
error_within_threshold_1_count = 0  # 記錄誤差在0.5公分內的樣本數

threshold_cm_2 = 1.0  # 第二個閾值：1.0公分
threshold_pixels_2 = threshold_cm_2 * 72.0  # 轉換為像素
error_within_threshold_2_count = 0  # 記錄誤差在1.0公分內的樣本數

with torch.no_grad():
    for test_img, test_mask in tqdm(test_loader, desc="Testing"):
        test_img = test_img.to(config["device"])
        test_mask = test_mask.to(config["device"])
        
        # 模型預測
        test_output = model(test_img)
        pred_mask = (torch.sigmoid(test_output) > 0.5).float()
        
        # 尾端Y座標誤差計算
        true_mask_np = test_mask.cpu().numpy().squeeze()
        pred_mask_np = pred_mask.cpu().numpy().squeeze()
        
        if np.sum(true_mask_np) > 0 and np.sum(pred_mask_np) > 0:
            true_mask_indices = np.where(true_mask_np > 0.5)
            pred_mask_indices = np.where(pred_mask_np > 0.5)
            
            if len(true_mask_indices[0]) > 0 and len(pred_mask_indices[0]) > 0:
                true_end_y = np.max(true_mask_indices[0])
                pred_end_y = np.max(pred_mask_indices[0])
                
                y_error_pixels = abs(true_end_y - pred_end_y)
                total_y_error_pixels += y_error_pixels
                valid_mask_count += 1

                print(y_error_pixels)
                
                # 檢查誤差是否在各閾值內
                if y_error_pixels <= threshold_pixels_1:
                    error_within_threshold_1_count += 1
                
                if y_error_pixels <= threshold_pixels_2:
                    error_within_threshold_2_count += 1

# 計算並顯示平均誤差和準確率
if valid_mask_count > 0:
    avg_y_error_pixels = total_y_error_pixels / valid_mask_count
    avg_y_error_cm = avg_y_error_pixels / 72.0  # 72像素 = 1公分
    
    # 計算準確率：誤差在閾值內的樣本比例
    accuracy_within_threshold_1 = (error_within_threshold_1_count / valid_mask_count) * 100
    accuracy_within_threshold_2 = (error_within_threshold_2_count / valid_mask_count) * 100
    
    print(f"\nTest Set Results ({valid_mask_count} valid masks):")
    print(f"Average Y-coordinate Error: {avg_y_error_pixels:.2f} pixels ({avg_y_error_cm:.2f} cm)")
    print(f"Accuracy within {threshold_cm_1} cm: {accuracy_within_threshold_1:.2f}% ({error_within_threshold_1_count}/{valid_mask_count})")
    print(f"Accuracy within {threshold_cm_2} cm: {accuracy_within_threshold_2:.2f}% ({error_within_threshold_2_count}/{valid_mask_count})")
else:
    print("\nNo valid mask pairs found in test set.")

# %%

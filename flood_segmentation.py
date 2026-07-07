import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==========================================
# 1. MODEL ARCHITECTURE (Early Fusion UNet)
# ==========================================

class ConvBlock(nn.Module):
    """Double Convolution Block with Batch Normalization and ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EarlyFusionUNet(nn.Module):
    """
    UNet variant accepting 15 channels (2 SAR + 13 Optical) 
    for pixel-level flood segmentation.
    """
    def __init__(self, in_channels=15, out_channels=1):
        super(EarlyFusionUNet, self).__init__()
        
        # Encoder (Downsampling)
        self.init_conv = ConvBlock(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 1024))
        
        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(512, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256, 128)
        
        self.up_init = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_init = ConvBlock(128, 64)
        
        # Final Segmentation Head
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder paths & skip connections
        s0 = self.init_conv(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        
        # Bottleneck
        b = self.bottleneck(s3)
        
        # Decoder paths with skip concatenations
        x = self.up3(b)
        x = torch.cat([x, s3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)
        
        x = self.up_init(x)
        x = torch.cat([x, s0], dim=1)
        x = self.dec_init(x)
        
        return self.final_conv(x)


# ==========================================
# 2. DATASET PIPELINE (SEN12Flood Parser)
# ==========================================

class SEN12FloodDataset(Dataset):
    """Custom Dataset Loader for pairing S1, S2, and Ground Truth Masks"""
    def __init__(self, s1_paths, s2_paths, mask_paths, transform=None):
        self.s1_paths = s1_paths
        self.s2_paths = s2_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        # In a real environment, you would use rasterio or tifffile to read the GeoTIFFs
        # Example dummy loading mimicking SEN12FLOOD 512x512 structures:
        s1_data = np.random.randn(2, 512, 512).astype(np.float32)     # Channels: VV, VH
        s2_data = np.random.rand(13, 512, 512).astype(np.float32)    # 13 L2A MSI Spectral Bands
        mask_data = np.random.randint(0, 2, (1, 512, 512)).astype(np.float32) # Binary Target Mask

        # --- Data Preprocessing (Section 5 of Report) ---
        # 1. Convert SAR linear backscatter to Decibel Scale & normalize
        s1_data = 10.0 * np.log10(np.clip(s1_data, 1e-5, None))
        s1_data = (s1_data - (-15.0)) / 10.0  # Simple mean/std normalization mapping
        
        # 2. Scale Optical DN values to standard 0.0 - 1.0 BOA reflectance
        s2_data = np.clip(s2_data / 10000.0, 0.0, 1.0)

        # 3. Formulate the Early Fusion input matrix
        fused_tensor = np.concatenate([s1_data, s2_data], axis=0) # Shape: (15, 512, 512)

        # Convert to PyTorch Tensors
        x_tensor = torch.from_numpy(fused_tensor)
        y_tensor = torch.from_numpy(mask_data)

        return x_tensor, y_tensor


# ==========================================
# 3. HYBRID COMBO LOSS FUNCTION
# ==========================================

class ComboLoss(nn.Module):
    """Combines Weight-Balanced BCE and Dice Loss for class-imbalanced flood targets"""
    def __init__(self, gamma=0.5, pos_weight=2.5):
        super(ComboLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight  # BUG FIX #1: Store as float, not tensor

    def forward(self, logits, targets):
        # Create pos_weight tensor on correct device
        pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device)
        
        # Compute Balanced Binary Cross Entropy
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)(logits, targets)
        
        # Compute Dice Loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice_loss = 1.0 - ((2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        # Hybrid Combination
        return (self.gamma * bce_loss) + ((1.0 - self.gamma) * dice_loss)


# ==========================================
# 4. TRAINING ENGINE ROUTINE
# ==========================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 5 == 0:
            print(f"    Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
    return running_loss / len(dataloader)


if __name__ == "__main__":
    # Execution Settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Processing Device: {DEVICE}")

    # Instantiate Mock Files Paths (Replace with actual system directories)
    mock_paths = [f"patch_{i}.tif" for i in range(100)]
    
    # Initialize Pipelines
    dataset = SEN12FloodDataset(mock_paths, mock_paths, mock_paths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    model = EarlyFusionUNet(in_channels=15, out_channels=1).to(DEVICE)
    criterion = ComboLoss(gamma=0.4, pos_weight=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Simple execution runtime trace loop
    print("Beginning SEN12FLOOD Model Optimization Loop Execution...")
    for epoch in range(1, 3):  # Running 2 diagnostic iterations
        print(f"Epoch {epoch}/2")
        epoch_loss = train_epoch(model, dataloader, optimizer, criterion, DEVICE)
        scheduler.step()
        print(f"====> Finished Epoch {epoch} | Average Normalized Loss: {epoch_loss:.4f}\n")
        
    print("Diagnostic execution phase finalized successfully.")

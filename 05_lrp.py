# %% Imports
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import cv2
import os
from torch.utils.data import Dataset, DataLoader

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% Real Brain MRI Dataset
class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))

        print(f"✅ Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# %% Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %% Load dataset
DATA_ROOT_TRAIN = "data/brain_mri/training"
DATA_ROOT_TEST = "data/brain_mri/testing"

if not os.path.exists(DATA_ROOT_TRAIN):
    raise FileNotFoundError(f"Training directory not found: {DATA_ROOT_TRAIN}")
if not os.path.exists(DATA_ROOT_TEST):
    raise FileNotFoundError(f"Testing directory not found: {DATA_ROOT_TEST}")

train_dataset = BrainMRIDataset(DATA_ROOT_TRAIN, transform=train_transform)
test_dataset = BrainMRIDataset(DATA_ROOT_TEST, transform=train_transform)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Model: VGG16 with 4 classes
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        in_feats = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_feats, 4)

    def forward(self, x):
        return self.vgg16(x)

model = CNNModel().to(device)

# Replace MaxPool2d with AvgPool2d
def replace_maxpool_with_avgpool(module):
    for name, child in module.named_children():
        if isinstance(child, nn.MaxPool2d):
            setattr(module, name, nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            replace_maxpool_with_avgpool(child)

replace_maxpool_with_avgpool(model)
print("✅ All MaxPool2d replaced with AvgPool2d")

# %% Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
epochs = 5

print("Starting training on real brain MRI data...")
model.train()
for epoch in range(epochs):
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}')

print("Training completed.")

# %% Evaluation
model.eval()
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels_np = labels.numpy()
with torch.no_grad():
    outputs = model(inputs)
preds = outputs.max(1).indices.cpu().numpy()

print("Batch accuracy:", (labels_np == preds).mean())
comparison = pd.DataFrame({"labels": labels_np, "outputs": preds})
print(comparison.head())

# %% LRP (keep for comparison)
def apply_lrp_on_vgg16(model, image):
    image = image.unsqueeze(0)
    layers = list(model.vgg16.features) + [model.vgg16.avgpool]

    classifier_layers = []
    for layer in model.vgg16.classifier:
        if isinstance(layer, nn.Linear):
            if len(classifier_layers) == 0:
                m, n = 512, layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, kernel_size=7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 7, 7))
                newlayer.bias = nn.Parameter(layer.bias)
            else:
                m, n = layer.weight.shape[1], layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, kernel_size=1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
                newlayer.bias = nn.Parameter(layer.bias)
            classifier_layers.append(newlayer)
        elif isinstance(layer, nn.ReLU):
            classifier_layers.append(nn.ReLU(inplace=False))

    full_layers = layers + classifier_layers + [nn.Flatten()]

    activations = [image]
    for layer in full_layers:
        out = layer(activations[-1])
        activations.append(out)

    output_act = activations[-1].detach().cpu().numpy()
    max_val = output_act.max()
    one_hot = np.where(output_act == max_val, max_val, 0.0)
    relevances = [None] * len(activations)
    relevances[-1] = torch.FloatTensor(one_hot).to(device)

    for idx in reversed(range(len(full_layers))):
        current = full_layers[idx]
        act = activations[idx].data.requires_grad_(True)

        if isinstance(current, (nn.Conv2d, nn.AvgPool2d)):
            z = current(act) + 1e-9
            s = (relevances[idx + 1] / z).data
            (z * s).sum().backward()
            c = act.grad
            relevances[idx] = (act * c).data
        else:
            relevances[idx] = relevances[idx + 1]

    return relevances[0]

# %% ✅ RISE Implementation (Model-Agnostic)
def generate_masks(n_masks=500, input_size=224, p1=0.5, seed=42):
    """Generate random binary masks at coarse resolution and upsample."""
    np.random.seed(seed)
    cell_size = 8
    n_cells = input_size // cell_size
    masks = []
    for _ in range(n_masks):
        # Random coarse mask
        coarse = (np.random.rand(n_cells, n_cells) < p1).astype(np.float32)
        # Upsample to full size
        fine = cv2.resize(coarse, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
        masks.append(fine)
    return np.stack(masks)  # Shape: [N, H, W]

def rise_explain(model, image_tensor, class_idx=None, n_masks=500, batch_size=32):
    """
    Compute RISE saliency map.
    Args:
        model: trained PyTorch model
        image_tensor: normalized tensor of shape [3, H, W]
        class_idx: target class index (int)
        n_masks: number of random masks
        batch_size: inference batch size
    Returns:
        saliency_map: numpy array [H, W], values in [0, 1]
    """
    model.eval()
    device = next(model.parameters()).device
    H, W = image_tensor.shape[1], image_tensor.shape[2]
    img = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

    # Generate masks
    masks = generate_masks(n_masks=n_masks, input_size=H, p1=0.5)

    saliency = np.zeros((H, W), dtype=np.float32)

    # Process in batches
    for i in range(0, n_masks, batch_size):
        end = min(i + batch_size, n_masks)  # 确保不超出范围
        batch_masks = masks[i:end]  # [B, H, W]
        B = len(batch_masks)
        if B == 0:
            continue
        batch_imgs = img.repeat(B, 1, 1, 1) * torch.from_numpy(batch_masks).float().unsqueeze(1).to(device)  # [B, 3, H, W]

        with torch.no_grad():
            outputs = model(batch_imgs)  # [B, num_classes]
            probs = torch.softmax(outputs, dim=1)  # [B, num_classes]
            if class_idx is None:
                class_idx = outputs[0].argmax().item()
            scores = probs[:, class_idx].cpu().numpy()  # [B]

        # Accumulate weighted masks —— 修复索引问题
        for j, score in enumerate(scores):
            saliency += score * batch_masks[j]  # 使用局部索引 j 而非全局索引 i+j

    # Normalize by expected mask sum (n_masks * p1)
    saliency = saliency / (n_masks * 0.5 + 1e-8)
    # Clamp and normalize to [0, 1]
    saliency = np.clip(saliency, 0, None)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

# %% Visualization
image_id = 0
image_tensor = inputs[image_id]

# --- LRP ---
print("Computing LRP...")
lrp_result = apply_lrp_on_vgg16(model, image_tensor)
lrp_heatmap = lrp_result.permute(0, 2, 3, 1).cpu().numpy()[0]
lrp_heatmap = np.interp(lrp_heatmap, (lrp_heatmap.min(), lrp_heatmap.max()), (0, 1))

# --- RISE ---
print("Computing RISE... (this may take 10-30 seconds)")
rise_saliency = rise_explain(model, image_tensor, class_idx=preds[image_id], n_masks=500)

# Denormalize original image
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
orig_img = inv_normalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
orig_img = np.clip(orig_img, 0, 1)

# Create overlay
rise_uint8 = np.uint8(255 * rise_saliency)
rise_heatmap_vis = cv2.applyColorMap(rise_uint8, cv2.COLORMAP_JET)
rise_heatmap_vis = cv2.cvtColor(rise_heatmap_vis, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(np.uint8(orig_img * 255), 0.6, rise_heatmap_vis, 0.4, 0)

# Plot
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
true_label = class_names[labels_np[image_id]]
pred_label = class_names[preds[image_id]]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(orig_img)
axes[0, 0].set_title(f"Original\n(True: {true_label}, Pred: {pred_label})")
axes[0, 0].axis('off')

axes[0, 1].imshow(lrp_heatmap[:, :, 0], cmap="seismic", vmin=0, vmax=1)
axes[0, 1].set_title("LRP Heatmap")
axes[0, 1].axis('off')

axes[1, 0].imshow(rise_saliency, cmap="jet", vmin=0, vmax=1)
axes[1, 0].set_title("RISE Saliency")
axes[1, 0].axis('off')

axes[1, 1].imshow(overlay)
axes[1, 1].set_title("RISE Overlay")
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig("xai_comparison_real.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Visualization saved as 'xai_comparison_real.png'")

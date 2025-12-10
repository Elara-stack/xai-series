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

# %% Real Brain MRI Dataset (Match Your Folder Names)
class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # ✅ Use exact folder names from your data
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

# %% Load dataset with correct paths
DATA_ROOT_TRAIN = "data/brain_mri/training"
DATA_ROOT_TEST = "data/brain_mri/testing"

if not os.path.exists(DATA_ROOT_TRAIN):
    raise FileNotFoundError(f"Training directory not found: {DATA_ROOT_TRAIN}")
if not os.path.exists(DATA_ROOT_TEST):
    raise FileNotFoundError(f"Testing directory not found: {DATA_ROOT_TEST}")

# Load full datasets
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

# Replace all MaxPool2d with AvgPool2d recursively
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

# %% Evaluation on a test batch
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

# %% Grad-CAM: SAFE VERSION
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self._register_forward_hook()

    def _register_forward_hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        found = False
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                found = True
                break
        if not found:
            available_convs = [name for name, m in self.model.named_modules() if isinstance(m, nn.Conv2d)]
            raise ValueError(
                f"Layer '{self.target_layer_name}' not found.\n"
                f"Available Conv2d layers:\n" + "\n".join(available_convs[:10]) + "..."
            )

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        with torch.enable_grad():
            outputs = self.model(input_tensor)
            if class_idx is None:
                class_idx = outputs.argmax(dim=1).item()
            one_hot = torch.zeros_like(outputs)
            one_hot[0][class_idx] = 1.0

            params = list(self.model.parameters())
            grads = torch.autograd.grad(outputs, params, grad_outputs=one_hot, retain_graph=True, allow_unused=True)

            target_grad = None
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                if self.target_layer_name in name and 'weight' in name:
                    target_grad = grad
                    break

            if target_grad is None or self.activations is None:
                raise RuntimeError("Failed to capture activations or gradients.")

            weights = torch.mean(target_grad, dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            return cam.squeeze().cpu().numpy()

# %% LRP (same as before)
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

# %% Visualization
image_id = 0
image_tensor = inputs[image_id]

# --- LRP ---
lrp_result = apply_lrp_on_vgg16(model, image_tensor)
lrp_heatmap = lrp_result.permute(0, 2, 3, 1).cpu().numpy()[0]
lrp_heatmap = np.interp(lrp_heatmap, (lrp_heatmap.min(), lrp_heatmap.max()), (0, 1))

# --- Grad-CAM ---
cam_map = np.zeros((224, 224))
try:
    grad_cam = GradCAM(model, target_layer_name='vgg16.features.24')  # 👈 更早的层
    raw_cam = grad_cam(image_tensor.unsqueeze(0), class_idx=preds[image_id])
    if raw_cam is not None and raw_cam.size > 0:
        cam_map = raw_cam
except Exception as e:
    print("❌ Grad-CAM failed:", e)

# Denormalize original image
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
orig_img = inv_normalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
orig_img = np.clip(orig_img, 0, 1)

# --- Safe Grad-CAM Visualization ---
if cam_map is not None and cam_map.size > 0:
    if cam_map.ndim == 3:
        cam_map = cam_map[0]
    elif cam_map.ndim != 2:
        cam_map = np.zeros((224, 224))
    cam_normalized = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
    cam_uint8 = np.uint8(255 * cam_normalized)
    cam_resized = cv2.resize(cam_uint8, (224, 224))
    cam_heatmap_vis = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    cam_heatmap_vis = cv2.cvtColor(cam_heatmap_vis, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.uint8(orig_img * 255), 0.6, cam_heatmap_vis, 0.4, 0)
else:
    overlay = np.uint8(orig_img * 255)
    cam_map = np.zeros((224, 224))

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

axes[1, 0].imshow(cam_map, cmap="jet", vmin=0, vmax=1)
axes[1, 0].set_title("Grad-CAM")
axes[1, 0].axis('off')

axes[1, 1].imshow(overlay)
axes[1, 1].set_title("Grad-CAM Overlay")
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig("xai_comparison_real.png", dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualization saved as 'xai_comparison_real.png'")

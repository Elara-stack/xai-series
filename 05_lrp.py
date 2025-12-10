# %% Imports
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd 
import cv2
from PIL import Image

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% Demo Dataset
class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image_np = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        image_pil = Image.fromarray(image_np)
        label = np.random.randint(0, 4)
        if self.transform:
            image_pil = self.transform(image_pil)
        return image_pil, label

# %% Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = train_transform

# %% Datasets & Loaders
train_dataset = DemoDataset(num_samples=200, transform=train_transform)
test_dataset = DemoDataset(num_samples=50, transform=test_transform)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        in_feats = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_feats, 4)

    def forward(self, x):
        return self.vgg16(x)

model = CNNModel().to(device)

# Replace all MaxPool2d with AvgPool2d to avoid view/inplace issues in Grad-CAM and LRP
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
optimizer = optim.Adam(model.parameters(), lr=1e-5)
epochs = 3

print("Starting training...")
model.train()
for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        if i > 10:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Batch {i}, Loss: {loss.item():.4f}')

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

# %% Grad-CAM (Safe version after MaxPool → AvgPool replacement)
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.clone().detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].clone().detach()

        found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                found = True
                break
        if not found:
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model.")

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured.")

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()

# %% Simplified LRP (now safe because MaxPool → AvgPool)
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
            classifier_layers.append(nn.ReLU(inplace=False))  # avoid inplace
        # Skip Dropout

    full_layers = layers + classifier_layers + [nn.Flatten()]

    # Forward pass
    activations = [image]
    for layer in full_layers:
        out = layer(activations[-1])
        activations.append(out)

    # One-hot relevance at output
    output_act = activations[-1].detach().cpu().numpy()
    max_val = output_act.max()
    one_hot = np.where(output_act == max_val, max_val, 0.0)
    relevances = [None] * len(activations)
    relevances[-1] = torch.FloatTensor(one_hot).to(device)

    # Backward relevance propagation
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

# LRP
lrp_result = apply_lrp_on_vgg16(model, image_tensor)
lrp_heatmap = lrp_result.permute(0, 2, 3, 1).cpu().numpy()[0]
lrp_heatmap = np.interp(lrp_heatmap, (lrp_heatmap.min(), lrp_heatmap.max()), (0, 1))

# Grad-CAM
try:
    grad_cam = GradCAM(model, target_layer_name='vgg16.features.28')
    cam_map = grad_cam(image_tensor.unsqueeze(0), class_idx=preds[image_id])
except Exception as e:
    print("❌ Grad-CAM failed:", e)
    cam_map = np.zeros((224, 224))

# Denormalize original image
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
orig_img = inv_normalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
orig_img = np.clip(orig_img, 0, 1)

# Overlay Grad-CAM
if cam_map.any():
    cam_resized = cv2.resize(cam_map, (224, 224))
    cam_heatmap_vis = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    cam_heatmap_vis = cv2.cvtColor(cam_heatmap_vis, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.uint8(orig_img * 255), 0.6, cam_heatmap_vis, 0.4, 0)
else:
    overlay = np.uint8(orig_img * 255)

# Plot
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
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
plt.savefig("xai_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualization saved as 'xai_comparison.png'")

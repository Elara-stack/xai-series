# %% Imports
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import copy
import pandas as pd 

# added sections
import cv2
from PIL import Image
import torch.nn.functional as F

# Set GPU device
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # added sections


# %% Load data - FIXED: Corrected test dataset path
TRAIN_ROOT = "data/brain_mri/training"
TEST_ROOT = "data/brain_mri/testing"

# added sections
# Load training dataset
train_dataset = torchvision.datasets.ImageFolder(
    root=TRAIN_ROOT,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)

# Load test dataset - FIXED: Now correctly uses TEST_ROOT
test_dataset = torchvision.datasets.ImageFolder(
    root=TEST_ROOT,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)


# %% Building the model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True) 

        # Replace output layer according to our problem
        in_feats = self.vgg16.classifier[6].in_features 
        self.vgg16.classifier[6] = nn.Linear(in_feats, 4)

    def forward(self, x):
        x = self.vgg16(x)
        return x

model = CNNModel()
model.to(device)

# added sections
# %% Create data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)


# added sections
# %% Train - with proper normalization for VGG16
# Prepare normalized datasets for training
normalized_train_dataset = torchvision.datasets.ImageFolder(
    root=TRAIN_ROOT,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
)

normalized_test_dataset = torchvision.datasets.ImageFolder(
    root=TEST_ROOT,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
)

# Update loaders with normalized data
normalized_train_loader = torch.utils.data.DataLoader(
    normalized_train_dataset,
    batch_size=batch_size,
    shuffle=True
)
normalized_test_loader = torch.utils.data.DataLoader(
    normalized_test_dataset,
    batch_size=batch_size,
    shuffle=True
)




# %% Train
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
epochs = 5 # Reduced for demonstration

print("Starting training...")
for epoch in range(epochs):  
    for i, batch in enumerate(normalized_train_loader, 0):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        oss = cross_entropy_loss(outputs, labels) # added sections
        loss = cross_entropy_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        # added sectons
        if i % 10 == 0:  # Print every 10 batches
            print(f'Epoch {epoch+1}/{epochs}, Batch {i}, Loss: {loss.item():.4f}')

print("Training completed.")

# %% Inspect predictions for first batch
import pandas as pd
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.numpy()
outputs = model(inputs).max(1).indices.detach().cpu().numpy()
comparison = pd.DataFrame()
# added sections
comparison["labels"] = labels
comparison["outputs"] = outputs
print(comparison.head())

# %% Layerwise relevance propagation for VGG16
# For other CNN architectures this code might become more complex
# Source: https://git.tu-berlin.de/gmontavon/lrp-tutorial
# http://iphome.hhi.de/samek/pdf/MonXAI19.pdf

def new_layer(layer, g):
    """Clone a layer and pass its parameters through the function g."""
    layer = copy.deepcopy(layer)
    try: layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError: pass
    try: layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError: pass
    return layer

def dense_to_conv(layers):
    """ Converts a dense layer to a conv layer """
    newlayers = []
    for i,layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            newlayer = None
            if i == 0:
                m, n = 512, layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,7,7))
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))
            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers

def get_linear_layer_indices(model):
    offset = len(model.vgg16._modules['features']) + 1
    indices = []
    for i, layer in enumerate(model.vgg16._modules['classifier']): 
        if isinstance(layer, nn.Linear): 
            indices.append(i)
    indices = [offset + val for val in indices]
    return indices

def apply_lrp_on_vgg16(model, image):
    image = torch.unsqueeze(image, 0)
    # >>> Step 1: Extract layers
    layers = list(model.vgg16._modules['features']) \
                + [model.vgg16._modules['avgpool']] \
                + dense_to_conv(list(model.vgg16._modules['classifier']))
    linear_layer_indices = get_linear_layer_indices(model)
    # >>> Step 2: Propagate image through layers and store activations
    n_layers = len(layers)
    activations = [image] + [None] * n_layers # list of activations
    
    for layer in range(n_layers):
        if layer in linear_layer_indices:
            if layer == 32:
                activations[layer] = activations[layer].reshape((1, 512, 7, 7))
        activation = layers[layer].forward(activations[layer])
        if isinstance(layers[layer], torch.nn.modules.pooling.AdaptiveAvgPool2d):
            activation = torch.flatten(activation, start_dim=1)
        activations[layer+1] = activation

    # >>> Step 3: Replace last layer with one-hot-encoding
    output_activation = activations[-1].detach().cpu().numpy()
    max_activation = output_activation.max()
    one_hot_output = [val if val == max_activation else 0 
                        for val in output_activation[0]]

    activations[-1] = torch.FloatTensor([one_hot_output]).to(device)

    # >>> Step 4: Backpropagate relevance scores
    relevances = [None] * n_layers + [activations[-1]]
    # Iterate over the layers in reverse order
    for layer in range(0, n_layers)[::-1]:
        current = layers[layer]
        # Treat max pooling layers as avg pooling
        if isinstance(current, torch.nn.MaxPool2d):
            layers[layer] = torch.nn.AvgPool2d(2)
            current = layers[layer]
        if isinstance(current, torch.nn.Conv2d) or \
           isinstance(current, torch.nn.AvgPool2d) or\
           isinstance(current, torch.nn.Linear):
            activations[layer] = activations[layer].data.requires_grad_(True)
            
            # Apply variants of LRP depending on the depth
            # see: https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
            # Lower layers, LRP-gamma >> Favor positive contributions (activations)
            if layer <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
            # Middle layers, LRP-epsilon >> Remove some noise / Only most salient factors survive
            if 17 <= layer <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            # Upper Layers, LRP-0 >> Basic rule
            if layer >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9
            
            # Transform weights of layer and execute forward pass
            z = incr(new_layer(layers[layer],rho).forward(activations[layer]))
            # Element-wise division between relevance of the next layer and z
            s = (relevances[layer+1]/z).data                                     
            # Calculate the gradient and multiply it by the activation
            (z * s).sum().backward(); 
            c = activations[layer].grad       
            # Assign new relevance values           
            relevances[layer] = (activations[layer]*c).data                          
        else:
            relevances[layer] = relevances[layer+1]

    # >>> Potential Step 5: Apply different propagation rule for pixels
    return relevances[0]

# %% Grad-CAM Implementation - ADDED
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]

        # Global average pooling of gradients to get weights
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]
        cam = torch.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()

# %% Calculate relevances for first image in test batch - FIXED
image_id = 0  # Changed from 31 to 0 to avoid index errors
image_tensor = inputs[image_id]

# Apply LRP
image_relevances = apply_lrp_on_vgg16(model, image_tensor)
image_relevances = image_relevances.permute(0,2,3,1).detach().cpu().numpy()[0]
image_relevances = np.interp(image_relevances, (image_relevances.min(),
                                                image_relevances.max()), 
                                                (0, 1))

# Apply Grad-CAM
grad_cam = GradCAM(model, target_layer='features.29')  # Last conv layer in VGG16
cam_map = grad_cam(image_tensor.unsqueeze(0), class_idx=outputs[image_id])

# Prepare original image for visualization
orig_img = image_tensor.permute(1,2,0).detach().cpu().numpy()

# Create Grad-CAM heatmap overlay
cam_heatmap = np.uint8(255 * cam_map)
cam_heatmap = cv2.applyColorMap(cv2.resize(cam_heatmap, (224, 224)), cv2.COLORMAP_JET)
cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(np.uint8(orig_img * 255), 0.6, cam_heatmap, 0.4, 0)

# Get class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
pred_label = class_names[outputs[image_id]]
true_label = class_names[labels[image_id]]

# Show results
if outputs[image_id] == labels[image_id]:
    print(f"Correctly classified as: {pred_label} (Ground truth: {true_label})")
else:
    print(f"Wrongly classified as: {pred_label} (Ground truth: {true_label})")

# Plot images in a 2x2 grid for comparison
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Original Image
axes[0,0].imshow(orig_img)
axes[0,0].set_title(f"Original MRI\n(True: {true_label}, Pred: {pred_label})")
axes[0,0].axis('off')

# LRP Heatmap
axes[0,1].imshow(image_relevances[:,:,0], cmap="seismic", vmin=0, vmax=1)
axes[0,1].set_title("LRP Heatmap")
axes[0,1].axis('off')

# Grad-CAM Heatmap
axes[1,0].imshow(cam_map, cmap="jet", vmin=0, vmax=1)
axes[1,0].set_title("Grad-CAM Heatmap")
axes[1,0].axis('off')

# Grad-CAM Overlay
axes[1,1].imshow(overlay)
axes[1,1].set_title("Grad-CAM Overlay")
axes[1,1].axis('off')

plt.tight_layout()
plt.show()


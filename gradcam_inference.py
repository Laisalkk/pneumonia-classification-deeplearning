import cv2
import numpy as np
import torch

class GradCAMInference:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_layers()

    def _hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        loss = output[0, target_class]
        loss.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        return heatmap

def apply_gradcam(model, target_layer, img_tensor, pred_class, original_img_np):
    """
    Menghasilkan gambar overlay Grad-CAM
    """
    cam = GradCAMInference(model, target_layer)
    heatmap = cam.generate_heatmap(img_tensor, pred_class)
    
    # Resize heatmap ke ukuran gambar asli
    heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    
    # Konversi ke colormap JET
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap di atas gambar asli
    overlayed_img = cv2.addWeighted(original_img_np, 0.6, heatmap_color, 0.4, 0)
    return overlayed_img

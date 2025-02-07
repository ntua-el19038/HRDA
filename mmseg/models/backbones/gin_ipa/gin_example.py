import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import interpolate
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from .adv_bias import AdvBias
from .utils import rescale_intensity



class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(self, out_channel = 32, in_channel = 3, scale_pool = [1, 3], layer_id = 0, use_act = True, requires_grad = False, **kwargs):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock, self).__init__()
        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.scale_pool     = scale_pool
        self.layer_id       = layer_id
        self.use_act        = use_act
        self.requires_grad  = requires_grad
        assert requires_grad == False

    def forward(self, x_in, requires_grad = True):
        """
        Args:
            x_in: [ nb (original), nc (original), nx, ny ]
        """
        # random size of kernel
        # torch.manual_seed(42)  # Replace 42 with any fixed seed
        idx_k = torch.randint(high = len(self.scale_pool), size = (1,))
        k = self.scale_pool[idx_k[0]]

        nb, nc, nx, ny = x_in.shape

        ker = torch.randn([self.out_channel * nb, self.in_channel , k, k  ], requires_grad = self.requires_grad  ).cuda()
        shift = torch.randn( [self.out_channel * nb, 1, 1 ], requires_grad = self.requires_grad  ).cuda() * 1.0

        x_in = x_in.view(1, nb * nc, nx, ny)
        x_conv = F.conv2d(x_in, ker, stride =1, padding = k //2, dilation = 1, groups = nb )
        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny)
        return x_conv


class GINGroupConv(nn.Module):
    def __init__(self, out_channel = 3, in_channel = 3, interm_channel = 2, scale_pool = [1, 3 ], n_layer = 4, out_norm = 'frob', **kwargs):
        '''
        GIN
        '''
        super(GINGroupConv, self).__init__()
        self.scale_pool = scale_pool # don't make it tool large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel

        self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = interm_channel, in_channel = in_channel, scale_pool = scale_pool, layer_id = 0).cuda()
                )
        for ii in range(n_layer - 2):
            self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = interm_channel, in_channel = interm_channel, scale_pool = scale_pool, layer_id = ii + 1).cuda()
                )
        self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = out_channel, in_channel = interm_channel, scale_pool = scale_pool, layer_id = n_layer - 1, use_act = False).cuda()
                )

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim = 0)

        nb, nc, nx, ny = x_in.shape

        alphas = torch.rand(nb)[:, None, None, None] # nb, 1, 1, 1
        alphas = alphas.repeat(1, nc, 1, 1).cuda() # nb, nc, 1, 1

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        mixed = alphas * x + (1.0 - alphas) * x_in

        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim = (-1, -2), p = 'fro', keepdim = False)
            _in_frob = _in_frob[:, None, None, None].repeat(1, nc, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
            _self_frob = _self_frob[:, None, None, None].repeat(1, self.out_channel, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5 ) ) * _in_frob

        return mixed

import torch
from torch import nn

class GINComponent(nn.Module):
    def __init__(self, in_channels, out_channels, interm_channels, scale_pool, n_layers, out_norm='frob'):
        """
        A reusable GIN block to integrate directly into your network.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            interm_channels (int): Number of intermediate channels in GIN.
            scale_pool (list): List of kernel sizes for multi-scale pooling.
            n_layers (int): Number of GIN layers.
            out_norm (str): Normalization type ('frob' or None).
        """
        super(GINComponent, self).__init__()
        self.gin = GINGroupConv(
            in_channel=in_channels,
            out_channel=out_channels,
            interm_channel=interm_channels,
            scale_pool=scale_pool,
            n_layer=n_layers,
            out_norm=out_norm
        )

    def forward(self, x):
        """
        Forward pass for the GIN component.
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width].
        Returns:
            Tensor after GIN processing.
        """
        # if not self.training:
        #     return x

        return self.gin(x)


# Apply GIN to an image
def apply_gin_to_image(image_path):
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Load the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

    # Initialize GIN component
    gin = GINComponent(
        in_channels=3,
        out_channels=3,
        interm_channels=8,
        scale_pool=[1, 3, 5],
        n_layers=4,
        out_norm='frob'
    ).cuda()  # Move GIN component to GPU

    # Apply the GIN component
    output_tensor = gin(input_tensor)

    # Post-process the output to convert back to an image
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # Move to CPU for visualization
    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # Normalize to [0, 1]

    # Display the original and processed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Processed Image (GIN)")
    plt.imshow(output_image)
    plt.axis("off")

    plt.show()
def apply_ipa(input_img, input_buffer, blend_config, blender_node):
    """
    Applies Intensity-Perturbing Augmentation (IPA) to the input images.

    Args:
        input_img (torch.Tensor): The input images tensor.
        input_buffer (torch.Tensor): Transformed images from GIN.
        blend_config (dict): Configuration for blending augmentation.
        blender_node (AdvBias): The blender node for applying IPA.

    Returns:
        torch.Tensor: Augmented images with IPA applied.
        torch.Tensor: The blend mask used in the transformation.
    """
    # Initialize blender node parameters
    blender_node.init_parameters()

    # Generate the blend mask and rescale its intensity
    blend_mask = rescale_intensity(blender_node.bias_field).repeat(1, 3, 1, 1)

    # Spatially-variable blending for augmented copies
    nb_current = input_img.shape[0]
    input_cp1 = (
        input_buffer[:nb_current].clone().detach() * blend_mask
        + input_buffer[nb_current:nb_current * 2].clone().detach() * (1.0 - blend_mask)
    )
    input_cp2 = (
        input_buffer[:nb_current] * (1.0 - blend_mask)
        + input_buffer[nb_current:nb_current * 2] * blend_mask
    )

    # Replace original buffers with blended copies
    input_buffer[:nb_current] = input_cp1
    input_buffer[nb_current:nb_current * 2] = input_cp2

    return input_buffer, blend_mask.data
def apply_gin_ipa_to_image(image_path):
    """
    Applies GIN and IPA augmentations to an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        None
    """
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.286, 0.325, 0.283], std=[0.186, 0.190, 0.187])
    ])
    # image = Image.open(image_path).convert("RGB")

    # input_tensor = transform(image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU
    input_tensor=image_path
    # Extract input data size
    bs, in_channels, height, width = input_tensor.shape
    ipa_config = {
        "epsilon": 0.1,
        "xi": 0,
        "control_point_spacing": [4, 4],
        "downscale": 1,
        "data_size": [bs, in_channels, height, width],
        "interpolation_order": 3,
        "init_mode": 'gaussian',
        "space": 'log'
    }
    # Dynamically configure GIN component based on input size
    # Initialize GIN component
    gin = GINComponent(
        in_channels=3,
        out_channels=3,
        interm_channels=8,
        scale_pool=[1, 3, 5],
        n_layers=4,
        out_norm='frob'
    ).cuda()  # Move GIN component to GPU

    # Apply the GIN component
    gin_output_tensor = gin(input_tensor)

    # Ensure GIN output has the same dimensions as the input tensor
    gin_output_resized = interpolate(
        gin_output_tensor,
        size=(height, width),
        mode='bilinear',
        align_corners=False
    )

    # Initialize IPA component
    blender_node = AdvBias(ipa_config).cuda()
    blender_node.init_parameters()

    # Generate the blend mask and resize it to match input dimensions
    blend_mask = rescale_intensity(blender_node.bias_field).repeat(1, in_channels, 1, 1)
    blend_mask_resized = interpolate(
        blend_mask,
        size=(height, width),
        mode='bilinear',
        align_corners=False
    )
    a=0.3
    # Apply IPA augmentation
    ipa_output_tensor = gin_output_resized * blend_mask_resized + input_tensor * (1.0 - blend_mask_resized)
    # ipa_output_tensor_reblend=ipa_output_tensor*a+input_tensor*(1-a)
    ipa_output_tensor=ipa_output_tensor*a+input_tensor*(1-a)
    return ipa_output_tensor
    # Post-process the outputs to convert back to images
    def tensor_to_image(tensor):
        tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # Move to CPU and rearrange dimensions
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
        return tensor

    gin_output_image = tensor_to_image(gin_output_resized)
    ipa_output_image = tensor_to_image(ipa_output_tensor)
    ipa_output_image_reblend=tensor_to_image(ipa_output_tensor_reblend)
    # Display the original and processed images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(1, 4, 2)
    plt.title("Processed Image (GIN)")
    plt.imshow(gin_output_image)
    plt.axis("off")
    plt.subplot(1, 4, 3)
    plt.title("Processed Image (GIN + IPA)")
    plt.imshow(ipa_output_image)
    plt.axis("off")
    plt.axis("off")
    plt.subplot(1, 4, 4)
    plt.title("Reblended Image")
    plt.imshow(ipa_output_image_reblend)
    plt.axis("off")
    plt.show()
# Apply GIN and IPA multiple times and visualize all results
def apply_gin_ipa_multiple(image_path, num_iterations=3):
    """
    Applies GIN and IPA augmentations multiple times to an image and displays results.

    Args:
        image_path (str): Path to the input image.
        num_iterations (int): Number of times to apply GIN+IPA.
    """
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.286, 0.325, 0.283], std=[0.186, 0.190, 0.187])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).cuda()

    # Extract input data size
    bs, in_channels, height, width = input_tensor.shape
    ipa_config = {
        "epsilon": 0.1,
        "xi": 0,
        "control_point_spacing": [4, 4],
        "downscale": 1,
        "data_size": [bs, in_channels, height, width],
        "interpolation_order": 3,
        "init_mode": 'gaussian',
        "space": 'log'
    }

    # Initialize GIN component
    gin = GINComponent(
        in_channels=3,
        out_channels=3,
        interm_channels=8,
        scale_pool=[1, 3, 5],
        n_layers=4,
        out_norm='frob'
    ).cuda()

    results = [image]  # Store images for visualization
    processed_tensor = input_tensor.clone()

    for _ in range(num_iterations):
        # Apply GIN
        gin_output_tensor = gin(processed_tensor)
        gin_output_resized = interpolate(gin_output_tensor, size=(height, width), mode='bilinear', align_corners=False)

        # Apply IPA
        blender_node = AdvBias(ipa_config).cuda()
        blender_node.init_parameters()
        blend_mask = rescale_intensity(blender_node.bias_field).repeat(1, in_channels, 1, 1)
        blend_mask_resized = interpolate(blend_mask, size=(height, width), mode='bilinear', align_corners=False)
        ipa_output_tensor = gin_output_resized * blend_mask_resized + input_tensor * (1.0 - blend_mask_resized)
        ipa_output_tensor = ipa_output_tensor * 0.2 + input_tensor * 0.8
        processed_tensor = ipa_output_tensor.clone()  # Update for next iteration

        # Convert tensor to image
        output_image = ipa_output_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
        results.append(output_image)

    # Display results
    plt.figure(figsize=(5, 1))
    for i, img in enumerate(results):
        plt.subplot(1, num_iterations + 1, i + 1)
        plt.title(f"Step {i}")
        plt.imshow(img)
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Path to the input image
    image_path = "./data/cityscapes/leftImg8bit/test/berlin_000000_000019_leftImg8bit.png"  # Replace with the actual image path
    apply_gin_ipa_to_image(image_path)
    # image_path = "./data/cityscapes/leftImg8bit/test/berlin_000000_000019_leftImg8bit.png"
    # apply_gin_ipa_multiple(image_path, num_iterations=5)
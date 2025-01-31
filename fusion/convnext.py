import logging

import torch
import torch.nn as nn
from transformers import ConvNextV2Config, ConvNextV2Model, PreTrainedModel

logger = logging.getLogger(__name__)


class ConvNextV2Classifier(PreTrainedModel):
    config_class = ConvNextV2Config
    main_input_name = 'pixel_values'
    _no_split_modules = ['ConvNextV2Stage']

    def __init__(self, config: ConvNextV2Config, vision_output_size=1):
        super().__init__(config)

        self.vision_model = ConvNextV2Model(config)
        self.vision_model = self.vision_model.to(torch.bfloat16)
        self.img_size = 224
        hidden_size = config.hidden_sizes[-1]  # Use the last stage's hidden size

        # Calculate the output dimension after global average pooling
        # ConvNeXtV2 already does average pooling, so we just need to handle the flattened features
        self.fc = nn.Linear(hidden_size, vision_output_size)

        # Initialize weights
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                m.bias.data.zero_()

        # Required attributes to match InternVLChatModel interface
        self.vision_only = True
        self.num_image_token = 1  # ConvNeXtV2 uses global average pooling

        logger.info(f'Initialized ConvNextV2Classifier with output size: {vision_output_size}')

    def forward_vision(self, pixel_values, attention_mask=None, classify=True, **kwargs):
        """
        Forward pass through the vision model.

        Args:
            pixel_values (torch.Tensor): Input images of shape (batch_size, num_frames, channels, height, width)
            attention_mask (torch.Tensor, optional): Attention mask for frame aggregation
            classify (bool): Whether to apply classification head

        Returns:
            torch.Tensor: Either features or classification logits
        """
        if len(pixel_values.shape) == 4:
            pixel_values = pixel_values.unsqueeze(1) # Add frame dimension
        b, n, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(b * n, c, h, w)

        # Forward through ConvNeXtV2
        outputs = self.vision_model(pixel_values)
        features = outputs.pooler_output  # Shape: (b*n, hidden_size)

        # Reshape back to include frame dimension
        features = features.view(b, n, -1)  # Shape: (b, n, hidden_size)

        if attention_mask is not None:
            # Apply temporal attention mask
            attention_mask = attention_mask.to(features.dtype)
            mask = attention_mask.unsqueeze(-1)
            features = features * mask
            features_sum = features.sum(dim=1)
            mask_sum = mask.sum(dim=1).clamp(min=1.0)
            features = features_sum / mask_sum
        else:
            # Simple average pooling over frames
            features = features.mean(dim=1)

        if classify:
            return self.classify(features)
        return features

    def classify(self, features):
        """
        Apply classification head to features.

        Args:
            features (torch.Tensor): Input features

        Returns:
            torch.Tensor: Classification logits
        """
        return self.fc(features)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, vision_output_size=1, *model_args, **kwargs):
        """
        Load pretrained ConvNeXtV2 model and add classification head.

        Args:
            pretrained_model_name_or_path (str): HuggingFace model name or path
            vision_output_size (int): Number of output classes
            *model_args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            ConvNextV2Classifier: Initialized model
        """
        config = ConvNextV2Config.from_pretrained(pretrained_model_name_or_path)
        model = cls(config, vision_output_size=vision_output_size, *model_args, **kwargs)

        # Load pretrained weights
        state_dict = ConvNextV2Model.from_pretrained(
            pretrained_model_name_or_path
        ).state_dict()

        # Load only the vision model weights
        missing_keys, unexpected_keys = model.vision_model.load_state_dict(state_dict, strict=False)
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys}")

        return model

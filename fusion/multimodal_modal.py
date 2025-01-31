import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Union, Optional
from fusion.convnext import ConvNextV2Classifier
from fusion.ecg_jepa import load_encoder


class MultimodalMedicalModel(nn.Module):
    def __init__(
            self,
            text_model_name: str = "medicalai/ClinicalBERT",
            convnext_checkpoint: str = "/mnt/8T/convnext/model_56000.pt",
            ecg_checkpoint: str = "/mnt/8T/ecg/ecg_encoder.pth",
            output_size: int = 100,
            device: str = "cuda",
            fusion_technique: str = "concatenate",
            task: str = "diagnosis_classification"
    ):
        super().__init__()
        self.device = device

        # Patch: If there exists /scratch folder, then change the path to /scratch
        if os.path.exists("/scratch"):
            convnext_checkpoint = convnext_checkpoint.replace("/mnt/8T", "/scratch")
            ecg_checkpoint = ecg_checkpoint.replace("/mnt/8T", "/scratch")

        # Initialize text encoder
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name).to(device)
        self.text_hidden_size = self.text_encoder.config.hidden_size

        self.ecg_encoder = load_encoder(ecg_checkpoint)[0].to(device)
        self.ecg_hidden_size = 768

        # Initialize image encoder (ConvNextV2)
        self.image_encoder = ConvNextV2Classifier.from_pretrained(
            "facebook/convnextv2-base-22k-224"
        ).to(device)

        # Load pretrained weights
        if convnext_checkpoint != 'none':
            checkpoint = torch.load(convnext_checkpoint, map_location=device)
            # drop parameters starting with "fc"
            checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("fc")}
            self.image_encoder.load_state_dict(checkpoint, strict=False)
        self.image_hidden_size = self.image_encoder.vision_model.config.hidden_sizes[-1]

        # Fusion layer
        self.cross_attention = MultiHeadCrossAttention(self.text_hidden_size, self.image_hidden_size, self.ecg_hidden_size)
        self.combined_hidden_size = self.text_hidden_size + self.image_hidden_size + self.ecg_hidden_size

        # Concatenate features
        if fusion_technique == "concatenate":
            self.head = self.create_mlp(dim=self.combined_hidden_size, task=task)
        else:  # cross attention
            self.head = self.create_mlp(dim=self.cross_attention.hidden_dim, task=task)
        
        # for late stage fusion
        self.text_head = self.create_mlp(dim=self.text_hidden_size, task=task)
        self.image_head = self.create_mlp(dim=self.image_hidden_size, task=task)
        self.ecg_head = self.create_mlp(dim=self.ecg_hidden_size, task=task)

    def create_mlp(self, dim, task: str = "diagnosis_classification"):
        if task == "diagnosis_classification":
            return nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim // 2, 100)
            )
        elif task == "survival_classification":
            return nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim // 2, 1),  # Output: binary classification
            )
        else: # length of stay regression
            return nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim // 2, dim // 8),
                nn.GELU(),  # Ensure positive LOS predictions
                nn.Linear(dim // 8, 1)
            )


    def encode_text(self, text_data: Union[str, List[str]]) -> torch.Tensor:
        """Encode text data using the text encoder."""
        inputs = self.text_tokenizer(
            text_data,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # Use CLS token embedding
            return outputs.last_hidden_state[:, 0, :]

    def format_patient_history(self, patient_data: Dict) -> str:
        """Format patient history into a text string."""
        text_parts = []

        # Add demographics
        demo = patient_data.get('demographics', {})
        text_parts.append(f"Patient demographics: {demo.get('gender', 'Unknown')} "
                          f"age {demo.get('anchor_age', 'Unknown')}")

        admissions = patient_data.get('admissions', [])
        for admission in admissions:
            text_parts.append(f"Admission: {admission['admission_type']} at "
                              f"{admission['admission_location']}")

        # Add lab values summary
        lab_values = patient_data.get('lab_values', {})
        for test_name, values in lab_values.items():
            if values:
                recent_value = values[-1]  # Most recent value
                text_parts.append(f"{test_name}: {recent_value['value']} "
                                  f"{recent_value.get('unit', '')}")

        # Add chart values summary
        chart_values = patient_data.get('chart_values', {})
        for measure_name, values in chart_values.items():
            if values:
                recent_value = values[-1]
                text_parts.append(f"{measure_name}: {recent_value['value']} "
                                  f"{recent_value.get('unit', '')}")

        final_text = " | ".join(text_parts)
        return final_text

    def forward(self,
                task: str,
                images: torch.Tensor,
                patient_histories: List[Dict],
                ecg: Optional[torch.Tensor] = None,
                fusion_technique: str = "concatenate",
                fusion_stage: str = "early",
                ) -> torch.Tensor:
        """
        Forward pass through the multimodal model.

        Args:
            images: Tensor of shape (batch_size, num_images, channels, height, width)
            patient_histories: List of patient history dictionaries

        Returns:
            logits: Tensor of shape (batch_size, output_size)
        """
        # Encode images
        image_features = self.image_encoder.forward_vision(
            images,
            classify=False
        )

        # Format and encode text
        text_data = [self.format_patient_history(hist) for hist in patient_histories]
        text_features = self.encode_text(text_data)

        if ecg is not None:
            ecg_features = self.ecg_encoder(ecg.float())[0]
            ecg_features = ecg_features.mean(dim=1)
        else:
            ecg_features = torch.zeros(image_features.shape[0], self.ecg_hidden_size).to(self.device)

        if fusion_stage == "early":
            # Concatenate features
            if fusion_technique == "concatenate":
                combined_features = torch.cat([image_features, text_features, ecg_features], dim=1)
            else:  # cross attention
                combined_features = self.cross_attention(text_features, image_features, ecg_features)

            logits = self.head(combined_features)

        else: # late
            text_logits = self.text_head(text_features.float())
            image_logits = self.image_head(image_features.float())
            ecg_logits = self.ecg_head(ecg_features.float())

            logits = (text_logits + image_logits + ecg_logits) / 3  # Average the logits

        if task == "survival_classification":
            logits = logits.squeeze(1)

        return logits

class CrossAttention(nn.Module):
    def __init__(self,
                 text_hidden_size,
                 image_hidden_size,
                 ecg_hidden_size,
                 hidden_dim):

        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim

        # Linear projections for Q, K, V
        self.query = nn.Linear(text_hidden_size, hidden_dim)
        self.key = nn.Linear(image_hidden_size + ecg_hidden_size, hidden_dim)
        self.value = nn.Linear(image_hidden_size + ecg_hidden_size, hidden_dim)
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

        # # Output projection
        # self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text_features, image_features, ecg_features):
        Q = self.query(text_features)
        image_ecg_concatenated = torch.cat([image_features, ecg_features], dim=-1)
        K = self.key(image_ecg_concatenated)
        V = self.value(image_ecg_concatenated)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, 1, combined_size)
        attn_weights = self.softmax(attn_scores)  # (batch_size, 1, combined_size)

        # Compute weighted sum of values (V)
        fused_output = torch.matmul(attn_weights, V)  # (batch_size, 1, hidden_dim)
        fused_output = fused_output.squeeze(1)  # Remove the singleton dimension (batch_size, hidden_dim)

        # # Final output projection (if needed)
        # output = self.out_proj(fused_output)  # (batch_size, hidden_dim)

        return fused_output
    
class MultiHeadCrossAttention(nn.Module):
    """ Multiple heads of cross-attention in parallel """

    def __init__(self, text_hidden_size,
                 image_hidden_size,
                 ecg_hidden_size, 
                 num_heads = 8, 
                 hidden_dim = 512):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttention(text_hidden_size, image_hidden_size, ecg_hidden_size,hidden_dim) for _ in range(num_heads)])
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, text_features, image_features, ecg_features):
        head_outputs = [h(text_features, image_features, ecg_features) for h in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.out_proj(concatenated)
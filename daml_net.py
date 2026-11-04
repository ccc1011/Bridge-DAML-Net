"""
DAML-Net model definition.

A shared encoder feeds two task-specific branches. Each branch extracts local
features with a TextCNN and combines them with the CLS feature in a gated fusion
block. Task 2 (component) runs first, and its predicted probabilities are passed
to Task 1 (rating) with the gradient detached.

Training has three stages:
  1. epoch < task2_freeze_epochs: the Task 2 loss is excluded and Task 1 receives
     a uniform vector.
  2. task2_freeze_epochs <= epoch < task2_feature_warmup_epochs: both losses are
     used and Task 1 still receives the uniform vector.
  3. epoch >= task2_feature_warmup_epochs: Task 1 receives the Task 2 probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TaskSpecificTextCNN(nn.Module):
    """TextCNN that extracts local features for one task."""

    def __init__(self, input_dim=768, num_filters=100, filter_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()

        self.output_dim = num_filters * len(filter_sizes)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, input_dim))
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)

        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def forward(self, sequence_output):
        # [batch, 1, seq_len, hidden_dim]
        x = sequence_output.unsqueeze(1)

        conv_outputs = []
        for conv in self.convs:
            # [batch, num_filters, seq_len - k + 1]
            conv_out = torch.relu(conv(x)).squeeze(3)
            # [batch, num_filters]
            conv_outputs.append(F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2))

        return self.dropout(torch.cat(conv_outputs, dim=1))


class TaskSpecificFusionBlock(nn.Module):
    """Gated fusion of the CLS feature, TextCNN feature and optional auxiliary input."""

    def __init__(self, bert_dim=768, cnn_dim=300, fusion_dim=512, dropout=0.3,
                 use_auxiliary_features=False, auxiliary_dim=3):
        super().__init__()

        self.use_auxiliary_features = use_auxiliary_features

        concat_dim = bert_dim + cnn_dim
        if use_auxiliary_features:
            concat_dim += auxiliary_dim

        # Linear fusion branch
        self.linear_fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Projections to a common dimension
        self.bert_projection = nn.Sequential(
            nn.Linear(bert_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        if use_auxiliary_features:
            self.auxiliary_projection = nn.Sequential(
                nn.Linear(auxiliary_dim, fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        # Gate weights in the range 0 to 1
        self.gate_generator = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.Sigmoid()
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, cls_features, cnn_features, auxiliary_features=None):
        if self.use_auxiliary_features:
            F_linear = self.linear_fusion(
                torch.cat([cls_features, cnn_features, auxiliary_features], dim=-1)
            )
        else:
            F_linear = self.linear_fusion(torch.cat([cls_features, cnn_features], dim=-1))

        cls_proj = self.bert_projection(cls_features)
        cnn_proj = self.cnn_projection(cnn_features)

        if self.use_auxiliary_features:
            auxiliary_proj = self.auxiliary_projection(auxiliary_features)
            gate_weight = self.gate_generator(F_linear)
            F_sum = gate_weight * cnn_proj + (1 - gate_weight) * (cls_proj + auxiliary_proj) / 2
        else:
            gate_weight = self.gate_generator(F_linear)
            F_sum = gate_weight * cnn_proj + (1 - gate_weight) * cls_proj

        # Residual connection
        return F_sum + F_linear


class DAMLNet(nn.Module):
    """Shared encoder with cascaded task-specific branches."""

    def __init__(self, bert_model_name, num_classes_task1=5, num_classes_task2=3,
                 task1_weight=0.8, cnn_num_filters=100, cnn_filter_sizes=(3, 4, 5),
                 fusion_dim=512, dropout=0.3, task2_feature_warmup_epochs=3,
                 task2_freeze_epochs=1):
        super().__init__()

        if task2_freeze_epochs > task2_feature_warmup_epochs:
            raise ValueError("task2_freeze_epochs cannot exceed task2_feature_warmup_epochs")

        self.num_classes_task2 = num_classes_task2
        self.task1_weight = task1_weight
        self.task2_weight = 1.0 - task1_weight
        self.task2_feature_warmup_epochs = task2_feature_warmup_epochs
        self.task2_freeze_epochs = task2_freeze_epochs
        self.current_epoch = 0

        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size

        # Task 2 (component) branch
        self.textcnn_task2 = TaskSpecificTextCNN(
            input_dim=bert_dim,
            num_filters=cnn_num_filters,
            filter_sizes=cnn_filter_sizes,
            dropout=dropout
        )
        self.fusion_task2 = TaskSpecificFusionBlock(
            bert_dim=bert_dim,
            cnn_dim=self.textcnn_task2.output_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            use_auxiliary_features=False
        )
        self.classifier_task2 = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes_task2)
        )

        # Task 1 (rating) branch
        self.textcnn_task1 = TaskSpecificTextCNN(
            input_dim=bert_dim,
            num_filters=cnn_num_filters,
            filter_sizes=cnn_filter_sizes,
            dropout=dropout
        )
        self.fusion_task1 = TaskSpecificFusionBlock(
            bert_dim=bert_dim,
            cnn_dim=self.textcnn_task1.output_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            use_auxiliary_features=True,
            auxiliary_dim=num_classes_task2
        )
        self.classifier_task1 = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes_task1)
        )

    def set_epoch(self, epoch):
        """Update the current epoch (0-indexed). Call this at the start of each epoch."""
        self.current_epoch = epoch

    def get_effective_task2_weight(self):
        """Return the Task 2 loss weight for the current epoch."""
        return 0.0 if self.current_epoch < self.task2_freeze_epochs else self.task2_weight

    def forward(self, input_ids, attention_mask=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        cls_features = bert_outputs.pooler_output

        # Task 2 runs first so that its prediction can inform Task 1
        cnn_features_task2 = self.textcnn_task2(sequence_output)
        fused_features_task2 = self.fusion_task2(cls_features, cnn_features_task2)
        logits_task2 = self.classifier_task2(fused_features_task2)

        task2_probabilities = F.softmax(logits_task2, dim=-1).detach()

        if self.current_epoch < self.task2_feature_warmup_epochs:
            task1_auxiliary_input = torch.full_like(task2_probabilities, 1.0 / self.num_classes_task2)
        else:
            task1_auxiliary_input = task2_probabilities

        cnn_features_task1 = self.textcnn_task1(sequence_output)
        fused_features_task1 = self.fusion_task1(
            cls_features, cnn_features_task1, auxiliary_features=task1_auxiliary_input
        )
        logits_task1 = self.classifier_task1(fused_features_task1)

        return logits_task1, logits_task2

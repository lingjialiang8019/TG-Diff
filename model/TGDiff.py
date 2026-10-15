"""Reserved model interface. Implementation is withheld until acceptance."""
from torch import nn


class TGDiff(nn.Module):
    """Predict clean-mask logits from conditional inputs."""

    def __init__(self, bert_name="bert-base-uncased", initialize_pretrained=True,
                 bert_config=None):
        super().__init__()
        raise NotImplementedError(
            "Model implementation will be released after acceptance. "
            "This repository currently provides the surrounding workflow only."
        )

    def forward(self, image, input_ids, attention_mask, noisy_mask, timesteps):
        """Return logits with shape [batch, 1, height, width]."""
        raise NotImplementedError("Model implementation is not yet public.")

    def export_text_config(self):
        """Return the text encoder configuration for checkpoint reconstruction."""
        raise NotImplementedError("Model implementation is not yet public.")

import torch
import torch.nn as nn
import torchvision.models as models


def get_model(num_classes=3, pretrained=True, freeze=True):
    """
    Function that returns the ResNet34FineTuner model with default parameters.

    Args:
        num_classes (int): Number of classes in the classification.
        pretrained (bool): Use pretrained weights from ImageNet if True.
        freeze (bool): If True, freeze all layers except the last fully connected layer.
    """
    model = ResNet34FineTuner(
        num_classes=num_classes, pretrained=pretrained, freeze=freeze
    )
    return model


class ResNet34FineTuner(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze=True):
        """
        ResNet-34 model for Fine-Tuning.

        Args:
            num_classes (int): Number of classes in the classification.
            pretrained (bool): Use pretrained weights from ImageNet if True.
            freeze (bool): If True, freeze all layers except the last fully connected layer.
        """
        super(ResNet34FineTuner, self).__init__()
        self.model = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )
        # Freeze all layers except the last fully connected layer (fc)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False  # Freeze all layers
            # Unfreeze the last fully connected layer
            for param in self.model.fc.parameters():
                param.requires_grad = True  # Unfreeze the final layer
        # Replace the last layer with the desired number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


def checktest():
    model = get_model()
    batch_size = 10
    dummy_input = torch.randn(batch_size, 3, 224, 224)  # [B,CH,W,H]
    logits = model(dummy_input)
    print(f"Output shape: {logits.shape}")


if __name__ == "__main__":
    print("Check if the model load")
    checktest()
    print("Done.")

"""Tests for activation capture."""

import pytest
import torch
import torch.nn as nn

from app.models.activation_capture import ActivationCapture


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


def test_activation_capture_initialization():
    """Test ActivationCapture initializes correctly."""
    model = SimpleModel()
    capture = ActivationCapture(model, layer_names=["layer1"])

    assert len(capture.layer_names) == 1
    assert capture.layer_names[0] == "layer1"


def test_activation_capture_context_manager():
    """Test ActivationCapture as context manager."""
    model = SimpleModel()
    capture = ActivationCapture(model, layer_names=["layer1"])

    x = torch.randn(4, 10)

    with capture:
        _ = model(x)

    activations = capture.get_activations()

    assert "layer1" in activations
    assert activations["layer1"].shape[0] == 4  # batch size
    assert activations["layer1"].shape[1] == 20  # layer1 output dim


def test_activation_capture_multiple_layers():
    """Test capturing from multiple layers."""
    model = SimpleModel()
    capture = ActivationCapture(model, layer_names=["layer1", "layer2"])

    x = torch.randn(4, 10)

    with capture:
        _ = model(x)

    activations = capture.get_activations()

    assert "layer1" in activations
    assert "layer2" in activations


def test_activation_capture_clear():
    """Test clearing activations."""
    model = SimpleModel()
    capture = ActivationCapture(model, layer_names=["layer1"])

    x = torch.randn(4, 10)

    with capture:
        _ = model(x)

    assert len(capture.activations) > 0

    capture.clear_activations()

    assert len(capture.activations) == 0

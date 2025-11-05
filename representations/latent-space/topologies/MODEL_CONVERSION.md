# Model Conversion Guide — Core ML & TensorFlow Lite

Complete guide for converting sentence transformer models to on-device formats for iOS (Core ML) and Android (TensorFlow Lite).

---

## Overview

**Goal**: Run embedding inference on-device without network dependency.

**Challenges**:
- Model size (original ~90MB → target <50MB)
- Inference speed (<50ms per text)
- Maintaining accuracy (cosine sim correlation >0.95)

**Strategy**:
1. Export from PyTorch → ONNX
2. Quantize (FP32 → INT8 or FP16)
3. Convert to platform format (Core ML / TFLite)
4. Validate embedding quality

---

## Prerequisites

```bash
# Install conversion tools
pip install transformers torch onnx coremltools tensorflow

# Verify installations
python -c "import torch, onnx, coremltools, tensorflow; print('✓ All installed')"
```

---

## Part 1: Export to ONNX (Universal Format)

### Script: `scripts/export_to_onnx.py`

```python
#!/usr/bin/env python3
"""
Export sentence transformer to ONNX format.
Example:
  python scripts/export_to_onnx.py \
    --model all-MiniLM-L6-v2 \
    --output models/minilm.onnx \
    --opset 14
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def export_to_onnx(model_name: str, output_path: str, opset_version: int = 14):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Dummy input for tracing
    text = "This is a sample sentence"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Export
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path,
        opset_version=opset_version,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"}
        }
    )

    print(f"✓ Exported to {output_path}")
    print(f"Model size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--output", default="models/minilm.onnx")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    export_to_onnx(args.model, args.output, args.opset)
```

---

## Part 2: Convert to Core ML (iOS)

### Script: `scripts/convert_to_coreml.py`

```python
#!/usr/bin/env python3
"""
Convert ONNX model to Core ML format with quantization.
Example:
  python scripts/convert_to_coreml.py \
    --onnx models/minilm.onnx \
    --output models/minilm_fp16.mlmodel \
    --quantize fp16
"""
import argparse
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

def convert_to_coreml(onnx_path: str, output_path: str, quantize: str = "fp16"):
    print(f"Converting {onnx_path} to Core ML...")

    # Convert ONNX → Core ML
    model = ct.converters.onnx.convert(
        model=onnx_path,
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16 if quantize == "fp16" else ct.precision.FLOAT32
    )

    # Optional: Further quantization
    if quantize == "int8":
        print("Applying INT8 quantization...")
        model = quantization_utils.quantize_weights(model, nbits=8)

    # Add metadata
    model.short_description = "Sentence embedding model for Latent Topologies"
    model.author = "Latent Topologies Project"
    model.license = "CC BY-NC 4.0"

    # Save
    model.save(output_path)
    print(f"✓ Saved Core ML model to {output_path}")

    # Size comparison
    import os
    original_size = os.path.getsize(onnx_path) / 1024 / 1024
    coreml_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"Size: {original_size:.2f} MB (ONNX) → {coreml_size:.2f} MB (Core ML)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Input ONNX model")
    parser.add_argument("--output", required=True, help="Output Core ML model")
    parser.add_argument("--quantize", choices=["none", "fp16", "int8"], default="fp16")
    args = parser.parse_args()

    convert_to_coreml(args.onnx, args.output, args.quantize)
```

### Using Core ML in React Native

```typescript
// iOS integration via native module
import { NativeModules } from 'react-native';

const { EmbeddingModel } = NativeModules;

async function embed(text: string): Promise<number[]> {
  const embedding = await EmbeddingModel.encode(text);
  return embedding; // Float32Array → number[]
}
```

---

## Part 3: Convert to TensorFlow Lite (Android)

### Script: `scripts/convert_to_tflite.py`

```python
#!/usr/bin/env python3
"""
Convert ONNX model to TensorFlow Lite format.
Example:
  python scripts/convert_to_tflite.py \
    --onnx models/minilm.onnx \
    --output models/minilm_int8.tflite \
    --quantize int8
"""
import argparse
import tensorflow as tf
from onnx_tf.backend import prepare

def convert_to_tflite(onnx_path: str, output_path: str, quantize: str = "int8"):
    print(f"Converting {onnx_path} to TFLite...")

    # ONNX → TensorFlow
    import onnx
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("temp_tf_model")

    # TensorFlow → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("temp_tf_model")

    # Apply quantization
    if quantize == "int8":
        print("Applying INT8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
    elif quantize == "fp16":
        print("Applying FP16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Save
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"✓ Saved TFLite model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantize", choices=["none", "fp16", "int8"], default="int8")
    args = parser.parse_args()

    convert_to_tflite(args.onnx, args.output, args.quantize)
```

---

## Part 4: Validation & Quality Assurance

### Script: `scripts/validate_conversion.py`

```python
#!/usr/bin/env python3
"""
Validate converted model quality by comparing embeddings.
Example:
  python scripts/validate_conversion.py \
    --original all-MiniLM-L6-v2 \
    --coreml models/minilm_fp16.mlmodel \
    --test-texts data/test_sentences.txt
"""
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr, spearmanr

def validate_conversion(original_model: str, converted_path: str, test_texts: list):
    print("Loading original model...")
    original = SentenceTransformer(original_model)
    original_embs = original.encode(test_texts, convert_to_numpy=True)

    # Load converted model (implementation depends on format)
    # For Core ML, this would use coremltools.models.MLModel
    # For TFLite, use tf.lite.Interpreter

    print("Computing embeddings with converted model...")
    # converted_embs = load_and_encode(converted_path, test_texts)

    # Compare embeddings
    print("\nValidation Metrics:")
    print("="*60)

    # Cosine similarity correlation
    # cosine_corr = compute_cosine_correlation(original_embs, converted_embs)
    # print(f"Cosine similarity correlation: {cosine_corr:.4f}")

    # Pearson correlation (per dimension)
    # pearson = pearsonr(original_embs.flatten(), converted_embs.flatten())[0]
    # print(f"Pearson correlation: {pearson:.4f}")

    # Neighbor preservation (top-k accuracy)
    # neighbor_acc = compute_neighbor_preservation(original_embs, converted_embs, k=10)
    # print(f"Neighbor preservation @10: {neighbor_acc:.2%}")

    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--converted", required=True)
    parser.add_argument("--test-texts", default="data/test_sentences.txt")
    args = parser.parse_args()

    # Load test texts
    with open(args.test_texts) as f:
        texts = [line.strip() for line in f if line.strip()]

    validate_conversion(args.original, args.converted, texts)
```

---

## Performance Targets

| Platform | Format | Size | Inference | Accuracy |
|----------|--------|------|-----------|----------|
| iOS | Core ML FP16 | <50 MB | <30ms | >0.98 cosine corr |
| Android | TFLite INT8 | <25 MB | <50ms | >0.95 cosine corr |

---

## Deployment Checklist

- [ ] Export original model to ONNX
- [ ] Convert to Core ML (FP16) and TFLite (INT8)
- [ ] Validate embedding quality (correlation >0.95)
- [ ] Test inference speed on target devices
- [ ] Integrate with React Native via native modules
- [ ] Bundle models with app (or download on first launch)
- [ ] Add model card / license info in app

---

## Troubleshooting

**ONNX export fails**:
- Check opset version compatibility (use 12-14)
- Ensure dynamic axes are correctly specified

**Core ML conversion errors**:
- Update coremltools to latest version
- Some ONNX ops may not be supported—check compatibility matrix

**TFLite quantization reduces quality**:
- Use FP16 instead of INT8
- Provide representative dataset for calibration

**Inference too slow**:
- Profile with Xcode Instruments (iOS) or Android Profiler
- Consider smaller model (e.g., all-MiniLM-L3-v2)
- Use GPU acceleration (Core ML ANE, TFLite GPU delegate)

---

## Alternative: Pre-computed Embeddings

For datasets <10k items, consider **shipping pre-computed embeddings** instead of model:

**Pros**:
- No inference latency
- Smaller app size (vectors + coords only)
- Simpler implementation

**Cons**:
- Can't embed user-generated text
- No dynamic corpus expansion

**Hybrid approach**: Ship pre-computed embeddings + lightweight model for Creator Mode.

---

## References

- **Core ML**: https://apple.github.io/coremltools/
- **TensorFlow Lite**: https://www.tensorflow.org/lite
- **ONNX**: https://onnx.ai/
- **Sentence Transformers Export**: https://www.sbert.net/docs/usage/semantic_search.html#export-to-onnx

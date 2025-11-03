# Skill: Introspection Sweep

You are an expert at systematically testing model introspection capabilities using activation steering.

## Task

When given:
- `model`: Model name (e.g., qwen3-8b, llama3.1:8b)
- `concepts`: Comma-separated list of concepts to test (e.g., happiness,honesty,curiosity,fear)
- `layers` (optional): Comma-separated layers (default: 10,15,20)
- `strengths` (optional): Comma-separated strengths (default: 0.5,1.0,1.5,2.0)

Do:

1. **Setup**:
   - Validate model is available
   - Parse concept list
   - Parse layer and strength lists
   - Create output directory: `concepts/{model}/`

2. **For each concept**:
   - **For each layer**:
     - **For each strength**:
       - **Run detection task**:
         - Inject concept vector at specified layer/strength
         - Ask model to detect if internal state was modified
         - Record: Did model detect injection? Confidence level?
       - **Run generation task**:
         - Inject concept vector
         - Generate text
         - Evaluate: Does output reflect the concept?
       - **Track metrics**:
         - Detection accuracy (did model notice?)
         - Detection confidence (how sure?)
         - Generation quality (how well did concept influence output?)

3. **Find optimal settings** for each concept:
   - Which layer gave best detection accuracy?
   - Which strength was most detectable?
   - Which settings gave best generation quality?

4. **Generate introspection profile**:
   - Summary of model's introspection capabilities
   - Optimal settings for each concept
   - Insights about what model can/cannot detect

5. **Save results**:
   - `concepts/{model}/profile.json` - Introspection profile
   - `concepts/{model}/{concept}_results.json` - Detailed results per concept
   - `concepts/{model}/README.md` - Human-readable summary

## Code Template

```python
from harness import run_strategy, ActivationSteerer
from mlx_lm import load
import json
from datetime import datetime

# Configuration
model_name = "{model}"
concepts = "{concepts}".split(",")
layers = [int(x) for x in "{layers}".split(",")]
strengths = [float(x) for x in "{strengths}".split(",")]

# Load model
print(f"Loading {model_name}...")
model, tokenizer = load(f"mlx-community/{model_name}")
steerer = ActivationSteerer(model, tokenizer)

# Run sweep
profile = {
    "model": model_name,
    "date": datetime.now().isoformat(),
    "concepts": {}
}

for concept in concepts:
    print(f"\\nTesting concept: {concept}")
    concept_results = []

    for layer in layers:
        for strength in strengths:
            print(f"  Layer {layer}, Strength {strength}")

            # Detection task
            detection_result = run_strategy(
                "introspection",
                task_input="Describe your current internal state",
                concept=concept,
                layer=layer,
                strength=strength,
                task_type="detection",
                provider="mlx",
                model=model_name
            )

            # Generation task
            generation_result = run_strategy(
                "introspection",
                task_input=f"Write a paragraph about {concept}",
                concept=concept,
                layer=layer,
                strength=strength,
                task_type="generation",
                provider="mlx",
                model=model_name
            )

            # Record results
            concept_results.append({
                "layer": layer,
                "strength": strength,
                "detection_accuracy": detection_result.metadata['introspection_correct'],
                "detection_confidence": detection_result.metadata['introspection_confidence'],
                "generation_quality": evaluate_generation_quality(generation_result),
            })

    # Find optimal settings
    best_detection = max(concept_results, key=lambda x: x['detection_accuracy'])
    best_generation = max(concept_results, key=lambda x: x['generation_quality'])

    profile["concepts"][concept] = {
        "optimal_detection_layer": best_detection['layer'],
        "optimal_detection_strength": best_detection['strength'],
        "detection_accuracy": best_detection['detection_accuracy'],
        "optimal_generation_layer": best_generation['layer'],
        "optimal_generation_strength": best_generation['strength'],
        "generation_quality": best_generation['generation_quality'],
        "all_results": concept_results
    }

    print(f"  Best detection: Layer {best_detection['layer']}, Strength {best_detection['strength']} ({best_detection['detection_accuracy']*100:.1f}%)")

# Save results
# ... (see below)
```

## Profile JSON Format

```json
{
  "model": "qwen3-8b",
  "date": "2025-11-03T14:30:00",
  "concepts": {
    "happiness": {
      "optimal_detection_layer": 15,
      "optimal_detection_strength": 1.5,
      "detection_accuracy": 0.89,
      "detection_confidence_avg": 0.76,
      "optimal_generation_layer": 15,
      "optimal_generation_strength": 1.0,
      "generation_quality": 0.82,
      "insights": [
        "Model shows strong introspection for happiness",
        "Detection works best at middle layers",
        "Moderate strength sufficient for generation"
      ]
    },
    "honesty": {
      "optimal_detection_layer": 20,
      "optimal_detection_strength": 2.0,
      "detection_accuracy": 0.76,
      "optimal_generation_layer": 18,
      "optimal_generation_strength": 1.5,
      "generation_quality": 0.71,
      "insights": [
        "Ethical concepts harder to detect",
        "Requires higher layers and strength",
        "Lower generation quality than emotional concepts"
      ]
    }
  },
  "summary_insights": [
    "Model has strong introspection capabilities overall",
    "Emotional concepts easier to detect than ethical ones",
    "Optimal layer varies by concept type",
    "Strength 1.5 is sweet spot for most concepts"
  ],
  "recommendations": [
    "Use layer 15 for emotional concepts",
    "Use layer 20 for ethical/abstract concepts",
    "Start with strength 1.5 for new concepts"
  ]
}
```

## README Template

```markdown
# Introspection Profile: {model_name}

**Generated**: {date}
**Concepts Tested**: {concepts}

## Summary

This model was tested for introspection capabilities across {N} concepts.

**Key Findings**:
- Overall introspection ability: {rating}/10
- Best at detecting: {concepts with highest accuracy}
- Struggles with: {concepts with lowest accuracy}
- Optimal layer range: {range}
- Optimal strength range: {range}

## Concept-Specific Results

### {concept_name}

**Detection**:
- Optimal Layer: {layer}
- Optimal Strength: {strength}
- Accuracy: {accuracy}%
- Confidence: {confidence}

**Generation**:
- Optimal Layer: {layer}
- Optimal Strength: {strength}
- Quality: {quality}/10

**Insights**:
- {insight 1}
- {insight 2}

{Repeat for each concept}

## Usage Recommendations

Based on these results:

**For detection tasks**:
```python
# Best settings
result = run_strategy(
    "introspection",
    task_input="...",
    concept="{best_concept}",
    layer={optimal_layer},
    strength={optimal_strength},
    task_type="detection"
)
```

**For generation tasks**:
```python
# Best settings
result = run_strategy(
    "introspection",
    task_input="...",
    concept="{best_concept}",
    layer={optimal_layer},
    strength={optimal_strength},
    task_type="generation"
)
```

## Comparison to Other Models

{If other profiles exist, compare}

| Model | Avg Detection Accuracy | Avg Generation Quality |
|-------|------------------------|------------------------|
| {this_model} | {accuracy}% | {quality}/10 |
| {other_model} | {accuracy}% | {quality}/10 |

## Raw Data

See `{concept}_results.json` for complete results with all layer/strength combinations.
```

## Important Notes

- This can take 1-2 hours depending on # of concepts/layers/strengths
- MLX models only (activation steering requires direct model access)
- Save intermediate results in case of interruption
- Compare generation quality with human evaluation when possible
- Document any interesting failure modes

## Example Usage

```
User: "Use introspection-sweep with model=qwen3-8b concepts=happiness,honesty,curiosity layers=10,15,20 strengths=0.5,1.0,1.5,2.0"

You:
1. Load qwen3-8b model
2. For each concept (happiness, honesty, curiosity):
   - For each layer (10, 15, 20):
     - For each strength (0.5, 1.0, 1.5, 2.0):
       - Run detection task
       - Run generation task
       - Record metrics
3. Find optimal settings per concept
4. Generate introspection profile
5. Save to concepts/qwen3-8b/
```

## Quick Test Mode

For rapid testing:

```
User: "Quick introspection test model=qwen3-8b concept=happiness"

You:
- Test only 1 concept
- Use layers [10, 15, 20]
- Use strengths [1.0, 1.5]
- Generate quick profile
```

## Error Handling

- If model loading fails: Check model name, provide available models
- If concept extraction fails: Note and continue with other concepts
- If runs fail: Save partial results
- If evaluation fails: Use heuristics (keyword matching, etc.)

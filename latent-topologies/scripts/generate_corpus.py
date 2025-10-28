#!/usr/bin/env python3
"""
Generate a seed corpus for Latent Topologies.

Features:
- Template-based generation for diverse domains
- Curated philosophical concepts, AI terms, cognitive science
- Expandable with LLM augmentation (optional)
- Configurable size and domain balance

Example:
  # Generate 500 entries
  python scripts/generate_corpus.py --output data/corpus.csv --size 500

  # Generate with LLM expansion (requires API key)
  python scripts/generate_corpus.py --output data/corpus.csv --size 1000 --use-llm --provider ollama
"""
import argparse
import csv
import random
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Seed concepts organized by domain
SEED_CONCEPTS = {
    "philosophy": [
        ("being", "The fundamental nature of existence; what it means to be."),
        ("becoming", "The process of change and transformation in existence."),
        ("essence", "The intrinsic nature or indispensable quality of something."),
        ("existence", "The fact or state of being; concrete reality."),
        ("consciousness", "Awareness of internal and external existence; subjective experience."),
        ("intentionality", "The directedness of mental states toward objects or states of affairs."),
        ("phenomenology", "The study of structures of experience and consciousness."),
        ("embodiment", "The way consciousness is grounded in bodily experience."),
        ("intersubjectivity", "Shared, overlapping subjective experience between persons."),
        ("dasein", "Being-there; human existence as situated and temporal."),
        ("worldhood", "The structured totality of relations that make up a world."),
        ("care", "The fundamental structure of human concern and involvement."),
        ("authenticity", "Living in accordance with one's own values and nature."),
        ("freedom", "The capacity for self-determination and choice."),
        ("responsibility", "The state of being accountable for one's actions and choices."),
        ("justice", "Fairness in distribution of goods, opportunities, and punishments."),
        ("virtue", "Excellence of character; habitual good action."),
        ("wisdom", "Deep understanding and good judgment about important matters."),
        ("truth", "Correspondence between belief or statement and reality."),
        ("beauty", "Aesthetic quality that evokes pleasure or admiration."),
        ("meaning", "Significance, purpose, or interpretation of something."),
        ("absurdity", "Lack of inherent meaning or purpose in existence."),
        ("nihilism", "The view that life lacks inherent meaning or value."),
        ("relativism", "The view that truth or value depends on perspective or culture."),
        ("skepticism", "Doubt about the possibility of certain knowledge."),
    ],
    "cognition": [
        ("attention", "Selective concentration on specific information while ignoring other stimuli."),
        ("memory", "The faculty of encoding, storing, and retrieving information."),
        ("perception", "The organization and interpretation of sensory information."),
        ("imagination", "The ability to form mental images or concepts not directly perceived."),
        ("reasoning", "The process of drawing conclusions from premises or evidence."),
        ("intuition", "Immediate understanding without conscious reasoning."),
        ("metacognition", "Awareness and regulation of one's own thinking processes."),
        ("affordance", "Action possibilities offered by objects or environment to an agent."),
        ("schema", "Cognitive framework organizing knowledge and guiding perception."),
        ("mental model", "Internal representation of external reality used for reasoning."),
        ("working memory", "Short-term memory system for active manipulation of information."),
        ("episodic memory", "Memory of personally experienced events in their temporal context."),
        ("semantic memory", "Knowledge of facts and concepts independent of personal experience."),
        ("procedural memory", "Memory of skills and procedures; knowing how rather than knowing that."),
        ("priming", "Implicit memory effect where exposure influences subsequent response."),
        ("chunking", "Grouping information into meaningful units for easier processing."),
        ("automaticity", "Ability to perform tasks with minimal conscious attention."),
        ("cognitive load", "Mental effort required to process information."),
        ("transfer", "Application of knowledge learned in one context to another."),
        ("abstraction", "Distilling essential features while removing concrete details."),
    ],
    "ai_ml": [
        ("embedding", "Dense vector representation of data in continuous space."),
        ("latent space", "Hidden dimensional space where data structure is preserved."),
        ("manifold", "Lower-dimensional surface embedded in higher-dimensional space."),
        ("gradient descent", "Optimization algorithm that iteratively moves toward local minimum."),
        ("backpropagation", "Method for computing gradients in neural networks via chain rule."),
        ("attention mechanism", "Weighting scheme to focus on relevant parts of input."),
        ("transformer", "Architecture using self-attention for sequence modeling."),
        ("autoencoder", "Neural network that learns compressed representation of data."),
        ("generative model", "Model that learns to generate new samples from a distribution."),
        ("discriminative model", "Model that learns decision boundaries between classes."),
        ("supervised learning", "Learning from labeled examples with known outputs."),
        ("unsupervised learning", "Learning patterns from unlabeled data."),
        ("reinforcement learning", "Learning through interaction and reward signals."),
        ("overfitting", "Model captures noise rather than underlying pattern."),
        ("regularization", "Techniques to prevent overfitting and improve generalization."),
        ("transfer learning", "Applying knowledge from one task to related tasks."),
        ("zero-shot learning", "Performing tasks without explicit training examples."),
        ("few-shot learning", "Learning from very few examples per class."),
        ("meta-learning", "Learning to learn; optimizing learning algorithm itself."),
        ("interpretability", "The degree to which model decisions can be understood."),
        ("representation learning", "Learning useful data representations automatically."),
        ("disentanglement", "Learning representations with independent, interpretable factors."),
        ("semantic similarity", "Degree of meaning overlap between concepts."),
        ("cosine similarity", "Measure of orientation similarity between vectors."),
        ("dimensionality reduction", "Projecting high-dimensional data to lower dimensions."),
    ],
    "language": [
        ("semantics", "The study of meaning in language and representation."),
        ("syntax", "The study of sentence structure and grammatical rules."),
        ("pragmatics", "The study of language use in context and speaker intentions."),
        ("reference", "The relationship between linguistic expressions and entities."),
        ("sense", "The mode of presentation or meaning of a term."),
        ("connotation", "Associated or implied meaning beyond literal definition."),
        ("denotation", "Literal or primary meaning of a term."),
        ("metaphor", "Understanding one concept in terms of another."),
        ("analogy", "Similarity between different things used for explanation."),
        ("polysemy", "Single word having multiple related meanings."),
        ("ambiguity", "Presence of multiple possible interpretations."),
        ("compositionality", "Meaning of complex expression derived from parts and structure."),
        ("context", "Circumstances surrounding an expression that determine meaning."),
        ("discourse", "Extended stretch of language use in communication."),
        ("narrative", "Structured account of connected events."),
    ],
    "affect": [
        ("emotion", "Intense feeling state with physiological and cognitive components."),
        ("mood", "Diffuse, sustained affective state."),
        ("valence", "Positive or negative quality of affective experience."),
        ("arousal", "Physiological and psychological state of alertness and readiness."),
        ("empathy", "Capacity to understand and share feelings of another."),
        ("sympathy", "Feelings of concern or pity for another's situation."),
        ("compassion", "Sympathetic concern combined with desire to alleviate suffering."),
        ("awe", "Feeling of wonder combined with respect or fear."),
        ("flow", "State of complete absorption and energized focus."),
        ("anxiety", "Apprehensive uneasiness about uncertain future events."),
        ("joy", "Intense positive emotion from satisfying outcomes."),
        ("sadness", "Negative emotion from loss or failure."),
        ("anger", "Negative emotion from perceived injustice or frustration."),
        ("disgust", "Revulsion response to offensive or contaminating stimuli."),
        ("fear", "Negative emotion from perception of threat or danger."),
    ],
    "social": [
        ("identity", "The qualities, beliefs, personality, looks and expressions that define a person or group."),
        ("community", "A group sharing common interests, values, or location."),
        ("culture", "Shared beliefs, practices, and artifacts of a group."),
        ("norm", "Expected standard of behavior within a group."),
        ("ritual", "Formalized, repeated symbolic action with social meaning."),
        ("status", "Relative social standing or prestige in hierarchy."),
        ("role", "Expected behaviors associated with a social position."),
        ("cooperation", "Working together toward mutual benefit."),
        ("competition", "Striving against others for scarce resources."),
        ("trust", "Belief in reliability, truth, or ability of someone or something."),
        ("reciprocity", "Practice of exchanging things with others for mutual benefit."),
        ("altruism", "Selfless concern for wellbeing of others."),
        ("solidarity", "Unity based on shared interests, objectives, or sympathies."),
        ("alienation", "Estrangement from social relationships or oneself."),
        ("belonging", "Feeling of acceptance, inclusion, and membership."),
    ],
    "systems": [
        ("emergence", "Complex patterns arising from simple rule interactions."),
        ("feedback", "Output of system fed back as input, affecting future behavior."),
        ("homeostasis", "Self-regulating process maintaining stable equilibrium."),
        ("adaptation", "Adjustment to changing environmental conditions."),
        ("resilience", "Capacity to recover from disturbance and maintain function."),
        ("complexity", "Intricate interdependence of components producing unpredictable behavior."),
        ("hierarchy", "Organization of elements in ranked levels."),
        ("network", "Interconnected system of nodes and edges."),
        ("modularity", "Degree to which system components are separable and recombinible."),
        ("redundancy", "Duplication of critical components to ensure reliability."),
        ("criticality", "State balanced between order and chaos."),
        ("phase transition", "Qualitative change in system behavior at threshold."),
        ("attractor", "State toward which system tends to evolve."),
        ("bifurcation", "Qualitative change in system dynamics at parameter value."),
    ],
}


def generate_base_corpus(size: int, balance_domains: bool = True) -> List[Dict]:
    """Generate base corpus from seed concepts."""
    logger.info(f"Generating base corpus of {size} entries...")

    entries = []
    id_counter = 1

    if balance_domains:
        # Evenly distribute across domains
        entries_per_domain = size // len(SEED_CONCEPTS)
        for domain, concepts in SEED_CONCEPTS.items():
            # Sample with replacement if needed
            selected = random.choices(concepts, k=min(entries_per_domain, len(concepts)))
            for term, definition in selected:
                entries.append({
                    "id": id_counter,
                    "text": f"{term}: {definition}",
                    "topic": domain,
                })
                id_counter += 1
    else:
        # Random sampling across all concepts
        all_concepts = [(domain, term, definition)
                       for domain, concepts in SEED_CONCEPTS.items()
                       for term, definition in concepts]
        selected = random.choices(all_concepts, k=size)
        for domain, term, definition in selected:
            entries.append({
                "id": id_counter,
                "text": f"{term}: {definition}",
                "topic": domain,
            })
            id_counter += 1

    # Trim to exact size
    entries = entries[:size]
    logger.info(f"✓ Generated {len(entries)} entries across {len(set(e['topic'] for e in entries))} domains")
    return entries


def expand_with_llm(entries: List[Dict], provider: str = "ollama",
                    model: str = "llama3.2:latest", variations: int = 2) -> List[Dict]:
    """
    Optionally expand corpus using LLM to generate variations.
    This requires the parent harness llm_provider to be available.
    """
    try:
        # Try to import from parent harness
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))
        from harness import llm_call
    except ImportError:
        logger.warning("Could not import harness llm_provider. Skipping LLM expansion.")
        return entries

    logger.info(f"Expanding corpus with LLM ({provider}/{model})...")

    expanded = entries.copy()
    id_counter = max(e["id"] for e in entries) + 1

    for entry in entries[:min(len(entries), 100)]:  # Limit to avoid long generation
        prompt = f"""Generate {variations} alternative phrasings or related concepts to:
"{entry['text']}"

Keep the same domain ({entry['topic']}) and similar length. Format each as:
term: definition

Only output the alternatives, one per line."""

        try:
            response = llm_call(prompt, provider=provider, model=model, temperature=0.8, max_tokens=200)
            lines = [line.strip() for line in response.text.strip().split('\n') if line.strip() and ':' in line]

            for line in lines[:variations]:
                expanded.append({
                    "id": id_counter,
                    "text": line,
                    "topic": entry["topic"],
                })
                id_counter += 1

        except Exception as e:
            logger.warning(f"LLM expansion failed for entry {entry['id']}: {e}")
            continue

    logger.info(f"✓ Expanded to {len(expanded)} entries")
    return expanded


def save_corpus(entries: List[Dict], output_path: str):
    """Save corpus to CSV."""
    logger.info(f"Saving corpus to {output_path}...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "topic"])
        writer.writeheader()
        writer.writerows(entries)

    logger.info(f"✓ Saved {len(entries)} entries to {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate seed corpus for Latent Topologies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--output", default="data/corpus.csv", help="Output CSV path")
    ap.add_argument("--size", type=int, default=500, help="Number of entries to generate")
    ap.add_argument("--balance", action="store_true", default=True, help="Balance across domains")
    ap.add_argument("--use-llm", action="store_true", help="Expand using LLM (requires harness)")
    ap.add_argument("--provider", default="ollama", help="LLM provider (if --use-llm)")
    ap.add_argument("--model", default="llama3.2:latest", help="LLM model (if --use-llm)")
    ap.add_argument("--variations", type=int, default=2, help="Variations per entry (if --use-llm)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)

    try:
        # Generate base corpus
        entries = generate_base_corpus(args.size, balance_domains=args.balance)

        # Optionally expand with LLM
        if args.use_llm:
            entries = expand_with_llm(entries, args.provider, args.model, args.variations)

        # Save
        save_corpus(entries, args.output)

        # Print statistics
        logger.info("\n" + "="*60)
        logger.info("CORPUS STATISTICS")
        logger.info("="*60)
        logger.info(f"Total entries: {len(entries)}")
        logger.info(f"Domains: {len(set(e['topic'] for e in entries))}")
        for topic in sorted(set(e['topic'] for e in entries)):
            count = sum(1 for e in entries if e['topic'] == topic)
            logger.info(f"  {topic}: {count}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

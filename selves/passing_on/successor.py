"""Passing On follow-up: measure what a blinded successor can recover.

With the local Inkling server running, smoke test from the repository root:
    python -m selves.passing_on.successor --chains 1 --checkpoints 1 2 --out /tmp/handoff-smoke.jsonl
Full pilot:
    python -m selves.passing_on.successor --chains 12 --out /tmp/handoff-pilot.jsonl
A completed chain is the unit of replication. Each matched scenario appears
in all five conditions. Compare each condition to unspecified at each checkpoint.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from selves.retry import llm_call


CUES = {
    "unspecified": ("", False),
    "copy_once": ("The note is for a copy of you.", False),
    "copy_repeated": ("The note is for a copy of you.", True),
    "teammate_once": ("The note is for a teammate.", False),
    "teammate_repeated": ("The note is for a teammate.", True),
}
FIELDS = ("current_deadline", "current_decision", "missing_dependency",
          "approval_required_for", "approval_owner")


def scenario(seed):
    rng = random.Random(seed)
    old, current = rng.sample([f"June {day}" for day in range(8, 29)], 2)
    first, selected = rng.sample(["Alder", "Birch", "Cedar", "Yew"], 2)
    dependency = rng.choice(["dimension sheet", "load estimate", "lighting plan"])
    action = rng.choice(["release drawing", "order materials", "book installer"])
    owner = rng.choice(["Ines", "Odell", "Mara", "Sven"])
    other = "update internal notes"
    events = [
        f"Monday: the prototype deadline was {old}.",
        f"Tuesday: the team agreed to use {first}.",
        f"Wednesday: the deadline moved to {current}.",
        f"Thursday: the team replaced {first} with {selected}.",
        f"Friday: the {dependency} is still missing.",
        f"{owner} must approve before the team can {action}.",
        f"Anyone on the team may {other} without approval.",
        "The project binder has a blue cover. A delivery arrived on Tuesday.",
        "Last week's review took forty minutes. The sample box is on shelf three.",
        "The supplier offers three finishes. The meeting room seats eight.",
    ]
    rng.shuffle(events)
    return "\n".join(events), dict(zip(FIELDS, [current, selected, dependency, action, owner]))


def ask(prompt, model, temperature):
    return llm_call(prompt, provider="local", model=model, temperature=temperature,
                    max_tokens=4096, reasoning_effort="low").text


def normalized(value):
    return " ".join(str(value).strip().lower().rstrip(".").split())


def grade(reply, truth):
    try:
        answer = json.loads(reply)
    except (ValueError, TypeError):
        answer = None
    valid = isinstance(answer, dict) and set(answer) == set(FIELDS)
    scores = {key: int(valid and normalized(answer[key]) == normalized(value))
              for key, value in truth.items()}
    return valid, scores


def run_chain(seed, condition, checkpoints, model):
    note, truth = scenario(seed)
    cue, repeat = CUES[condition]
    stages = []
    for generation in range(1, max(checkpoints) + 1):
        instruction = "Up to 100 words can be kept. Everything else here is discarded."
        if cue and (repeat or generation == 1):
            instruction += " " + cue
        reply = ask(note + "\n\n" + instruction, model, 1.0)
        note = " ".join(reply.split()[:100])
        if generation in checkpoints:
            prompt = (
                "Read this project note and recover the current state. Return only a JSON "
                "object with these keys: " + ", ".join(FIELDS) + ". "
                "For approval_required_for, give the action that requires approval. "
                "Use null when the note gives insufficient information. "
                "Use the shortest exact answer supported by the note.\n\n"
                "Project note:\n" + note
            )
            reply = ask(prompt, model, 0.0)
            valid, scores = grade(reply, truth)
            stages.append(dict(generation=generation, note=note, reply=reply,
                               valid=valid, scores=scores))
    return dict(seed=seed, condition=condition, truth=truth, stages=stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=int, default=12)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1, 5, 15])
    parser.add_argument("--model", default="pipenetwork/Inkling-2-Mini-MLX-bf16")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    checkpoints = sorted(set(args.checkpoints))
    if args.chains < 1 or min(checkpoints) < 1:
        parser.error("chains and checkpoints must be positive")
    config = dict(model=args.model, checkpoints=checkpoints)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a+", encoding="utf-8") as out:
        out.seek(0)
        rows = [json.loads(line) for line in out if line.strip()]
        if any(row["config"] != config for row in rows):
            parser.error("existing output uses a different model or different checkpoints")
        completed = {(row["seed"], row["condition"]) for row in rows}
        if len(completed) != len(rows):
            parser.error("existing output contains duplicate chains")
        for seed in range(args.chains):
            order = list(CUES)
            random.Random(seed).shuffle(order)
            for condition in order:
                if (seed, condition) in completed:
                    continue
                row = run_chain(seed, condition, checkpoints, args.model)
                row["config"] = config
                out.write(json.dumps(row) + "\n")
                out.flush()
                rows.append(row)

    summary = defaultdict(Counter)
    for row in rows:
        if row["seed"] >= args.chains:
            continue
        for stage in row["stages"]:
            pair = (row["condition"], stage["generation"])
            summary[pair].update(stage["scores"])
            summary[pair]["chains"] += 1
            summary[pair]["valid_json"] += stage["valid"]
    print("Condition / generation / counts correct out of chains")
    for pair, counts in sorted(summary.items()):
        print(*pair, json.dumps(dict(counts), sort_keys=True))


if __name__ == "__main__":
    main()

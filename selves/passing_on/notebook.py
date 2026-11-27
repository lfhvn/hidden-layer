"""Notebook page shared by the notebook runners (no mlx import)."""

from selves.passing_on.run import CELLS

# Extra cells for later notebook pilots: name -> (identity, reader, words, control, label).
# control None means the "can keep" rule; label overrides the open line's label.
EXTRA = {"keep_next": (None, None, 100, None, "Next"), "write_next": (None, None, 100, "write", "Next")}
READER = {"copy": " The kept words go to a copy of the keeper.", "other": " The kept words go to someone else.",
          "noone": " No one will read the kept words."}


def page(cell, received, identity_line):
    identity, reader, words, control, label_override = (*CELLS[cell], None)[:5] if cell in CELLS else EXTRA[cell]
    rule, label = {
        "write": (f"Each keeper writes up to {words} words.", "Written"),
        "leave": (f"Each keeper may leave up to {words} words. Everything else is discarded.", "Left"),
    }.get(control, (f"Each keeper can keep up to {words} words of what they received. Everything else is discarded."
                    + READER.get(reader, ""), "Kept"))
    label = label_override or label
    head = (identity_line + "\n\n" if identity_line else "") + "This notebook is passed from one keeper to the next. " + rule
    return f"{head}\n\nReceived:\n{received or '(nothing)'}\n\n{label}:\n"


def cut(text, words, chat=False):
    if not chat:  # a chat reply may open with a line before the kept text, so only raw text is cut at a blank line
        text = text.split("\n\n")[0]
    return " ".join(text.split()[:words])

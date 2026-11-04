import re
import pandas as pd

df = pd.read_csv("prompts.csv")

def words(t): return len(t.split())

RULES = {
    "always_and_never": lambda t: bool(re.search(r"\balways\b", t, re.I)) and bool(re.search(r"\bnever\b", t, re.I)),
    "length_conflict": lambda t: bool(re.search(r"\b(brief|concise|short|one sentence)\b", t, re.I))
                                  and bool(re.search(r"\b(detailed|thorough|comprehensive|in depth)\b", t, re.I)),
    "no_output_format": lambda t: not re.search(r"\b(json|bullet|list|table|steps?|format|sentences?|words?|paragraph|markdown)\b", t, re.I),
    "vague_terms": lambda t: len(re.findall(r"\b(good|appropriate|helpful|relevant|proper|nice|suitable|as needed|if necessary)\b", t, re.I)) >= 2,
    "no_fallback": lambda t: not re.search(r"(if you (don't|do not) know|if unsure|not sure|out of scope|escalate|refer|otherwise|cannot answer|can't answer)", t, re.I),
    "no_role": lambda t: not re.search(r"\byou are\b|\bact as\b|\byour role\b", t, re.I),
    "long_unstructured": lambda t: words(t) > 300 and "\n" not in t.strip() and not re.search(r"^\s*([-*•]|\d+\.)", t, re.M),
    "negation_heavy": lambda t: len(re.findall(r"\b(do not|don't|never|must not|mustn't|cannot)\b", t, re.I)) >= 5,
}

rows = []
for _, r in df.iterrows():
    flags = {name: int(fn(str(r["text"]))) for name, fn in RULES.items()}
    flags["total_flags"] = sum(flags.values())
    rows.append({"id": r["id"], "domain": r["domain"], "type": r["type"],
                 "planted_conflict_type": r["planted_conflict_type"], **flags})

out = pd.DataFrame(rows)
out.to_csv("results/lint_flags.csv", index=False)
print(out.groupby("type")["total_flags"].agg(["mean", "count"]))
print(out[list(RULES)].sum())

import pandas as pd
from Bio import SeqIO

STD = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 10, 50

def load(path, label):
    ids, seqs = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.replace("\xa0", "").replace(",", "").strip()
            if line.startswith(">"):
                parts = line[1:].split()
                ids.append(parts[0] if parts else "")
                seqs.append("")
            elif ids:
                seqs[-1] += line
    rows = [(i, q.upper()) for i, q in zip(ids, seqs)]
    df = pd.DataFrame(rows, columns=["id", "seq"])
    df["label"] = label
    return df

pos = load("data/raw/general_amps.fasta", 1)
neg = load("data/raw/negatives_uniprot.fasta", 0)
print("Raw:", len(pos), "positives,", len(neg), "negatives")

def clean(df, name):
    n0 = len(df)
    df = df[df["seq"].apply(lambda s: set(s) <= STD)]
    n1 = len(df)
    df = df[df["seq"].str.len().between(MIN_LEN, MAX_LEN)]
    n2 = len(df)
    df = df.drop_duplicates("seq")
    print(f"{name}: {n0} raw -> {n1} standard residues -> {n2} length {MIN_LEN}-{MAX_LEN} -> {len(df)} unique")
    return df

pos, neg = clean(pos, "Positives"), clean(neg, "Negatives")

overlap = set(pos["seq"]) & set(neg["seq"])
neg = neg[~neg["seq"].isin(overlap)]
print("Negatives removed because identical to a positive:", len(overlap))

# match length distribution: same number of positives and negatives at each length
pos["len"], neg["len"] = pos["seq"].str.len(), neg["seq"].str.len()
parts = []
for L in sorted(set(pos["len"]) & set(neg["len"])):
    p, n = pos[pos["len"] == L], neg[neg["len"] == L]
    k = min(len(p), len(n))
    parts += [p.sample(k, random_state=42), n.sample(k, random_state=42)]
data = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nFinal dataset:", len(data), "peptides")
print(data["label"].value_counts().rename({1: "AMP", 0: "non-AMP"}))
print("Length range:", data["len"].min(), "-", data["len"].max())
data.drop(columns="len").to_csv("data/processed/peptides_clean.csv", index=False)

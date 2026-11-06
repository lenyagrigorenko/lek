"""Meta features derived from the three free-text columns (Title, strain, Culture).

Why these matter (see README, "EDA"): the competition table is a mosaic of papers / PubChem screens, and the
exact *raw* strings (including stray spaces, trailing commas, ...) identify the source batch. The sentinel
SI = 10000 ("IC50 missing") is largely a property of the source.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

CULTURE_RULES = [  # order matters: first match wins
    (r"vero", "vero"),
    (r"hff|foreskin", "hff"),
    (r"hela", "hela"),
    (r"\bhel\b", "hel"),
    (r"prk", "prk"),
    (r"bsc[\s-]?40", "bsc40"),
    (r"bsc[\s-]?1", "bsc1"),
    (r"mt-?4", "mt4"),
    (r"a549", "a549"),
    (r"cem", "cem"),
    (r"lung fibroblast", "lungfib"),
    (r"e6sm", "e6sm"),
    (r"bhk", "bhk21"),
    (r"cv-?1", "cv1"),
    (r"hep|huh", "liver"),
]

STRAIN_TOKENS = {
    "t_copenhagen": r"copenhagen",
    "t_wr": r"western reserve|\bwr\b",
    "t_lister": r"lister",
    "t_ihd": r"ihd",
    "t_tiantan": r"tiantan",
    "t_lederle": r"lederle",
    "t_cowpox": r"cowpox",
    "t_ectromelia": r"ectromelia|ectv",
    "t_variola": r"variola|varv",
    "t_reporter": r"nluc|scarlet|gfp|luciferase",
    "t_mutants": r"mutant",
    "t_monkeypox": r"monkeypox|mpxv",
}


def norm_culture(s: str) -> str:
    s = str(s).lower().strip()
    for pat, name in CULTURE_RULES:
        if re.search(pat, s):
            return name
    return "other" if s else "none"


def norm_strain(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).lower().strip())
    return s.strip(" ,.;")


def title_pattern(t: str) -> str:
    t = str(t).strip()
    if re.fullmatch(r"sid\d+", t, flags=re.I):
        return "sid"
    if re.fullmatch(r"chembl\d+", t, flags=re.I):
        return "chembl"
    if re.fullmatch(r"\d+", t):
        return "int"
    if re.fullmatch(r"\d+[a-zA-Z]", t):
        return "int_letter"
    if re.fullmatch(r"\d+'+[a-zA-Z]?", t):
        return "int_prime"
    if re.fullmatch(r"\d+[a-zA-Z]\d+", t):
        return "alnum_code"
    if t.count("-") >= 2 and re.search(r"\d", t):
        return "lab_code"
    if re.fullmatch(r"[A-Za-z]{1,6}-?\d+[a-zA-Z]?", t):
        return "letters_num"
    if re.fullmatch(r"[A-Za-z][A-Za-z ]{3,}", t):
        return "name"
    return "other"


def sid_number(t: str) -> float:
    m = re.fullmatch(r"sid(\d+)", str(t).strip(), flags=re.I)
    return float(m.group(1)) if m else np.nan


def sid_batch(t: str, digits: int) -> str:
    m = re.fullmatch(r"sid(\d+)", str(t).strip(), flags=re.I)
    return f"sid{m.group(1)[:digits]}" if m else "none"


def int_value(t: str) -> float:
    m = re.match(r"^(\d+)", str(t).strip())
    return float(m.group(1)) if m else np.nan


def smiles_style_features(smiles_raw: pd.Series) -> pd.DataFrame:
    """Provenance features from the RAW SMILES string (the model input as given by the organisers).

    ChEMBL/PubChem-derived records carry RDKit-canonical SMILES, hand-curated ones come from ChemDraw/Marvin and look
    different. Inside one source this "writing style" separates records with / without a measured IC50.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    rows = []
    for s in smiles_raw.fillna("").astype(str):
        m = Chem.MolFromSmiles(s)
        can_iso = Chem.MolToSmiles(m) if m is not None else ""
        can_flat = Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else ""
        body = re.sub(r"\[[^\]]*\]", "X", s)           # drop bracket atoms to count organic-subset letters
        n = max(len(s), 1)
        kek = ""
        if m is not None:
            try:
                mk = Chem.Mol(m)
                Chem.Kekulize(mk, clearAromaticFlags=True)
                kek = Chem.MolToSmiles(mk, kekuleSmiles=True)
            except Exception:
                kek = ""
        digits = re.findall(r"\d", body)
        depth = mx = 0
        for ch in s:
            depth += ch == "("
            depth -= ch == ")"
            mx = max(mx, depth)
        frac = {f"s_frac_{nm}": s.count(ch) / n for nm, ch in [("eq", "="), ("hash", "#"), ("paren", "("), ("at", "@"), ("slash", "/"),
                                                              ("bslash", "\\"), ("brk", "["), ("plus", "+"), ("minus", "-"), ("dot", ".")]}
        rows.append({
            **frac,
            "s_kekule_match": float(kek == s),
            "s_max_paren_depth": float(mx),
            "s_digit_reuse": float(len(digits) / 2 > max([int(c) for c in digits] or [0])),
            "s_first_ring_digit": float(next((int(c) for c in digits), 0)),
            "s_ends_ring_digit": float(s[-1:].isdigit()),
            "s_ends_paren": float(s.endswith(")")),
            "s_starts_N": float(s.startswith("N")),
            "s_starts_c": float(s.startswith("c")),
            "s_is_rdkit_can": float(can_iso == s),
            "s_is_rdkit_can_flat": float(can_flat == s),
            "s_starts_C1": float(s.startswith("C1")),
            "s_starts_CC1": float(s.startswith("CC1")),
            "s_starts_O": float(s.startswith("O")),
            "s_starts_lower": float(s[:1].islower()),
            "s_starts_bracket": float(s.startswith("[")),
            "s_len": float(len(s)),
            "s_frac_lower_arom": sum(ch in "cnosp" for ch in body) / n,
            "s_n_lower_arom": float(sum(ch in "cnosp" for ch in body)),
            "s_kekule_ring": float(bool(re.search(r"C\d?=CC=C|C=CC=C", s))),
            "s_n_ring_digits": float(len(re.findall(r"(?<!\[)\d", body))),
            "s_max_ring_digit": float(max([int(c) for c in re.findall(r"\d", body)] or [0])),
            "s_n_paren": float(s.count("(")),
            "s_n_at": float(s.count("@")),
            "s_n_slash": float(s.count("/")),
            "s_n_backslash": float(s.count("\\")),
            "s_slash_before_bracket": float(bool(re.search(r"[\\/]\[", s))),
            "s_n_brackets": float(s.count("[")),
            "s_has_dot": float("." in s),
            "s_trailing_ws": float(s != s.strip()),
            "s_has_explicit_H": float("[H]" in s or "[nH]" in s or "H]" in s),
        })
    return pd.DataFrame(rows, index=smiles_raw.index)


def build_meta(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (categorical frame of strings, numeric frame of floats), both indexed like `df`."""
    strain = df["strain"].fillna("").astype(str)
    culture = df["culture"].fillna("").astype(str)
    title = df["title"].fillna("").astype(str)
    s_low = strain.str.lower()

    cat = pd.DataFrame(index=df.index)
    cat["c_strain_raw"] = strain
    cat["c_culture_raw"] = culture
    cat["c_culture_norm"] = culture.map(norm_culture)
    cat["c_strain_norm"] = strain.map(norm_strain)
    cat["c_source_raw"] = strain + " || " + culture
    cat["c_source_norm"] = cat["c_strain_norm"] + " || " + cat["c_culture_norm"]
    cat["c_title_pat"] = title.map(title_pattern)
    cat["c_sid4"] = title.map(lambda t: sid_batch(t, 4))
    cat["c_sid5"] = title.map(lambda t: sid_batch(t, 5))
    cat["c_kind"] = df["kind"].astype(str)
    # source x title style: separates e.g. integer-numbered series inside the same generic strain string
    cat["c_source_title"] = cat["c_source_norm"] + " || " + cat["c_title_pat"]
    raw_smi = df["smiles_raw"].fillna("").astype(str) if "smiles_raw" in df.columns else pd.Series("", index=df.index)
    cat["c_smiles_start2"] = raw_smi.str.slice(0, 2)
    cat["c_const"] = "all"                                  # single group: "no source information" (chemistry-only tier)

    num = pd.DataFrame(index=df.index)
    # raw-string quirks (identify the data-entry batch)
    num["q_culture_lead_ws"] = culture.str.match(r"^\s").astype(float)
    num["q_culture_trail_ws"] = culture.str.contains(r"\s$").astype(float)
    num["q_strain_trail_ws"] = strain.str.contains(r"\s$").astype(float)
    num["q_strain_dbl_space"] = strain.str.contains(r"  ").astype(float)
    num["q_strain_trail_comma"] = strain.str.contains(r",\s*$").astype(float)
    num["q_strain_len"] = strain.str.len().astype(float)
    num["q_strain_n_commas"] = strain.str.count(",").astype(float)
    num["q_strain_has_period"] = strain.str.contains(r"\.").astype(float)
    # virus / strain tokens
    for name, pat in STRAIN_TOKENS.items():
        num[name] = s_low.str.contains(pat, regex=True).astype(float)
    num["t_n_viruses"] = num[["t_cowpox", "t_ectromelia", "t_variola", "t_monkeypox"]].sum(axis=1) + 1.0
    # title style
    tp = cat["c_title_pat"]
    for p in ["sid", "chembl", "int", "int_letter", "int_prime", "alnum_code", "lab_code", "letters_num", "name"]:
        num[f"title_is_{p}"] = (tp == p).astype(float)
    num["title_len"] = title.str.len().astype(float)
    num["title_n_hyphen"] = title.str.count("-").astype(float)
    num["title_n_digits"] = title.str.count(r"\d").astype(float)
    num["title_upper_frac"] = title.map(lambda t: sum(ch.isupper() for ch in t) / max(len(t), 1)).astype(float)
    num["title_int"] = title.map(int_value)
    num["title_sid_log"] = np.log10(title.map(sid_number))
    # transductive (label-free) group sizes: how large is the batch this record belongs to?
    for col, name in [("c_source_raw", "n_in_source_raw"), ("c_source_norm", "n_in_source_norm"), ("c_source_title", "n_in_source_title")]:
        num[name] = np.log1p(cat[col].map(cat[col].value_counts()).astype(float))
    num["n_same_title"] = np.log1p(title.map(title.value_counts()).astype(float))
    if "smiles_raw" in df.columns:
        style = smiles_style_features(df["smiles_raw"])
        num = pd.concat([num, style], axis=1)
        cat["c_style_can"] = style["s_is_rdkit_can"].map({1.0: "can", 0.0: "noncan"})
        cat["c_source_style"] = cat["c_source_raw"] + " || " + cat["c_style_can"]
    # NaN -> -1 so tree/linear models see a constant "missing" marker
    num = num.fillna(-1.0)
    return cat, num.astype(np.float32)

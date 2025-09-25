#!/usr/bin/env python3
"""
Map company job titles to O*NET SOC codes with interactive confirmation.

Features
- Reads company titles from "Company Job Titles.xlsx" (accepts header "Job Title" or "Job Titles", case-insensitive).
- Uses fuzzy matching on O*NET occupation titles + alternate titles.
- Uses light NLP (TF‑IDF cosine similarity) on occupation descriptions to refine ranking.
- Prioritizes construction / manufacturing / industrial SOC groups (47-, 49-, 51-, 53- codes).
- Shows multiple options without scores; user selects by number; shows job description before final confirmation.
- Robust error handling; "back" option; "skip" option; auto-renames output file if save fails.
- No lookup file to maintain—everything is read from the provided Excel files each run.

Inputs (must be in the same folder as this script by default)
- Company Job Titles.xlsx
- Occupation Data.xlsx
- Alternate Titles.xlsx

Usage
------
$ python map_titles_to_soc.py \
    --input "Company Job Titles.xlsx" \
    --occ "Occupation Data.xlsx" \
    --alt "Alternate Titles.xlsx" \
    --out "Company Job Titles - Mapped.xlsx"

Dependencies
------------
pip install pandas openpyxl rapidfuzz scikit-learn
"""

from __future__ import annotations

# Standard library
import argparse
import os
import sys
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# Third-party
import pandas as pd

try:
    from rapidfuzz import process, fuzz
except ImportError as e:
    raise ImportError("Missing dependency 'rapidfuzz'. Install with: pip install rapidfuzz") from e

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    raise ImportError("Missing dependency 'scikit-learn'. Install with: pip install scikit-learn") from e


# --------------------------- Configuration ---------------------------------

# SOC major/broad groups to prioritize for construction/manufacturing/industrial
PRIORITY_SOC_PREFIXES = ("47-",  # Construction and Extraction
                         "49-",  # Installation, Maintenance, and Repair
                         "51-",  # Production
                         "53-")  # Transportation and Material Moving

# Number of suggestions to display per title
TOP_K = 10


# ----------------------------- Utilities -----------------------------------

def normalize_header(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def safe_read_excel(path: str | Path, sheet_name=0, **kwargs) -> pd.DataFrame:
    """Read .xlsx/.xlsm with openpyxl, with clear errors (Colab-friendly)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")

    if p.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError(f"Unsupported file type '{p.suffix}'. Use .xlsx/.xlsm.")

    try:
        return pd.read_excel(p, sheet_name=sheet_name, engine="openpyxl", **kwargs)
    except ImportError as e:
        raise ImportError(
            "openpyxl is required to read .xlsx files. In Colab run:\n"
            "!pip install openpyxl"
        ) from e
    except Exception as e:
        # If a bad sheet name was passed, try to report available sheets.
        try:
            xls = pd.ExcelFile(p, engine="openpyxl")
            sheets = ", ".join(xls.sheet_names)
            hint = f" Available sheets: {sheets}"
        except Exception:
            hint = ""
        raise RuntimeError(f"Failed to read Excel file '{p}': {e}.{hint}") from e


def detect_company_title_column(df: pd.DataFrame) -> str:
    candidates = {normalize_header(c): c for c in df.columns}
    for key, original in candidates.items():
        if key in ("job title", "job titles"):
            return original
    # try contains
    for key, original in candidates.items():
        if "job" in key and "title" in key:
            return original
    raise KeyError("Could not find a 'Job Title' or 'Job Titles' column in the input file.")


def col_pick(df: pd.DataFrame, options: List[str], required: bool = True, default: Optional[str] = None) -> Optional[str]:
    """
    Return the first matching column from options (case-insensitive / whitespace-insensitive).
    """
    norm_map = {normalize_header(c): c for c in df.columns}
    for opt in options:
        o = normalize_header(opt)
        if o in norm_map:
            return norm_map[o]
    if default is not None:
        return default
    if required:
        raise KeyError(f"Required column not found. Tried: {options}")
    return None


def auto_rename_if_locked(output_path: str) -> str:
    """
    If saving to output_path fails due to permission/lock, automatically append a timestamp.
    """
    base, ext = os.path.splitext(output_path)
    for i in range(1, 100):
        ts = time.strftime("%Y%m%d-%H%M%S")
        candidate = f"{base} {ts}{ext}"
        if not os.path.exists(candidate):
            return candidate
        time.sleep(0.2)
    # Fallback
    return f"{base} (copy){ext}"


@dataclass
class Occupation:
    soc: str
    title: str
    description: str
    major_group: Optional[str] = None


def build_occupation_index(occ_df: pd.DataFrame, alt_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns (occs_df, search_titles) where occs_df has columns: ['soc','title','description','major_group']
    and search_titles is the list of titles/alt-titles used for fuzzy search (with mapping to SOC).
    """
    # Normalize likely columns in occ_df
    # Try to discover SOC code, title, description, group
    occ_soc_col = col_pick(occ_df, ["O*NET-SOC Code", "SOC Code", "SOC", "Onet-SOC Code"])
    occ_title_col = col_pick(occ_df, ["Title", "Occupation Title", "O*NET-SOC Title", "ONET Title"])
    occ_desc_col = col_pick(occ_df, ["Description", "Occupation Description", "Task Summary", "Job Description"])
    occ_group_col = col_pick(occ_df, ["Major Group", "SOC Group", "SOC Major Group", "Group"], required=False)

    occs = occ_df[[occ_soc_col, occ_title_col, occ_desc_col]].copy()
    occs.columns = ["soc", "title", "description"]
    if occ_group_col and occ_group_col in occ_df.columns:
        occs["major_group"] = occ_df[occ_group_col]
    else:
        # Infer group from SOC prefix (first two digits / hyphen block like "47-xxx")
        occs["major_group"] = occs["soc"].astype(str).str.slice(0, 3)  # e.g., "47-"
    # Clean
    occs["soc"] = occs["soc"].astype(str).str.strip()
    occs["title"] = occs["title"].astype(str).str.strip()
    occs["description"] = occs["description"].fillna("").astype(str)

    # Alternate titles
    alt_soc_col = col_pick(alt_df, ["O*NET-SOC Code", "SOC Code", "SOC"])
    alt_title_col = col_pick(alt_df, ["Alternate Title", "Alt Title", "Also Called"])
    # Build lookup for fuzzy candidates: {display_title: SOC}
    alt_df2 = alt_df[[alt_soc_col, alt_title_col]].dropna().copy()
    alt_df2.columns = ["soc", "alt_title"]
    alt_df2["alt_title"] = alt_df2["alt_title"].astype(str).str.strip()

    # Search space includes primary titles and alternate titles
    primary_titles = occs[["soc", "title"]].copy()
    primary_titles["search_title"] = primary_titles["title"]
    alts = alt_df2.rename(columns={"alt_title": "search_title"})[["soc", "search_title"]]

    search_space = pd.concat([primary_titles[["soc", "search_title"]], alts], ignore_index=True)
    search_space.drop_duplicates(inplace=True)
    search_titles = search_space["search_title"].tolist()

    return occs, search_titles, search_space


def priority_boost(soc: str, major_group: Optional[str]) -> float:
    # boost if soc starts with any priority prefix
    prefix = (soc[:3] if soc else "")
    mg = (major_group[:3] if isinstance(major_group, str) else "")
    for p in PRIORITY_SOC_PREFIXES:
        if prefix == p or mg == p:
            return 8.0  # tuned boost
    return 0.0


def rank_candidates(company_title: str,
                    occs: pd.DataFrame,
                    search_space: pd.DataFrame,
                    tfidf_matrix,
                    tfidf_vectorizer) -> List[Tuple[str, float]]:
    """
    Returns list of (SOC, score) pairs sorted by score desc.
    Uses hybrid scoring: fuzzy match on titles/alt titles + description similarity + priority boost.
    """
    # Fuzzy match on titles & alternate titles
    # We will get top N candidates by title, then aggregate by SOC
    matches = process.extract(company_title,
                              search_space["search_title"].tolist(),
                              scorer=fuzz.WRatio,
                              limit=50)
    # matches: list of (matched_title, score, index)
    # Map to SOCs
    scores_by_soc: Dict[str, float] = {}

    for matched_title, fuzzy_score, idx in matches:
        soc = search_space.iloc[idx]["soc"]
        # Normalize fuzzy_score to 0..1
        fscore = float(fuzzy_score) / 100.0
        # Description similarity: compare company title against occupation description text
        # vectorize company title and compare to all occs? That would be heavy per row.
        # Instead compare only to the specific SOC description.
        try:
            occ_row = occs.loc[occs["soc"] == soc].iloc[0]
            desc = occ_row["description"]
            if isinstance(desc, str) and len(desc) > 0:
                q_vec = tfidf_vectorizer.transform([company_title])
                d_vec = tfidf_vectorizer.transform([desc])
                cos = cosine_similarity(q_vec, d_vec)[0, 0]
            else:
                cos = 0.0
        except Exception:
            cos = 0.0

        # Priority boost based on SOC group
        boost = priority_boost(soc, occ_row.get("major_group") if 'occ_row' in locals() else None)

        # Combined score (weights tuned empirically)
        score = 0.70 * fscore + 0.25 * cos + boost * 0.05
        scores_by_soc[soc] = max(scores_by_soc.get(soc, 0.0), score)

    # Sort by score
    ranked = sorted(scores_by_soc.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:TOP_K]


def interactive_select_for_title(company_title: str,
                                 occs: pd.DataFrame,
                                 search_space: pd.DataFrame,
                                 tfidf_matrix,
                                 tfidf_vectorizer) -> Optional[Occupation]:
    """
    Show top candidates and prompt user to select; show description then confirm.
    Returns Occupation or None if skipped.
    """
    ranked = rank_candidates(company_title, occs, search_space, tfidf_matrix, tfidf_vectorizer)

    if not ranked:
        print(f"\nNo candidates found for: {company_title!r}. You can type 's' to skip or 'b' to go back.")
        return None

    while True:
        print("\n" + "=" * 80)
        print(f"Company Title: {company_title}")
        print("-" * 80)
        for i, (soc, score) in enumerate(ranked, start=1):
            row = occs.loc[occs["soc"] == soc].iloc[0]
            print(f"[{i}] {row['soc']} — {row['title']}")
        print("[b] Back to previous title")
        print("[s] Skip this title")
        print("[m] Manual SOC code entry")
        print("[q] Quit program")

        choice = input(
            "Select a number, 'b' to go back, 's' to skip, 'm' for manual SOC code, or 'q' to quit: "
        ).strip().lower()

        if choice == "q":
            print("Exiting program...")
            sys.exit(0)

        if choice == "b":
            return None  # go back signal to caller

        if choice == "s":
            return None  # skip
        
        if choice == "m":
            manual_code = input("Enter SOC code manually (e.g., 11-1011.00): ").strip()
            # Use occs DataFrame instead of onet_df, which is not defined here
            if manual_code in occs["soc"].values:
                row = occs.loc[occs["soc"] == manual_code].iloc[0]
                soc_code = row["soc"]
                soc_title = row["title"]
                description = row["description"]
                print(f"\nSelected {soc_code} — {soc_title}")
                print(f"Description: {description}\n")
                confirm = input("Confirm this selection? (y/n): ").strip().lower()
                if confirm == "y":
                    return Occupation(soc=soc_code, title=soc_title, description=description, major_group=row.get("major_group"))
                else:
                    continue  # back to menu
            else:
                print("Invalid SOC code. Try again.")
                continue

    # Remove this block; digit choices are handled below using ranked and occs

        if not choice.isdigit():
            print("Please enter a valid number, 'b', or 's'.")
            continue

        idx = int(choice)
        if not (1 <= idx <= len(ranked)):
            print("Invalid selection. Try again.")
            continue

        soc = ranked[idx - 1][0]
        row = occs.loc[occs["soc"] == soc].iloc[0]
        # Show description and ask to confirm
        print("\n" + "-" * 80)
        print(f"Selected: {row['soc']} — {row['title']}")
        print("-" * 80)
        desc = (row["description"] or "").strip()
        if len(desc) == 0:
            desc = "(No description available.)"
        print(desc)
        print("-" * 80)
        confirm = input("Use this occupation? [y/n]: ").strip().lower()
        if confirm in ("y", "yes"):
            return Occupation(soc=row["soc"], title=row["title"], description=row["description"], major_group=row.get("major_group"))
        else:
            # loop to reselect
            continue


def main():
    parser = argparse.ArgumentParser(description="Map company job titles to O*NET SOC codes.")
    parser.add_argument("--input", "-i", default="Company Job Titles.xlsx", help="Path to the company titles Excel file.")
    parser.add_argument("--occ", "-o", default="Occupation Data.xlsx", help="Path to the O*NET occupation data Excel file.")
    parser.add_argument("--alt", "-a", default="Alternate Titles.xlsx", help="Path to the O*NET alternate titles Excel file.")
    parser.add_argument("--out", "-O", default=None, help="Path to save the output Excel file (default: '<input> - Mapped.xlsx').")
    args = parser.parse_args()

    # Load files
    comp_df = safe_read_excel(args.input)
    occ_df = safe_read_excel(args.occ)
    alt_df = safe_read_excel(args.alt)

    # Detect company title column
    comp_title_col = detect_company_title_column(comp_df)

    # Build occupation index and search space
    occs, search_titles, search_space = build_occupation_index(occ_df, alt_df)

    # Prepare TF-IDF on all occupation descriptions (fit once)
    corpus = occs["description"].fillna("").astype(str).tolist()
    # Include titles too, to give the vectorizer vocabulary across both
    titles_corpus = occs["title"].astype(str).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(corpus + titles_corpus)  # we don't use this matrix directly, but it fits the vocabulary

    # Iterate over company titles
    results = []
    titles = comp_df[comp_title_col].dropna().astype(str).tolist()
    total = len(titles)

    i = 0
    while i < len(titles):
        company_title = titles[i].strip()
        if not company_title:
            i += 1
            continue

        # ==== Progress header (add these three lines) ====
        print("\n" + "="*80)
        print(f"Occupation {i+1}/{total}  —  {company_title}")
        print("-"*80)
        # =================================================

        try:
            occ = interactive_select_for_title(company_title, occs, search_space, tfidf_matrix, vectorizer)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            sys.exit(1)

        if occ is None:
            # Ask whether to go back or skip
            back_or_skip = input("Type 'b' to go back to previous title, or press Enter to skip: ").strip().lower()
            if back_or_skip == "b" and i > 0:
                i -= 1
                continue
            else:
                # skip current
                results.append({"Company Job Title": company_title, "O*NET-SOC Code": "", "O*NET Title": "", "Job Description": ""})
                i += 1
                continue

        # store confirmed selection
        results.append({
            "Company Job Title": company_title,
            "O*NET-SOC Code": occ.soc,
            "O*NET Title": occ.title,
            "Job Description": occ.description
        })
        i += 1

    # Build output DataFrame merged with original rows (preserve original order)
    out_df = pd.DataFrame(results)

    # If the input file has more columns (e.g., notes), merge on the title column when exact match
    try:
        merged = comp_df.merge(out_df,
                               left_on=comp_title_col,
                               right_on="Company Job Title",
                               how="left",
                               suffixes=("", " (mapped)"))
        merged.drop(columns=["Company Job Title"], inplace=True)
        final_df = merged
    except Exception:
        final_df = out_df

    # Save the output
    if args.out is None:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base} - Mapped.xlsx"
    else:
        output_path = args.out

    try:
        final_df.to_excel(output_path, index=False)
        print(f"\nSaved results to: {output_path}")
    except PermissionError:
        # Auto-rename and try again
        new_path = auto_rename_if_locked(output_path)
        try:
            final_df.to_excel(new_path, index=False)
            print(f"\nThe output file appeared to be locked. Saved instead to: {new_path}")
        except Exception as e:
            print(f"Failed to save the output even after renaming: {e}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Failed to save the output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

    print()
    print("   WORK LOCOMOTION: Make Potential Actual.")
    print()

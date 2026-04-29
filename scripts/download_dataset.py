"""
Phase 1 — Dataset Download
Downloads legal Q&A text from HuggingFace using datasets library (v4.x compatible).
Primary source: pile-of-law r_legaladvice (via direct parquet files)
Fallback:       eureka-research/legal-qa (publicly available, no script required)
Saves to data/legal_corpus.jsonl (up to 200K rows)
"""
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "legal_corpus.jsonl"
MAX_ROWS = 200_000


def try_pile_of_law():
    """Try loading pile-of-law directly from parquet files (no loading script)."""
    from datasets import load_dataset
    print("📥 Trying pile-of-law parquet files directly …")
    # Use the parquet files directly, bypassing the loading script
    ds = load_dataset(
        "parquet",
        data_files={
            "train": "hf://datasets/pile-of-law/pile-of-law/data/r_legaladvice/train/*.parquet"
        },
        split="train",
        streaming=True,
    )
    return ds


def try_eureka_legal():
    """Fallback: eureka-research/legal-qa — standard dataset, no script."""
    from datasets import load_dataset
    print("📥 Falling back to eureka-research/legal-qa …")
    ds = load_dataset("eureka-research/legal-qa", split="train", streaming=True)
    return ds


def try_lex_glue():
    """Second fallback: lex_glue scotus — legal court opinions."""
    from datasets import load_dataset
    print("📥 Falling back to lex_glue / scotus …")
    ds = load_dataset("lex_glue", "scotus", split="train", streaming=True)
    return ds


def write_dataset(ds, field_names: list[str], append: bool = False):
    """Write dataset rows to JSONL, trying multiple field names."""
    rows_written = 0
    mode = "a" if append else "w"
    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        for row in ds:
            if rows_written >= MAX_ROWS:
                break
            # Try known text field names
            text = ""
            for fn in field_names:
                val = row.get(fn, "")
                if isinstance(val, str) and len(val) > 100:
                    text = val.strip()
                    break
            if not text:
                continue
            record = {
                "text": text,
                "url": row.get("url", row.get("source", "")),
                "created_timestamp": str(row.get("created_timestamp", row.get("date", ""))),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1
            if rows_written % 10_000 == 0:
                print(f"   Written {rows_written:,} rows …")
    return rows_written


def try_lex_glue_ecthr():
    """lex_glue ecthr_a — European Court of Human Rights cases."""
    from datasets import load_dataset
    print("📥 Loading lex_glue / ecthr_a …")
    ds = load_dataset("lex_glue", "ecthr_a", split="train", streaming=True)
    return ds, ["text"]


def try_lex_glue_eurlex():
    """lex_glue eurlex — EU legislation documents."""
    from datasets import load_dataset
    print("📥 Loading lex_glue / eurlex …")
    ds = load_dataset("lex_glue", "eurlex", split="train", streaming=True)
    return ds, ["text"]


def try_lex_glue_ledgar():
    """lex_glue ledgar — US contract provisions."""
    from datasets import load_dataset
    print("📥 Loading lex_glue / ledgar …")
    ds = load_dataset("lex_glue", "ledgar", split="train", streaming=True)
    return ds, ["text"]


def try_lex_glue_unfair_tos():
    """lex_glue unfair_tos — Terms of service clauses."""
    from datasets import load_dataset
    print("📥 Loading lex_glue / unfair_tos …")
    ds = load_dataset("lex_glue", "unfair_tos", split="train", streaming=True)
    return ds, ["text"]


def try_caselaw():
    """Harvard Caselaw Access Project — US court opinions."""
    from datasets import load_dataset
    print("📥 Loading caselaw access project …")
    ds = load_dataset("free-law/courtlistener-opinions", split="train", streaming=True)
    return ds, ["text", "plain_text"]


def main():
    # Append-friendly: collect rows across multiple sources to reach ~480MB
    total_rows = 0
    total_size = 0

    # Ordered list of (loader_fn, field_names)
    all_sources = [
        (lambda: (try_lex_glue()[0], ["text"]), ["text"]),
        (try_lex_glue_ecthr, None),
        (try_lex_glue_eurlex, None),
        (try_lex_glue_ledgar, None),
        (try_lex_glue_unfair_tos, None),
        (try_caselaw, None),
    ]

    append_mode = OUTPUT_FILE.exists()
    if append_mode:
        total_size = OUTPUT_FILE.stat().st_size
        print(f"📂 Existing corpus found: {total_size/1024/1024:.1f} MB — appending …")

    TARGET_BYTES = 480 * 1024 * 1024  # 480MB

    for source_fn, _ in all_sources:
        if total_size >= TARGET_BYTES:
            break
        try:
            result = source_fn()
            ds, fields = result
            rows = write_dataset(ds, fields, append=True)
            total_rows += rows
            total_size = OUTPUT_FILE.stat().st_size
            size_mb = total_size / 1024 / 1024
            print(f"   → Total: {total_rows:,} rows | {size_mb:.1f} MB")
            if total_size >= TARGET_BYTES:
                break
        except Exception as e:
            print(f"   ⚠️  Source failed: {e}")

    if total_rows > 0 or OUTPUT_FILE.exists():
        size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
        print(f"\n✅ Final corpus: {OUTPUT_FILE}  ({size_mb:.1f} MB)")
    else:
        print("\n❌ No data downloaded.")
        sys.exit(1)


if __name__ == "__main__":
    main()


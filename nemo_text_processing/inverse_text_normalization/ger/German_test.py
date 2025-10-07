#!/usr/bin/env python3
import argparse
import sys
import pynini
from pynini.lib import rewrite
# import pdb; pdb.set_trace()

# Try importing ElectronicFst from both taggers and verbalizers
try:
    from nemo_text_processing.inverse_text_normalization.ger.taggers.electronic import ElectronicFst as Tagger
except ImportError:
    Tagger = None
try:
    from nemo_text_processing.inverse_text_normalization.ger.verbalizers.electronic import ElectronicFst as Verbalizer
except ImportError:
    Verbalizer = None

def main():
    parser = argparse.ArgumentParser(description="Test German ElectronicFst (email, URLs, etc.)")
    parser.add_argument("--lang", default="ger", help="Language code (default: ger)")
    parser.add_argument("--verbose", action="store_true", help="Print debug output")
    parser.add_argument("--text", required=True, help="Input text to test")
    args = parser.parse_args()

    # Ensure both tagger and verbalizer are available
    if Tagger is None or Verbalizer is None:
        print("❌ Could not import both tagger and verbalizer ElectronicFst.")
        sys.exit(1)

    if args.text is None:
        print("❌ Error: --text argument is required and must not be None.")
        sys.exit(1)

    # 1st: applying tagger to raw input
    try:
        tagged = rewrite.top_rewrite(args.text, Tagger().fst)
        if args.verbose:
            print(f"[INPUT]: {args.text}")
            print(f"[TAGGED]: {tagged}")
        else:
            print(tagged)
    except rewrite.Error:
        print(f"⚠ No match for input (tagger): {args.text}")
        return

    # 2nd: applying verbalizer to tagged output
    try:
        verbalized = rewrite.top_rewrite(tagged, Verbalizer().fst)
        if args.verbose:
            print(f"[VERBALIZED]: {verbalized}")
        else:
            print(verbalized)
    except rewrite.Error:
        print(f"⚠ No match for tagged output (verbalizer): {tagged}")

if __name__ == "__main__":
    main()

# import pdb; pdb.set_trace()
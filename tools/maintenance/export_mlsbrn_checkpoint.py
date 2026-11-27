#!/usr/bin/env python3
"""Export the tensors and class names needed by MLS-BRN inference."""

import argparse
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Original training checkpoint")
    parser.add_argument("output", type=Path, help="New inference checkpoint")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    import torch

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", mmap=True, weights_only=True)
    state = checkpoint["state_dict"]
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        parser.error("Expected a nonempty MLS-BRN state_dict of tensors")
    classes = checkpoint.get("meta", {}).get("CLASSES")
    if classes is not None and (
            not isinstance(classes, (list, tuple)) or not all(isinstance(name, str) for name in classes)):
        parser.error("Expected string class names in checkpoint metadata")
    meta = {"CLASSES": tuple(classes)} if classes is not None else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".mlsbrn-", dir=args.output.parent) as temporary:
        torch.save({"state_dict": state, "meta": meta}, temporary, pickle_protocol=2)
        temporary.flush()
        os.link(temporary.name, args.output)
    print(f"{args.output}: {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

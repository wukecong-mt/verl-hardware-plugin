#!/usr/bin/env python3
"""Reproduce MUSA IPC storage growth from repeated tensor serialization."""

import argparse
import gc
import os


def report(torch, tag):
    free, total = torch.musa.mem_get_info()
    print(
        f"[IPC-SERIALIZER-REPRO] {tag}: "
        f"allocated={torch.musa.memory_allocated()/2**30:.2f} "
        f"reserved={torch.musa.memory_reserved()/2**30:.2f} "
        f"used={(total-free)/2**30:.2f}/{total/2**30:.2f} GiB",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--serialize-copies", type=int, default=8)
    parser.add_argument("--tensor-mb", type=int, default=512)
    args = parser.parse_args()

    import torch
    from sglang.srt.utils import MultiprocessingSerializer

    if not torch.musa.is_available():
        raise SystemExit("MUSA is unavailable")
    torch.musa.set_device(0)
    elements = args.tensor_mb * 1024 * 1024 // 2  # BF16

    for round_id in range(args.rounds):
        tensor = torch.empty(elements, dtype=torch.bfloat16, device="musa")
        tensor.fill_(round_id % 2)
        # VERL's request construction serializes the same LocalSerializedTensor
        # list once per TP rank.  This is the critical repeated IPC export.
        serialized = [
            MultiprocessingSerializer.serialize(tensor)
            for _ in range(args.serialize_copies)
        ]
        del tensor, serialized
        gc.collect()
        torch.musa.empty_cache()
        report(torch, f"round {round_id} after release")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "0")
    main()

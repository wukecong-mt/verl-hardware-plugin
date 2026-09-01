#!/usr/bin/env python3
"""Minimal two-consumer MUSA IPC bucket lifetime reproducer.

The producer sends each MUSA bucket through two multiprocessing queues (target
and draft), matching VERL's dual-model weight-sync topology.  It records the
producer allocator after both consumers acknowledge deletion.  A monotonic
increase indicates that exported IPC storage/ref-counters are being retained.
"""

import argparse
import gc
import multiprocessing as mp
import os
import queue
import time


def _mem(torch, tag):
    dev = torch.musa.current_device()
    alloc = torch.musa.memory_allocated(dev) / 2**30
    reserved = torch.musa.memory_reserved(dev) / 2**30
    free, total = torch.musa.mem_get_info(dev)
    used = (total - free) / 2**30
    print(f"[IPC-REPRO] {tag}: allocated={alloc:.2f} reserved={reserved:.2f} used={used:.2f}/{total/2**30:.2f} GiB", flush=True)


def consumer(name, in_q, ack_q):
    import torch

    torch.musa.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    while True:
        item = in_q.get()
        if item is None:
            ack_q.put((name, "done"))
            return
        bucket, tensor = item
        # Touch the tensor to force IPC mapping, then release all Python refs.
        _ = tensor.numel()
        del tensor, item
        gc.collect()
        ack_q.put((name, bucket))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", type=int, default=10)
    ap.add_argument("--bucket-mb", type=int, default=256)
    args = ap.parse_args()

    import torch

    if not torch.musa.is_available():
        raise SystemExit("MUSA is unavailable")
    torch.musa.set_device(0)
    ctx = mp.get_context("spawn")
    target_q, draft_q, ack_q = ctx.Queue(2), ctx.Queue(2), ctx.Queue(4)
    target = ctx.Process(target=consumer, args=("target", target_q, ack_q))
    draft = ctx.Process(target=consumer, args=("draft", draft_q, ack_q))
    target.start(); draft.start()

    elements = args.bucket_mb * 1024 * 1024 // 2  # BF16
    for bucket in range(args.buckets):
        tensor = torch.empty(elements, dtype=torch.bfloat16, device="musa")
        tensor.fill_(bucket % 2)
        _mem(torch, f"bucket {bucket} before send")
        target_q.put((bucket, tensor))
        draft_q.put((bucket, tensor))
        del tensor
        # Wait until both target and draft have dropped their IPC views.
        seen = set()
        while len(seen) < 2:
            name, value = ack_q.get(timeout=300)
            if value == bucket:
                seen.add(name)
        gc.collect()
        torch.musa.empty_cache()
        _mem(torch, f"bucket {bucket} after both consumers released")

    target_q.put(None); draft_q.put(None)
    target.join(300); draft.join(300)
    print(f"[IPC-REPRO] exitcodes target={target.exitcode} draft={draft.exitcode}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

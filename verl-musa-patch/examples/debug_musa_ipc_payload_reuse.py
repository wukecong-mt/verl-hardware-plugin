#!/usr/bin/env python3
"""Check whether deserializing one MUSA IPC payload twice leaks its producer storage."""

import argparse
import gc
import multiprocessing as mp


def consumer(payload_q, ack_q, deserialize_count, materialize_once, plugin_cache):
    import torch
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    torch.musa.set_device(0)
    monkey_patch_torch_reductions()
    payload = payload_q.get()
    if plugin_cache:
        from verl_musa_patch.sglang import _install_mtp_ipc_tensor_cache

        _install_mtp_ipc_tensor_cache()
        named_tensors = MultiprocessingSerializer.deserialize(payload)
        for index in range(deserialize_count):
            tensor = named_tensors[0][1].get(0)
            ack_q.put((index, float(tensor[0].item())))
            del tensor
        del named_tensors
    elif materialize_once:
        tensor = MultiprocessingSerializer.deserialize(payload)
        for index in range(deserialize_count):
            model_input = tensor.to(torch.musa.current_device())
            ack_q.put((index, float(model_input[0].item())))
            del model_input
        del tensor
    else:
        for index in range(deserialize_count):
            tensor = MultiprocessingSerializer.deserialize(payload)
            value = float(tensor[0].item())
            del tensor
            gc.collect()
            torch.musa.synchronize()
            ack_q.put((index, value))
    del payload
    gc.collect()


def report(torch, tag):
    free, total = torch.musa.mem_get_info()
    print(
        f"[IPC-PAYLOAD-REUSE] {tag}: allocated={torch.musa.memory_allocated()/2**30:.2f} "
        f"reserved={torch.musa.memory_reserved()/2**30:.2f} "
        f"used={(total-free)/2**30:.2f}/{total/2**30:.2f} GiB",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--deserialize-count", type=int, choices=(1, 2), required=True)
    parser.add_argument("--materialize-once", action="store_true")
    parser.add_argument("--plugin-cache", action="store_true")
    parser.add_argument("--tensor-mb", type=int, default=256)
    args = parser.parse_args()

    import torch
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    torch.musa.set_device(0)
    monkey_patch_torch_reductions()
    ctx = mp.get_context("spawn")
    elements = args.tensor_mb * 1024 * 1024 // 2

    for round_id in range(args.rounds):
        payload_q, ack_q = ctx.Queue(1), ctx.Queue(args.deserialize_count)
        proc = ctx.Process(
            target=consumer,
            args=(
                payload_q,
                ack_q,
                args.deserialize_count,
                args.materialize_once,
                args.plugin_cache,
            ),
        )
        proc.start()
        tensor = torch.full((elements,), round_id, dtype=torch.bfloat16, device="musa")
        if args.plugin_cache:
            from sglang.srt.model_executor.model_runner import LocalSerializedTensor

            tensor_payload = MultiprocessingSerializer.serialize(tensor)
            payload = MultiprocessingSerializer.serialize(
                [("weight", LocalSerializedTensor(values=[tensor_payload]))]
            )
        else:
            payload = MultiprocessingSerializer.serialize(tensor)
        payload_q.put(payload)
        del tensor, payload
        for _ in range(args.deserialize_count):
            ack_q.get(timeout=120)
        proc.join(120)
        gc.collect()
        torch.musa.ipc_collect()
        torch.musa.empty_cache()
        report(torch, f"round {round_id} consumers={args.deserialize_count}")
        if proc.exitcode != 0:
            raise RuntimeError(f"consumer exited with {proc.exitcode}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

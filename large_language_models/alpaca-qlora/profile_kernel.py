import argparse

import torch
import torch.backends.cudnn as cudnn

from utils import QuantLinear


def bench_func_latency(func, args, warm_iters=100, iters=1000):
    cudnn.benchmark = True
    # Warm up
    for i in range(warm_iters):
        with torch.no_grad():
            func(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        with torch.no_grad():
            func(*args)
    end.record()
    torch.cuda.synchronize()
    print(f"Average inference time: {start.elapsed_time(end) / iters} ms")
    return start.elapsed_time(end) / iters


@torch.no_grad()
def quantize_pertensor_absmax(w, bit):
    maxq = 2 ** (bit - 1) - 1
    minq = -(2 ** (bit - 1))
    scale = w.abs().max() / maxq
    scale = scale.view(-1, 1)
    if not w.is_cuda:
        w = w.float()
    w_q = (w / scale).round().clamp_(minq, maxq)
    w_q = w_q.to(torch.int8)
    return w_q, scale


def main(args, device):
    if args.intermediate_size is None:
        args.intermediate_size = args.hidden_size * 4

    weight = torch.randn(args.intermediate_size, args.hidden_size)
    print("Weight shape: ", weight.shape)

    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size).to(device)

    print("Input shape: ", x.shape)

    if args.dtype == "int8":
        w_q, scale = quantize_pertensor_absmax(weight, bit=8)

        module = QuantLinear(args.hidden_size, args.hidden_size, bit=8)
        module.qweight.data = w_q
        module.scales.data = scale
    elif args.dtype == "fp16":
        module = torch.nn.Linear(args.hidden_size, args.hidden_size, bias=False).to(
            torch.half
        )
        module.weight.data = weight.to(torch.half)
        x = x.to(torch.half)
    elif args.dtype == "fp32":
        module = torch.nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        module.weight.data = weight.to(torch.float)

    module = module.to(device)
    module.eval()

    if args.nsys:
        for i in range(args.iters):
            with torch.no_grad():
                outputs = module(x)
        print("Nsys Done.")
    else:
        wait = args.wait_iters
        warmup = args.warm_iters
        active = args.iters
        repeat = 1
        schedule = torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        )
        # schedule = torch.profiler.schedule(
        #     wait=1, warmup=1, active=1, repeat=repeat
        # )
        tb_handler = torch.profiler.tensorboard_trace_handler(
            "./profile_outputs/profile_kernel/" + args.dtype + "/"
        )
        with torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=tb_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for i in range(wait + warmup + active):
                with torch.no_grad():
                    outputs = module(x)
                prof.step()
    print("Profile Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="int8")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--wait-iters", type=int, default=100)
    parser.add_argument("--warm-iters", type=int, default=500)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument(
        "--nsys",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(args, device)

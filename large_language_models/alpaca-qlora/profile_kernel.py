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
    w_q, scale = quantize_pertensor_absmax(weight, bit=8)

    int8_linear = QuantLinear(args.hidden_size, args.hidden_size, bit=8)
    int8_linear.qweight.data = w_q
    int8_linear.scales.data = scale

    fp32_linear = torch.nn.Linear(args.hidden_size, args.hidden_size, bias=False)
    fp32_linear.weight.data = weight.to(torch.float)

    fp16_linear = torch.nn.Linear(args.hidden_size, args.hidden_size, bias=False).to(
        torch.half
    )
    fp16_linear.weight.data = weight.to(torch.half)

    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size).to(device)

    print("Input shape: ", x.shape)

    int8_linear = int8_linear.to(device)
    int8_linear.eval()

    fp32_linear = fp32_linear.to(device)
    fp32_linear.eval()

    fp16_linear = fp16_linear.to(device)
    fp16_linear.eval()

    print("INT8")
    with torch.no_grad():
        _ = int8_linear(x)
    bench_func_latency(
        int8_linear.forward, (x,), warm_iters=args.warm_iters, iters=args.iters
    )

    print("FP16")
    bench_func_latency(
        fp16_linear.forward,
        (x.to(torch.half),),
        warm_iters=args.warm_iters,
        iters=args.iters,
    )

    print("FP32")
    bench_func_latency(
        fp32_linear.forward, (x,), warm_iters=args.warm_iters, iters=args.iters
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--warm-iters", type=int, default=500)
    parser.add_argument("--iters", type=int, default=1000)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(args, device)

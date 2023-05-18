import time
from tqdm import tqdm
import warnings
import os

import torch
import torch.distributed as dist


import transformers
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM


def calc_model_size(model: torch.nn.Module):
    numel_per_device = 0
    for n, p in model.named_parameters():
        if "embedding" in n:  # skip word-embedding
            continue
        numel_per_device += p.numel()
    return numel_per_device


def load_llama(model_name, checkpoint="", load_quant=False, bnb=False, groupsize=-1):
    def skip(*args, **kwargs):
        pass

    config = transformers.AutoConfig.from_pretrained(args.model_name)

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)

    print("Loading model... ", end="")
    if not load_quant:
        model = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, load_in_8bit=bnb, device_map="auto"
        )
        model.seqlen = 2048
    else:
        assert os.path.exists(checkpoint), "loading low-bit model requires checkpoint"
        # model = LLaMAClass(config)
        model = LlamaForCausalLM(config)

    torch.set_default_dtype(torch.float)
    model.eval()

    if load_quant:
        from utils import find_layers
        from utils import make_quant

        layers = find_layers(model)
        for name in ["lm_head"]:
            if name in layers:
                del layers[name]
        ckpt = torch.load(checkpoint)
        if ckpt["hyper_parameters"]["groupsize"] != groupsize:
            warnings.warn("Quantization group-size not-set / mismatch detected.")
        make_quant(model, ckpt["layers_bit"])
        print("Loading Quant model ...")
        model.load_state_dict(ckpt["model"])
        model.seqlen = 2048

    print("done.")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="decapoda-research/llama-7b-hf")
    parser.add_argument("--prompt", default="Tell me a story, no less than 200 words")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--run-iters", type=int, default=50)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="a checkpoint path from local storage",
    )
    parser.add_argument(
        "--load-quant",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--bnb",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)

    model = load_llama(
        args.model_name,
        load_quant=args.load_quant,
        checkpoint=args.checkpoint,
    )

    DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inputs = tokenizer.encode(args.prompt, return_tensors="pt").to(DEV)
    if args.batch_size != 1:
        inputs.repeat(args.batch_size, 1)

    torch.backends.cudnn.benchmark = True

    # warm-up
    for i in range(args.warmup_iters):
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
    output_new_tokens = outputs.shape[-1] - inputs.shape[-1]
    if args.max_new_tokens != output_new_tokens:
        warnings.warn(
            f"The token length of output {output_new_tokens} is less than max-new-tokens {args.max_new_tokens}"
        )
        args.max_new_tokens = outputs.shape[-1] - inputs.shape[-1]

    # run
    total_time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in tqdm(range(args.run_iters)):
        start.record()
        with torch.no_grad():
            outputs = model.generate(
                inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        end.record()
        torch.cuda.synchronize()
        total_time += start.elapsed_time(end)
    max_cuda_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    latency = total_time / args.run_iters / (args.batch_size * outputs.shape[-1])
    print(
        f"The latency of {args.model_name} is {latency:.3f} ms/token in input:output = {inputs.shape[-1]}:{args.max_new_tokens}, cuda_memory={max_cuda_memory_allocated:.3f}GB"
    )
    numel = calc_model_size(model)
    iters_per_seconds = args.run_iters / (total_time / 1000)
    tflops = (
        2
        * numel
        * args.batch_size
        * outputs.shape[-1]
        * iters_per_seconds
        / (1024**4)
    )  # why *8?
    print(
        f"The throughput of {args.model_name} is {tflops:.3f} TFlops in bs={args.batch_size}, cuda_memory={max_cuda_memory_allocated:.3f}GB"
    )

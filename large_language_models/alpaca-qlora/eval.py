import os

import torch
import torch.nn as nn


import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer


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
        model = model.to(DEV)

    print("done.")
    return model


def get_torch_rot(device, dtype):
    return torch.tensor(O, device=device, dtype=dtype)


@torch.no_grad()
def rotate_weight(W, axis):
    assert axis in (0, 1)
    O = get_torch_rot(W.device, torch.float32)
    if axis == 0:
        W.copy_(O.T @ W.float())
    else:
        W.copy_(W.float() @ O)


def rotate_layer(l):
    rotate_weight(l.self_attn.q_proj.weight, 1)
    rotate_weight(l.self_attn.k_proj.weight, 1)
    rotate_weight(l.self_attn.v_proj.weight, 1)
    rotate_weight(l.self_attn.o_proj.weight, 0)

    rotate_weight(l.mlp.gate_proj.weight, 1)
    rotate_weight(l.mlp.up_proj.weight, 1)
    rotate_weight(l.mlp.down_proj.weight, 0)


def rotate_model(model):
    rotate_weight(model.model.embed_tokens.weight, 1)
    for l in model.model.layers:
        rotate_layer(l)
    rotate_weight(model.lm_head.weight, 1)


@torch.no_grad()
def fuse_norm(norm, *linears):
    for i in linears:
        i.weight *= norm.weight[None]
    norm.weight.fill_(1)


def get_wikitext2(nsamples, seed, seqlen, model_name):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # from transformers import AutoTokenizer

    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


@torch.no_grad()
def llama_eval(model, testenc, dev, args):
    print("Evaluation...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    nlls = []
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        lm_logits = model(batch)[0]

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][
            :, 1:
        ].to(shift_logits.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float().cpu() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="decapoda-research/llama-7b-hf")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument("--random-rotate", default=False, action="store_true")
    parser.add_argument("--bnb", default=False, action="store_true")
    parser.add_argument("--load-quant", default=False, action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = load_llama(
        args.model_name,
        checkpoint=args.checkpoint,
        load_quant=args.load_quant,
        bnb=args.bnb,
    )

    # load dataloaders
    dataloader, testloader = get_wikitext2(
        nsamples=args.nsamples,
        seed=args.seed,
        model_name=args.model_name,
        seqlen=model.seqlen,
    )

    # evaluation
    print("The Perplexity on wikiText2: ")
    llama_eval(model, testloader, DEV, args)

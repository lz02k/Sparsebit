import argparse
import numpy as np
from scipy import stats

import torch

import transformers
from transformers import LlamaForCausalLM

from utils import find_layers, QuantLinear


def get_llama(model_name, checkpoint, device):
    print("Loading model...", end=" ", flush=True)

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)

    if checkpoint is None:
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    else:
        config = transformers.AutoConfig.from_pretrained(model_name)
        model = LlamaForCausalLM(config)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    torch.set_default_dtype(torch.float)
    model.eval()
    # model = model.to(device)

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


def quantize(module, layers_bit, name=""):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in layers_bit:
            print("Quantizing", name1, "to", layers_bit[name1], "bits")
            new_module = QuantLinear(
                tmp.in_features, tmp.out_features, bit=layers_bit[name1]
            )
            weight = tmp.weight.data
            qweight, scales = quantize_pertensor_absmax(weight, layers_bit[name1])
            new_module.qweight.data = qweight
            new_module.scales.data = scales

            if tmp.bias is not None:
                new_module.bias.data = tmp.bias.data
            setattr(
                module,
                attr,
                new_module,
            )
    for name1, child in module.named_children():
        quantize(child, layers_bit, name + "." + name1 if name != "" else name1)


def main(args, device):
    model = get_llama(args.model_name, args.checkpoint, device)

    if args.fuse_norm:
        print("Fusing norm...", end=" ", flush=True)
        fuse_norm(model.model.norm, model.lm_head)
        list(
            map(
                lambda l: [
                    fuse_norm(
                        l.input_layernorm,
                        l.self_attn.q_proj,
                        l.self_attn.k_proj,
                        l.self_attn.v_proj,
                    ),
                    fuse_norm(
                        l.post_attention_layernorm, l.mlp.gate_proj, l.mlp.up_proj
                    ),
                ],
                model.model.layers,
            )
        )
        print("done.")

    if args.random_rotate:
        print("Randomly rotating model...", end=" ", flush=True)
        rotate_model(model)
        print("done.")

    if args.save_rotated:
        assert args.fuse_norm and args.random_rotate
        print("Saving rotated model...", end=" ", flush=True)
        torch.save(model.state_dict(), args.model_name.split("/")[-1] + "-fn-rr.pt")
        print("done.")

    print("Quantizing model...", end=" ", flush=True)
    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    layers_bit = {name: args.bit for name in layers}
    quantize(model, layers_bit)
    print("done.")

    if args.save:
        print("Saving quantized model...", end=" ", flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "hyper_parameters": {"groupsize": args.groupsize},
                "layers_bit": layers_bit,
            },
            args.save,
        )
        print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="decapoda-research/llama-7b-hf")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--fuse-norm", default=False, action="store_true")
    parser.add_argument("--random-rotate", default=False, action="store_true")
    parser.add_argument("--save-rotated", default=False, action="store_true")
    parser.add_argument("--groupsize", type=int, default=-1)
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save quantized checkpoint under this name.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.random_rotate:
        print("Computing random rotation matrix...")
        if args.save_rotated:
            config = transformers.AutoConfig.from_pretrained(args.model_name)
            O = stats.ortho_group.rvs(config.hidden_size)
            np.save(args.model_name.split("/")[-1] + "-random-matrix.npy", O)
        else:
            O = np.load(args.model_name.split("/")[-1] + "-random-matrix.npy").astype(
                "float32"
            )
        print("done.")


    main(args, device)

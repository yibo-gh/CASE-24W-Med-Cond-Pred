
import os;

import torch;


def register_forward_hooks(model):
    hooks = []

    def hook_fn(module, input, output):
        name = module.__class__.__name__
        print(f"{name:<30} | input: {tuple(input[0].shape)} → output: {tuple(output.shape)}")

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只 hook 到底层（无子模块）层
            h = module.register_forward_hook(hook_fn)
            hooks.append(h)

    return hooks


def main() -> int:
    uri: str = "../modelOut/1743699483456029566-1-0.19074810224998387.pt";
    assert os.path.exists(uri);
    m: torch.nn.Module = torch.load(uri);
    register_forward_hooks(m);
    print("Model loaded");
    dev: torch.device = torch.device("cuda:0");
    # t::109 torch.Size([34, 256, 205]) torch.Size([34, 256, 208]) torch.Size([34, 256])
    m(torch.rand((36, 256, 208)).to(dev), torch.ones((36, 256, 208), dtype=int).to(dev), torch.ones((36, 256), dtype=int).to(dev), dev);
    return 0;


if __name__ == "__main__":
    main();


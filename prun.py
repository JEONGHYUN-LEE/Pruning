import torch


def conv_pruning(model, sparsity):
    with torch.no_grad():
        for name, module in model.named_modules():
            if type(module) == torch.nn.modules.conv.Conv2d:
                weight = module.weight.detach()
                sorted_weight = torch.sort(weight.reshape(-1))[0]
                arg_threshold = int(float(sorted_weight.shape[0])*sparsity)
                threshold = sorted_weight[arg_threshold]
                module.weight[module.weight <= threshold] = 0


def get_sparsity(model):
    num_zero = 0
    num_elem = 0
    for name, module in model.named_modules():
        if type(module) == torch.nn.modules.conv.Conv2d:
            num_zero += (module.weight == 0).sum()
            num_elem += module.weight.numel()
    return float(num_zero) / float(num_elem)


def get_layer_sparsity(model):
    for name, module in model.named_modules():
        if type(module) == torch.nn.modules.conv.Conv2d:
            num_zero = (module.weight == 0).sum()
            num_elem = module.weight.numel()
            print(name, float(num_zero)/float(num_elem))

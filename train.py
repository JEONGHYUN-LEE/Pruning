import torch


def retrain(loader, model, optimizer, criterion, device):
    model.train()
    # Generating Mask
    masks = []
    for name, module in model.named_modules():
        if type(module) == torch.nn.modules.conv.Conv2d:
            mask = torch.zeros(module.weight.shape).to(device)
            mask[module.weight != 0] = 1
            masks.append(mask)

    # General Training Procedure
    corrects = 0
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).long()

        optimizer.zero_grad()

        outputs = model(x)
        predicts = torch.argmax(outputs, 1).detach()
        corrects += (predicts == y).sum()

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # Masking Procedure
        with torch.no_grad():
            idx = 0
            for name, module in model.named_modules():
                if type(module) == torch.nn.modules.conv.Conv2d:
                    module.weight *= masks[idx]
                    idx += 1
    return float(corrects) / float(len(loader.dataset))


def train(loader, model, optimizer, criterion, device):
    model.train()
    corrects = 0
    for x,y in loader:
        x = x.to(device).float()
        y = y.to(device).long()

        optimizer.zero_grad()

        outputs = model(x)
        predicts = torch.argmax(outputs, 1).detach()
        corrects += (predicts == y).sum()

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return float(corrects)/float(len(loader.dataset))


def test(loader, model, device):
    model.eval()
    corrects = 0
    for x,y in loader:
        x = x.to(device).float()
        y = y.to(device).long()

        outputs = model(x)
        predicts = torch.argmax(outputs, 1).detach()
        corrects += (predicts == y).sum()

    return float(corrects)/float(len(loader.dataset))
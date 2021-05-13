import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet50()
device = torch.device('cpu')
model.to(device)
traced_module = torch.jit.script(model.eval())
frozen = torch.jit.freeze(traced_module)

frozen.save("resnet50.pt")

import torch
import torchvision
import torch.nn as nn


# An instance of your model.
model = torchvision.models.resnet18()
device = torch.device('cpu')
model.to(device)
traced_module = torch.jit.script(model.eval())
frozen = torch.jit.freeze(traced_module)

#print(frozen.graph)

model = nn.LSTMCell(10, 10)
input = torch.randn(1, 10)
h0 = torch.randn(1, 10)
c0 = torch.randn(1, 10)
x = model(input, (h0, c0))
print(x)

sc = torch.jit.script(model.eval())
fr = torch.jit.freeze(sc)
print (fr.graph)

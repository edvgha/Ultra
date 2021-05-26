import torch
import torch.nn as nn

class FCs(torch.nn.Module):
    def __init__(self):
        super(FCs, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )

    def forward(self, input):
        return self.fc(input)

model = FCs()
input = torch.randint(low = 0, high = 255, size = (32, 128)).type(torch.float)
output = model.forward(input)
print(output)
print(output.shape)
device = torch.device('cpu')
model.to(device)
sc = torch.jit.script(model.eval())
fr = torch.jit.freeze(sc)

# print (fr.graph)

#fr.save("LLD6.pt")

lstm = nn.LSTMCell(10, 20, bias=True)
sc = torch.jit.script(lstm.eval())
fr = torch.jit.freeze(sc)

print(fr.graph)

# input = torch.randn(3, 10)
# hx = torch.randn(3, 20)
# cx = torch.randn(3, 20)

# for _ in range(6):
#     hx, cx = lstm(input, (hx, cx))

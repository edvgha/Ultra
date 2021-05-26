import torch
import torch.nn as nn


print(torch.__version__)

def get_lstm_inputs(device='cpu', training=False, seq_length=3):
    input_shape = (3, 10) if seq_length is None else (seq_length, 3, 10)
    input = torch.randn(*input_shape, dtype=torch.float, device=device, requires_grad=training)
    hx = torch.randn(3, 20, dtype=torch.float, device=device, requires_grad=training)
    cx = torch.randn(3, 20, dtype=torch.float, device=device, requires_grad=training)
    module = nn.LSTMCell(10, 20).to(device, torch.float)  # Just to allocate weights with correct sizes
    if training:
        params = tuple(module.parameters())
    else:
        params = tuple(p.requires_grad_(False) for p in module.parameters())
    print(len(params))
    print(params[0].shape)
    print(params[1].shape)
    print(params[2].shape)
    print(params[3].shape)
    return (input, (hx, cx)) + params

class LSTM(torch.nn.Module):
    def lstm_cell(self, input, hidden, w_ih, w_hh, b_ih, b_hh):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        hx, cx = hidden
        gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy
    def forward(self, input, hidden, wih, whh, bih, bhh):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        outputs = []
        inputs = input.unbind(0)
        for seq_idx in range(len(inputs)):
            hidden = self.lstm_cell(inputs[seq_idx], hidden, wih, whh, bih, bhh)
            hy, _ = hidden
            outputs.append(hy)
        return hidden

sc = torch.jit.script(LSTM().eval())
fr = torch.jit.freeze(sc)
fr.save("LSTM.pt")
#print (fr.graph)

input, hidden, w_ih, w_hh, b_ih, b_hh = get_lstm_inputs()

zz = LSTM()
print(zz(input=input, hidden=hidden, wih=w_ih, whh=w_hh, bih=b_ih, bhh=b_hh))

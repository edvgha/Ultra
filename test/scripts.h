#pragma once 

const auto ir_if = R"JIT(
  def forward(self, x, y):
      if y :
        return 2 * x
      else:
        return x
)JIT";

const auto ir_if_1 = R"JIT(
  def forward(self, x, y):
      if y :
        return 2 + x
      return torch.sum(x, 1, True)
)JIT";

const auto ir_if_2 = R"JIT(
  def forward(self, x, y):
      if y > 0 :
        return 2 + x
      elif y < 0:
        return 2 * x
      else:
        return x - 4
)JIT";

const auto ir_for = R"JIT(
    def forward(x):
        z = torch.ones([2, 2])
        for i in [1, 2, 3, 4]:
            z += torch.randn([2, 2])
        return z
)JIT";

const auto ir_for_for = R"JIT(
    def forward(x):
        z = torch.ones([2, 2])
        for i in [1, 2, 3, 4]:
            for j in [1, 2, 3, 4]:
                z -= torch.randn([2, 2]) * i * j
        return z
)JIT";

const auto ir_for_for_if = R"JIT(
    def forward(x, y):
        z = torch.ones([2, 2])
        for i in [1, 2, 3, 4]:
            for j in [1, 2, 3, 4]:
                if y :
                    z -= torch.randn([2, 2]) * i * j
                else:
                    z += torch.randn([2, 2]) * i * j
        return z
)JIT";

const auto list_construct_script = R"JIT(
  def forward(self, a, b):
    return [a, b]
)JIT";

const auto list_unpack_script = R"JIT(
  def forward(self, a, b):
    c = [a, b]
    x, y = c
    z = x + y
    return z
)JIT";

const auto tuple_construct_script = R"JIT(
  def forward(self, a, b):
    return (a, b)
)JIT";

const auto add_script = R"JIT(
  def forward(self, a, b):
      return a + b
)JIT";

const auto reshape_script_1 = R"JIT(
  def forward(self, a: Tensor, shape: List[int]):
      b = a.reshape(shape)
      return b + b
)JIT";

const auto reshape_script_2 = R"JIT(
  def forward(self, a: Tensor, shape: List[int]):
      b = a.transpose(0, 1)
      return b.reshape(shape)
)JIT";

const auto flatten_script_1 = R"JIT(
  def forward(self, a: Tensor, start_dim: int, end_dim: int):
      b = torch.flatten(a, start_dim, end_dim)
      return b + b
)JIT";

const auto flatten_script_2 = R"JIT(
  def forward(self, a: Tensor, start_dim: int, end_dim: int):
      b = a.transpose(0, 1)
      return torch.flatten(b, start_dim, end_dim)
)JIT";

const auto aten_sum = R"JIT(
  def forward(self, input):
      return torch.sum(input)
)JIT";

const auto aten_sum_0 = R"JIT(
  def forward(self, input):
      return torch.sum(input, 0)
)JIT";

const auto aten_sum_1 = R"JIT(
  def forward(self, input):
      return torch.sum(input, 1)
)JIT";

const auto aten_sum_0_true = R"JIT(
  def forward(self, input):
      return torch.sum(input, 0, True)
)JIT";

const auto aten_sum_1_true = R"JIT(
  def forward(self, input):
      return torch.sum(input, 1, True)
)JIT";

const auto ir_lstm_cell = R"JIT(
)JIT";

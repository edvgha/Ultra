#pragma once 

const auto ir_if = R"JIT(
  def forward(self, x : Tensor, y : bool):
      if y :
        return x + x
      else:
        return x
)JIT";

const auto ir_if_1 = R"JIT(
  def forward(self, x : Tensor, y : bool):
      if y :
        return x + x
      return torch.sum(x, 1, True)
)JIT";

const auto ir_if_2 = R"JIT(
  def forward(self, x : Tensor, y : int):
      if y > 0 :
        return x + x
      elif y < 0:
        return x + x + x
      else:
        return x
)JIT";

const auto ir_if_3 = R"JIT(
  def forward(self, x : Tensor, y : bool, z : int):
      if z > 0 :
        if y:
          return x + x
        else:
          return x + x + x
      elif z < 0:
        return x + x + x + x
      else:
        return x + x + x + x + x
)JIT";

const auto ir_if_4 = R"JIT(
  def forward(self, x : Tensor, y : float):
      if y > 0 :
        return x + x
      elif y < 0:
        return x + x + x
      else:
        return x + x + x + x
)JIT";

const auto ir_for = R"JIT(
    def forward(self, x : Tensor, y : Tensor):
        z = y
        for i in [1, 2, 3, 4]:
            z += x
        return z
)JIT";

const auto ir_for_for = R"JIT(
    def forward(self, x : Tensor, y : Tensor):
        z = y
        for i in [1, 2, 3, 4]:
            for j in [1, 2, 3, 4]:
                z -= x
        return z
)JIT";

const auto ir_for_for_if = R"JIT(
    def forward(self, x : Tensor, y : Tensor, b : bool):
        z = y
        for i in [1, 2, 3, 4]:
            for j in [1, 2, 3, 4]:
                if b :
                    z -= x
                else:
                    z += x
        return z
)JIT";

const auto ir_list_pack_unpack = R"JIT(
  def forward(self, a : Tensor, b : Tensor):
    l = [a, b]
    x, y = l
    z = x + y
    return z
)JIT";

const auto ir_tuple_pack_unpack = R"JIT(
  def forward(self, a : Tensor, b : Tensor):
    t = (a, b)
    x, y = t
    z = x + y
    return z
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

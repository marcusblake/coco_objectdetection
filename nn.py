import torch
import torch.nn as nn

class ConvolutionalNetwork(nn.Module):
  def __init__(self):
    super(ConvolutionalNetwork, self).__init__()
    network_layers, _ = self.__initialize_network(network_architecture)
    self.network_layer_types = list(map(lambda tup : tup[0], network_layers))
    self.network_layers = torch.nn.ModuleList(list(map(lambda tup : tup[1], network_layers)))
    if DEBUG:
      print("Number of layers", len(self.network_layers))
      print(self.network_layers)

  def __initialize_network(self, network_layers, input_channel_size = 3):
    layers = []
    first_fc = True
    for value in network_layers:
      layer_type = value[0]
      parameters = value[1]
      if layer_type == "conv":
        kernel_size = parameters["kernel_size"]
        out_channel = parameters["output_channel"]
        stride = parameters["stride"]
        padding = parameters["padding"]
        layers.append(("conv", torch.nn.Conv2d(input_channel_size, out_channel, kernel_size,stride, padding)))
        input_channel_size = out_channel
      elif layer_type == "maxpool":
        kernel_size = parameters["kernel_size"]
        stride = parameters["stride"]
        layers.append(("maxpool", torch.nn.MaxPool2d(kernel_size, stride)))
        input_channel_size = out_channel
      elif layer_type == "linear":
        output = parameters["output"]
        input_size = input_channel_size
        if first_fc:
          input_size = input_channel_size * GRID_SIZE * GRID_SIZE
          first_fc = False
        layers.append(("linear", torch.nn.Linear(input_size, output)))
        input_channel_size = output
      elif layer_type == "repeat":
        count = parameters["count"]
        _layers = parameters["layers"]
        for _ in range(count):
          new_layers, input_channel_size = self.__initialize_network(_layers, input_channel_size = input_channel_size)
          layers.extend(new_layers)
      else:
        raise Exception(f"Unexpected type {layer_type} in json")

    return layers, input_channel_size

  def forward(self, x):
    first_fc = True
    for index, layer in enumerate(zip(self.network_layer_types, self.network_layers)):
      if DEBUG:
        print(index)
        print(x.size())
      layer_type = layer[0]
      torch_layer = layer[1]
      if layer_type == "conv":
        x = F.leaky_relu(torch_layer(x), inplace=False)
      elif layer_type == "maxpool":
        x = torch_layer(x)
      elif layer_type == "linear":
        if first_fc:
          x = x.view(-1, 1024 * GRID_SIZE * GRID_SIZE)
          first_fc = False
        x = F.leaky_relu(torch_layer(x), inplace=False)
      else:
        raise Exception(f"Invalid layer type")

    return x.view(-1, 7, 7, 9)
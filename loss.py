import torch
import torch.nn as nn
import json

class YoloLoss(nn.Module):
  def __init__(self, S = None, C = None, B = None):
    super(YoloLoss, self).__init__()
    self.mse = torch.nn.MSELoss(reduction="sum")
    with open("parameters.json") as f:
        self.parameters = json.load(f)
    
    self.lambda_coord = self.parameters["training_parameters"]["loss_function"]["lambda_coord"]
    self.lambda_noobj = self.parameters["training_parameters"]["loss_function"]["lambda_noobj"]
    self.S = self.parameters["model"]["grid_size"] if S is None else S
    self.C = len(self.parameters["model"]["classes"]) if C is None else C
    self.B = self.parameters["model"]["bounding_boxes"] if B is None else B

  def forward(self, predictions, labels):
    """
    Computes the custom loss for YOLO as described here https://arxiv.org/pdf/1506.02640.pdf.
    This video is helpful in understanding how to implement this loss function efficiently https://www.youtube.com/watch?v=n9_XyCGr-MI
    Input:
      predictions: PyTorch tensor representing the result predicted from the model. Will be a N * S * S * (C + 5 * B) tensor since there are N predictions and each prediction is of dimensions S * S * (C + 5 * B)
      labels: PyTorch tensor representing the actual result. N * S * S * (C + 5 * B) tensor since there are N predictions and each prediction is of dimensions S * S * (C + 5 * B)
    Output:
      Scalar representing the loss
    """

    object_exists = labels[..., self.C:self.C+1] # (N * S * S * 1)

    # Coordinate losses
    bbox_points = object_exists * labels[..., self.C+1:]
    pred_bbox_points = object_exists * predictions[..., self.C+1:]

    # Width/height losses
    sqrts = torch.sqrt(bbox_points[...,2:])

    # Use absolute value to prevent from taking square root of negative number and multiply by sign to make sure that gradient is in the correct direction
    pred_sqrts = torch.sign(pred_bbox_points[...,2:]) * torch.sqrt(torch.abs(pred_bbox_points[...,2:]) + 1e-6)

    x1 = torch.cat([bbox_points[...,:2], sqrts], dim=3)
    x2 = torch.cat([pred_bbox_points[...,:2], pred_sqrts], dim=3)
    bbox_loss = self.mse(
        x1.flatten(end_dim=-2), # Dimension is (N * S * S * 4), will be (N * S * S, 4)
        x2.flatten(end_dim=-2)
    )

    if self.parameters["debug"]:
      print("BBOXLOSS", bbox_loss)


    # Probability object P(object) losses
    pred_prob_objectexists = predictions[..., self.C:self.C+1]

    object_prob_loss = self.mse(
        torch.flatten(object_exists * object_exists),  # Dimension is (N * S * S * 1), will be (N * S * S)
        torch.flatten(object_exists * pred_prob_objectexists)
    )

    if self.parameters["debug"]:
      print("OBJECT PROB", object_prob_loss)

    # Probability no object
    object_doesntexist = torch.abs(1 - object_exists)
    objectnoexist_prob_loss = self.mse(
        torch.flatten(object_doesntexist * object_exists),
        torch.flatten(object_doesntexist * pred_prob_objectexists)
    )

    if self.parameters["debug"]:
      print("OBJECT NO EXIST", objectnoexist_prob_loss)

    # Class P(C | object) probability losses
    class_probs = labels[..., 0:self.C]
    pred_class_probs = predictions[..., 0:self.C]

    class_probs_loss = self.mse(
        torch.flatten(object_exists * class_probs, end_dim=-2), # Dimension is (N * S * S * C), will be (N * S * S, C)
        torch.flatten(object_exists * pred_class_probs, end_dim=-2)
    )

    if self.parameters["debug"]:
      print("CLASS PROBABILITY", class_probs_loss)

    # aggregate loss
    return self.lambda_coord * bbox_loss + object_prob_loss + self.lambda_noobj * objectnoexist_prob_loss + class_probs_loss
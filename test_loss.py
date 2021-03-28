import numpy as np
import unittest
import json
import random
import torch
from loss import YoloLoss


class YoloLossTests(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        self.loss = YoloLoss()
        with open("parameters.json") as f:
            parameters = json.load(f)
            self.grid_size = parameters["model"]["grid_size"]
            self.num_classes = len(parameters["model"]["classes"])
            self.bounding_boxes = parameters["model"]["bounding_boxes"]
            self.lambda_coord = parameters["training_parameters"]["loss_function"]["lambda_coord"]
            self.lambda_noobj = parameters["training_parameters"]["loss_function"]["lambda_noobj"]

    def test_prediction_same_target_returns_zero(self):
        # Arrange
        image1 = torch.rand(3, self.grid_size, self.grid_size, self.num_classes + 5 * self.bounding_boxes)
        image1[...,self.num_classes] = 1.0
        image2 = image1

        # Act
        output = self.loss(image2, image1)

        # Assert
        np.testing.assert_almost_equal(output, 0)


    def test_different_xy_coords_returns_expected_loss(self):
        # Arrange
        image1 = torch.rand(3, self.grid_size, self.grid_size, self.num_classes + 5 * self.bounding_boxes)
        image1[...,self.num_classes] = 1.0
        image2 = image1.clone()

        # Edit the x,y coordinates so that they are not the same for some of the cells so that the prediction is slightly different
        # and will generate a loss
        for i in range(self.grid_size):
            image2[1,i,2,self.num_classes+1] = image1[1,i,2,self.num_classes+1] + 5
            image2[1,i,2,self.num_classes+2] = image1[1,i,2,self.num_classes+2] - 2

        # Act
        output = self.loss(image2, image1)

        # Assert
        np.testing.assert_almost_equal(output, 1015)


    def test_different_wh_returns_expected_loss(self):
        # Arrange
        image1 = torch.rand(3, self.grid_size, self.grid_size, self.num_classes + 5 * self.bounding_boxes)
        image1[...,self.num_classes] = 1.0
        image1[1,:,:,self.num_classes] = 0.0 # No objects in this entire sample
        image2 = image1.clone()
        expected_loss = 0
        for i in range(3):
            image2[i,2,2,self.num_classes+3] = image1[i,2,2,self.num_classes+3] * 4
            image2[i,2,2,self.num_classes+4] = image1[i,2,2,self.num_classes+4] * 9
            expected_loss += image1[i,2,2,self.num_classes] * (image1[i,2,2,self.num_classes+3] + 4 * image1[i,2,2,self.num_classes+4])

        # Act
        output = self.loss(image2, image1)

        # Assert
        print("LOSS", output)
        print("EXPECTED", self.lambda_coord * expected_loss)
        np.testing.assert_almost_equal(output, self.lambda_coord * expected_loss, decimal=4)

    def test_noobj_and_incorrect_class_predictions_returns_expected_loss(self):
        # Arrange
        image1 = torch.rand(3, self.grid_size, self.grid_size, self.num_classes + 5 * self.bounding_boxes)
        image1[...,self.num_classes] = 1.0
        image2 = image1.clone() 

        image1[1,:,2:5,self.num_classes] = 0.0 # No objects in these 7 x 3 = 21 cells for this example. Is predicting objects in these cells although none are truly present
        expected_loss = 0
        for i in range(self.num_classes):
            image2[0,3,2,i] = image1[0,3,2,i] + (i+1)
            expected_loss += (i+1) ** 2
        
        expected_loss += self.lambda_noobj * 7 * 3
        
        # Act
        output = self.loss(image2, image1)

        # Assert
        print("LOSS", output)
        print("EXPECTED", expected_loss)
        np.testing.assert_almost_equal(output, expected_loss, decimal=4)

    def test_incorrect_class_probabilities_returns_expected_loss(self):
        # Arrange
        image1 = torch.rand(3, self.grid_size, self.grid_size, self.num_classes + 5 * self.bounding_boxes)
        image1[...,self.num_classes] = 1.0
        image2 = image1.clone() 

        expected_loss = 0
        for i in range(self.grid_size):
            num = random.uniform(0, 4)
            image2[1, i, i, self.num_classes] = num # for first sample, set all class probabilities along diagonal to random number
            expected_loss += (1 - num) ** 2
        
        # Act
        output = self.loss(image2, image1)

        # Assert
        print("LOSS", output)
        print("EXPECTED", expected_loss)
        np.testing.assert_almost_equal(output, expected_loss, decimal=4)

if __name__ == '__main__':
    unittest.main()
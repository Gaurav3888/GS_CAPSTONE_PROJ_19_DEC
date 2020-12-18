#!/usr/bin/env python
"""
model Unit tests Scripts
"""
#Section Imports all the required modules
import unittest

## refers to the model.py File shared
from model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(os.path.join("cs-train"),test=False)
        saved_model = os.path.join("cs-train","models","sl-france-0_1.joblib")
        self.assertTrue(os.path.exists(saved_model))

    def test_02_load(self):
        
        """
        test the train functionality
        """
                        
        ## train the model

        all_data, all_models = model_load()
        model = all_models['france']
		
        
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## ensure that a list can be passed        
        result = model_predict('france','2017','11','30',test=False)
        y_pred = result['y_pred']
        self.assertTrue(y_pred >= 0.0)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()

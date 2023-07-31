import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 CPU:str,
                 RAM:int,
                 Storage:int,
                 Storage_type:str,
                 GPU:str,
                 Screen:float
                 ):
        
        
        self.CPU=CPU
        self.RAM=RAM
        self.Storage=Storage
        self.Storage_type=Storage_type
        self.GPU=GPU
        self.Screen=Screen
        
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
            
                'CPU':[self.CPU],
                'RAM':[self.RAM],
                'Storage':[self.Storage],
                'Storage type':[self.Storage_type],
                'GPU':[self.GPU],
                'Screen':[self.Screen]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
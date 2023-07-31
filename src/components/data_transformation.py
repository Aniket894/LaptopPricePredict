import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Laptop', 'Status','Model','CPU','Storage type','GPU','Touch']
            numerical_cols = ['RAM','Storage','Screen']
            
            # Define the custom ranking for each ordinal variable
            CPU_categories = ['Intel Celeron','Microsoft SQ1','Qualcomm Snapdragon 7','Qualcomm Snapdragon 8','Mediatek MT8183','AMD 3015e',
                              'AMD 3015Ce','AMD 3020e', 'AMD Athlon','Intel Core M3','AMD Radeon 5','Intel Pentium','Apple M2','Apple M2 Pro',
                              'AMD Radeon 9','Intel Core i3','AMD Ryzen 3','Intel Evo Core i5','Intel Evo Core i7','Intel Evo Core i9',
                              'AMD Ryzen 5','Apple M1','Apple M1 Pro','Intel Core i5','Intel Core i7','AMD Ryzen 7','Intel Core i9','AMD Ryzen 9']

            Storage_type_categories = [ 'eMMC','SSD']

            GPU_categories =['A 370M','610 M','A 730M','GTX 1050','GTX 1070','GTX 1650','GTX 1660','MX 130','MX 330','MX 450','MX 550',
                             'P 500','Radeon Pro 5300M','Radeon Pro 5500M','Radeon RX 6600M','Radeon Pro RX 560X','RX 6500M','RX 6600M',
                             'RX 6700M','RX 6800S','RX 7600S','RTX 2050','RTX 2060','RTX 2070','RTX 2080','RTX 3000','RTX 3050','RTX 3060',
                             'RTX 3070','RTX 3080','RTX 4050','RTX 4060','RTX 4070','RTX 4080','RTX 4090','RTX A1000','RTX A2000','RTX A3000',
                             'RTX A5500','T 1000','T 1200','T 2000','T 500','T 550','T 600']
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[CPU_categories,Storage_type_categories,GPU_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Final Price'
            drop_columns = [target_column_name,'Laptop','Status','Touch','Model']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
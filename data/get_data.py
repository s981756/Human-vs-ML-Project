import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_student_data():

# -----------------------------------------------------------------------
    #copied directly from the UCI Machine Learning Repo package documentation
    from ucimlrepo import fetch_ucirepo
  
    # fetch dataset 
    student_performance = fetch_ucirepo(id=320) 
  
    # data (as pandas dataframes) 
    X = student_performance.data.features 
    y = student_performance.data.targets 
  
    # metadata 
    print(student_performance.metadata.uci_id) 
    print(student_performance.metadata.num_instances) 
    print(student_performance.metadata.additional_info.summary) 

    # variable information 
    print(student_performance.variables) 
    # -----------------------------------------------------------

    feature_names = student_performance.variables[student_performance.variables['role'] == 'Feature']['name'].tolist()
    target_name = student_performance.variables[student_performance.variables['role'] == 'Target']['name'].values
    
    df = pd.DataFrame(student_performance.data.features, columns=feature_names)
    # df[target_name] = student_performance.targets

    return df, target_name

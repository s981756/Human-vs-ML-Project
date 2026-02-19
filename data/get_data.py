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
    # print(student_performance.metadata.uci_id) 
    # print(student_performance.metadata.num_instances) 
    # print(student_performance.metadata.additional_info.summary) 

    # # variable information 
    # print(student_performance.variables) 
    # -----------------------------------------------------------
    # create DataFrame from features

    feature_names = student_performance.variables[student_performance.variables['role'] == 'Feature']['name'].tolist()
    target_name = student_performance.variables[student_performance.variables['role'] == 'Target']['name'].values[2]

    df = pd.DataFrame(student_performance.data.features, columns=feature_names)

    # attach the target column (extract correct series if targets is a DataFrame)
    targets = student_performance.data.targets
    try:
        if isinstance(targets, pd.DataFrame):
            if target_name in targets.columns:
                target_series = targets[target_name]
            else:
                # fall back to the last column if naming differs
                target_series = targets.iloc[:, -1]
        else:
            target_series = pd.Series(targets)

        # ensure alignment by index and assign the values
        df[target_name] = pd.Series(target_series).reset_index(drop=True)
    except Exception:
        pass

    # add `passing?` column computed from the target column
    if target_name in df.columns:
        df['passing?'] = df[target_name].apply(lambda x: 'Yes' if x >= 12 else 'No')

    print(target_name)

    return df, target_name

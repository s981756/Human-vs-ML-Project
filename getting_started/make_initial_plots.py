import os
from data.get_data import load_student_data
import matplotlib.pyplot as plt
import seaborn as sns

def make_plots(variable_1, variable_2):
    variable_1_label = variable_1.replace('_',' ')
    variable_2_label = variable_2.replace('_',' ')

    df, target_name = load_student_data()

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=variable_1,
        y=variable_1, 
        hue=target_name,
        style=target_name,
        s=90
    )

    plt.title(f"Factors that may affect student performance: {variable_1} vs. {variable_2}")
    plt.xlabel(f"{variable_1_label}")
    plt.ylabel(f"{variable_2_label}")
    plt.legend(title="Student Performance")
    plt.grid(True)
    
make_plots('studytime', 'G3')
make_plots('famrel', 'G3')
make_plots('absences', 'G3')
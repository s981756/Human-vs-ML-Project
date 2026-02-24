import os
from data.get_data import load_student_data
import matplotlib.pyplot as plt
import seaborn as sns

def make_plots(variable_1, variable_2):
    variable_1_label = variable_1.replace('_',' ')
    variable_2_label = variable_2.replace('_',' ')

    df, df['passing?'] = load_student_data()

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=variable_1,
        y=variable_2,
        hue=df["passing?"],
        style=df["passing?"],
        s=90
    )

    plt.title(f"Factors that may affect student performance: {variable_1} vs. {variable_2}")
    plt.xlabel(f"{variable_1_label}")
    plt.ylabel(f"{variable_2_label}")
    plt.legend(title="Passing?")
    plt.grid(True)

    # print(df)
    # print(df['passing?'])

#/workspaces/Human-vs-ML-Project/getting_started/plots
    plt.savefig(f"/workspaces/Human-vs-ML-Project/getting_started/plots/{variable_1}vs{variable_2}.png", dpi=150)
    plt.close()
    
# make_plots('studytime', 'famrel')
# make_plots('health', 'freetime')
# make_plots('absences', 'age')

make_plots('Dalc', 'Walc')

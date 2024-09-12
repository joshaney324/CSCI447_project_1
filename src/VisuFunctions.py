import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(matrix, filename):
    """
    Visualizes a 2x2 confusion matrix via a heatmap.
    """
    fig = plt.figure()
    plt.imshow(matrix, cmap="turbo")

    # Add colorbar
    plt.colorbar(label='Values')
    matrix_sum = np.sum(matrix)
    matrix_labels = [["True Pos", "False Pos"],["False Neg","True Neg"]]
    plt.xticks([0,1], ["Predicted True","Predicted False"])
    plt.yticks([0,1], ["Actual True","Actual False"])
    plt.title("Confusion Matrix")

    # Add annotations
    for x in range(2):
        for y in range(2):
            text = plt.text(x,y, matrix_labels[x][y] + "\n" + str(round(matrix[x][y]/matrix_sum*100,2)) + "%", ha='center', va='bottom', color='black')
            text = plt.text(x,y, matrix[x][y], ha='center', va='top', color='black')
    plt.savefig("../output/" + filename + ".svg", format='svg', dpi=1200)

def plot_avgs(avg_data, filename):
    """
    Plots a bar chart of a list of the percision, recall and accuracy averages of individual datasets.
    """
    
    # round percentages
    for i in range(len(avg_data)):
        for j in range(len(avg_data[i])):
            avg_data[i][j] = round(avg_data[i][j]*100, 2)

    labels = ["Precision", "Recall", "Accuracy"]
    colors = ["darkcyan", "gold", "darkred"]
    dataset_names = ["Breast Cancer Data", "Iris Data", "House Vote Data", "Soy Beans Data", "Glass Data"]
    x = np.arange(5)
    width = 0.75

    fig, ax = plt.subplots(layout='constrained')
    for i in range(3):
        avgs = [dataset[i] for dataset in avg_data]
        # offset to group bars at the same x-tick
        offset = (i - 1) * width
        perc = ax.barh(3*x + offset, avgs, width, label=labels[i], color=colors[i])
        ax.set_yticks(3*x, labels=dataset_names)
        ax.bar_label(perc)

    ax.set(xlabel='percentage', title='Average results for precision, recall and accuracy', xlim=(0, 140))
    ax.set_xticks(np.arange(0, 101, step=20))  
    ax.legend(loc="upper right")
    plt.savefig("../output/" + filename + ".svg", format='svg', dpi=1200)
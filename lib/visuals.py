import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

class ClassifierVisual:
    def __init__(self, X, y, classifier):
        self.X_set = X
        self.y_set = y
        self.classifier = classifier

    def visualize(self, title = 'Title', xlab = 'X Label', ylab = 'Y Label'):
        X1, X2 = np.meshgrid(np.arange(start = self.X_set[:, 0].min() - 1, stop = self.X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = self.X_set[:, 1].min() - 1, stop = self.X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, self.classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(self.y_set)):
            plt.scatter(self.X_set[self.y_set == j, 0], self.X_set[self.y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.show()

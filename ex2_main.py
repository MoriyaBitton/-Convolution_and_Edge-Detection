from ex2_utils import *
import matplotlib.pyplot as plt
from random import randrange
import numpy as np
import cv2


def presentation(plots, titles):
    n = len(plots)

    if n == 1:
        plt.imshow(plots[0], cmap='gray')
        plt.title(titles[0])
        plt.show()
        return

    if n == 2:
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    elif n % 2 == 0:
        fig = plt.figure(figsize=(12, 8))
        plt.gray()
        for i in range(n):
            ax = fig.add_subplot(2, 2, i + 1)
            ax.imshow(plots[i])
            ax.title.set_text(titles[i])
        plt.show()
        return
    else:
        fig, ax = plt.subplots(1, n, figsize=(4 * n, 4))

    for i in range(n):
        ax[i].set_title(titles[i])
        ax[i].imshow(plots[i], cmap='gray')
        plt.tight_layout()
    plt.show()


def conv1Demo():
    n = randrange(10)
    Signals, Kernels = list(), list()

    for i in range(n):
        Signals.append(np.random.randint(5, size=10))
        Kernels.append(np.random.randint(5, size=10))

    good_ans = 0
    for i in range(n):
        for j in range(n):
            np_convolution = np.convolve(Signals[i], Kernels[j])
            my_convolution = conv1D(Signals[i], Kernels[j])
            if np_convolution.all() == my_convolution.all():
                good_ans += 1

    if good_ans == len(Signals) * len(Kernels):
        print("conv1Demo: All test are passed!\nGood Job!\n")
    else:
        print("conv1Demo: Some of test aren't passed!\nTry Again!\n")


def conv2Demo():
    img = cv2.imread('pool_balls.jpeg', 0)
    Kernels = [np.array([[-1, 1], [1, 1]], dtype=np.float64),
               np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64),
               np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
               np.array([[0., 0.25, 0.5, 0.75, 1], [0.2, 0.4, 0.6, 0.8, 1],
                         [1., 1.25, 1.5, 1.75, 2], [1.2, 1.4, 1.6, 1.8, 2]], dtype=np.float64)]

    for i in range(4):
        if Kernels[i].sum() != 0:
            Kernels[i] /= (Kernels[i].sum())

    good_ans = 0
    for kernel in Kernels:
        cv2_convolution = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        my_convolution = conv2D(img, kernel)
        if cv2_convolution.all() == my_convolution.all():
            good_ans += 1

    if good_ans == len(Kernels):
        print("conv2Demo: All test are passed!\nGood Job!\n")
    else:
        print("conv1Demo: Some of test aren't passed!\nTry Again!\n")


def derivDemo():
    img = cv2.imread('pool_balls.jpeg', 0)
    direction, magnitude, x_der, y_der = convDerivative(img)

    plots = [direction, magnitude, x_der, y_der]
    titles = ["Direction", "Magnitude", "X Derivative", "Y Derivative"]
    presentation(plots=plots, titles=titles)
    print("derivDemo: Good Job!\n")


def blurDemo():
    img = cv2.imread("coins.jpg", 0)
    kernel_size = 5
    plots = [img, blurImage2(img, kernel_size)]
    titles = ['Image - non blurring', 'CV2 Blur']
    presentation(plots=plots, titles=titles)
    print("blurDemo: Good Job!\n")


def edgeDetectionSobelDemo():
    img = cv2.imread("boxman.jpg", 0)
    opencv_solution, my_solution = edgeDetectionSobel(img, thresh=0.1)
    plots = [img, opencv_solution, my_solution]
    titles = ['Original Image', 'CV2 Sobel', 'My Sobel']
    presentation(plots=plots, titles=titles)
    print("edgeDetectionSobelDemo: Good Job!\n")


def edgeDetectionZeroCrossingLOGDemo():
    img = cv2.imread("boxman.jpg", 0)
    edge_matrix = edgeDetectionZeroCrossingLOG(img)
    presentation(plots=[edge_matrix], titles=["Laplacian of Gaussian\nZero Crossing Edge Detection"])
    print("edgeDetectionZeroCrossingLOGDemo: Good Job!\n")


def edgeDetectionCannyDemo():
    img = cv2.imread("pool_balls.jpeg", 0)
    cv2_canny, my_canny = edgeDetectionCanny(img, 50, 100)
    plots = [img, cv2_canny, my_canny]
    titles = ['Original Image', 'CV2 Canny Edge Detection', 'My Canny Edge Detection']
    presentation(plots=plots, titles=titles)
    print("edgeDetectionCannyDemo: Good Job!\n")


def edgeDemo():
    edgeDetectionSobelDemo()
    edgeDetectionZeroCrossingLOGDemo()
    edgeDetectionCannyDemo()


def houghDemo():
    img = cv2.imread('coins.jpg', 0)
    min_radius, max_radius = 10, 20

    circles = houghCircle(img, min_radius, max_radius)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    for x, y, radius in circles:
        circles_plots = plt.Circle((x, y), radius, color='r', fill=False)
        ax.add_artist(circles_plots)
    plt.title("Circle\nMy houghCircle Implementation")
    plt.show()
    print("houghDemo: Good Job!\n")


def main():
    print("ID: 316451749\nHave Fun! :)\n")
    conv1Demo()
    conv2Demo()
    derivDemo()
    blurDemo()
    edgeDemo()
    houghDemo()


if __name__ == '__main__':
    main()

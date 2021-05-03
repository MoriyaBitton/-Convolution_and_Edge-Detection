import numpy as np
import cv2


def normalizeImg(img: np.ndarray) -> np.ndarray:
    """
    Normalize Image in range (-1,1)
    :param img:
    :return:
    """
    if img.max() > 1:
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array --> np.convolve(signal, kernel, ’full’)
    """
    isSignalZeros = np.zeros(inSignal.size + (kernel1.size - 1) * 2)
    for i in range(inSignal.size):
        isSignalZeros[i + kernel1.size - 1] = inSignal[i]

    ans = np.zeros(inSignal.size + (kernel1.size - 1))

    for i in range(ans.size):
        for j in range(kernel1.size):
            ans[i] += isSignalZeros[i + j] * kernel1[kernel1.size - 1 - j]

    return ans.astype(int)


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    1:return: The convolved image --> cv2.filter2D with option ’borderType’=cv2.BORDER REPLICATE
    """
    # normalize img
    inImage = normalizeImg(inImage)

    kernel = np.flip(kernel2)
    padImg = cv2.copyMakeBorder(inImage, int((kernel.shape[0] / 2)), int((kernel.shape[0] / 2)),
                                int((kernel.shape[1] / 2)), int((kernel.shape[1] / 2)),
                                cv2.BORDER_CONSTANT)

    ans = np.ndarray(inImage.shape)
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            ans[i, j] = np.multiply(padImg[i:i + kernel.shape[0], j:j + kernel.shape[1]], kernel).sum()

    return ans


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale image
    :return: (directions, magnitude,x_der,y_der)
    """
    # normalize img
    inImage = normalizeImg(inImage)

    kernel_x_der = np.array([[0, 0, 0],
                             [1, 0, -1],
                             [0, 0, 0]])
    x_der = conv2D(inImage, kernel_x_der)

    kernel_y_der = kernel_x_der.T
    y_der = conv2D(inImage, kernel_y_der)

    # MagG = ||G|| = (Ix^2 + Iy^2)^(0.5)
    magnitude = np.sqrt(np.power(x_der, 2) + np.power(y_der, 2))

    # DirectionG = tan^(-1) (Iy/ Ix)
    directions = np.arctan2(y_der, x_der)

    return directions, magnitude, x_der, y_der


def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param kernel_size: Kernel size:
    :return: The Blurred image
    """
    pass


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    if kernel_size % 2 == 0:
        print("Tis kernel size is not an odd number\nTry again")

    # Creat a Gaussian Kernel
    kernel = cv2.getGaussianKernel(int(kernel_size), -1)
    kernel = kernel @ kernel.T

    # Make a convolution between the original img to theGaussian Kernel
    blurred_image = cv2.filter2D(in_image, -1, kernel)

    return blurred_image


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    # normalize img
    img = normalizeImg(img)

    # _____________________
    # CV implementation:
    # ---------------------
    cv_sobel_x = cv2.Sobel(img, -1, 1, 0)
    cv_sobel_y = cv2.Sobel(img, -1, 0, 1)
    cv_sobel_magnitude = np.sqrt(cv_sobel_x ** 2 + cv_sobel_y ** 2)
    opencv_solution = np.zeros(cv_sobel_magnitude.shape)
    opencv_solution[cv_sobel_magnitude >= thresh] = 1

    # _____________________
    # my implementation:
    # ---------------------
    my_sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    my_sobel_y = my_sobel_x.T

    Gx = conv2D(inImage=img, kernel2=my_sobel_x)
    Gy = conv2D(inImage=img, kernel2=my_sobel_y)

    my_sobel_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    my_solution = np.zeros(my_sobel_magnitude.shape)
    my_solution[my_sobel_magnitude >= thresh] = 1

    return opencv_solution, my_solution


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    crossing_zero_img = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try:
                # to check if there is "zero cross"
                # we'll have to count the positive and the negative value as well
                positive_count = 0
                negative_count = 0

                neighbour_pixels = [img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1], img[i, j - 1],
                                    img[i, j + 1], img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1]]

                max_pixel_value = max(neighbour_pixels)
                min_pixel_value = min(neighbour_pixels)

                for pixel in neighbour_pixels:
                    if pixel > 0:
                        positive_count += 1

                    elif pixel < 0:
                        negative_count += 1

                if (negative_count > 0) and (positive_count > 0):
                    # there is "zero cross"
                    if img[i, j] > 0:
                        crossing_zero_img[i, j] = img[i, j] + np.abs(min_pixel_value)
                    elif img[i, j] < 0:
                        crossing_zero_img[i, j] = np.abs(img[i, j]) + max_pixel_value

            except IndexError as e:
                pass

    return crossing_zero_img


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Apply Laplacian operator in some higher datatype
    LOG = cv2.Laplacian(blur, cv2.CV_64F)

    # But this tends to localize the edge towards the brighter side.
    LOG /= LOG.max()

    return edgeDetectionZeroCrossingSimple(LOG)


def NMS(img, angle):
    """
    Performing non-maximum suppression.
    :param img:
    :param angle:
    :return:
    """
    nms = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try:
                # For each pixel (x,y) compare to pixels along its gradient direction
                d1, d2 = 255, 255

                # q1: 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    d1 = img[i, j + 1]
                    d2 = img[i, j - 1]

                # q2: 45
                elif 22.5 <= angle[i, j] < 67.5:
                    d1 = img[i - 1, j - 1]
                    d2 = img[i + 1, j + 1]

                # q3: 90
                elif 67.5 <= angle[i, j] < 112.5:
                    d1 = img[i + 1, j]
                    d2 = img[i - 1, j]

                # q4: 135
                elif 112.5 <= angle[i, j] < 157.5:
                    d1 = img[i + 1, j - 1]
                    d2 = img[i - 1, j + 1]

                # check If |G(x,y)| is a local maximum
                if (img[i, j] >= d1) and (img[i, j] >= d2):
                    nms[i, j] = img[i, j]
                else:
                    nms[i, j] = 0

            except IndexError as e:
                pass

    return nms


def hysteresis(img, thrs_1, thrs_2):
    """
    Performing Hysteresis and finding false edges
    :param img:
    :param thrs_2:
    :param thrs_1:
    :return:
    # """
    strong_edge = 255
    week_edge = 100

    # Save the coordination who classify our img to 3 parts
    zeros_rows, zeros_cols = np.where(img < thrs_2)
    strong_rows, strong_cols = np.where(img >= thrs_1)
    weak_rows, weak_cols = np.where((img <= thrs_1) & (img >= thrs_2))

    # Apply those coordination to the img arr and update each pixel value
    img = np.zeros(img.shape)
    img[zeros_rows, zeros_cols] = 0
    img[strong_rows, strong_cols] = strong_edge
    img[weak_rows, weak_cols] = week_edge

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try:
                if img[i][j] == week_edge:
                    neighbor_matrix = img[i - 1: i + 2, j - 1: j + 2]
                    row_ni, _, = np.where(neighbor_matrix == strong_edge)
                    if len(row_ni) > 0:
                        # connected to an edge pixel
                        img[i, j] = strong_edge
                    else:
                        img[i, j] = 0

            except IndexError as e:
                pass

    return img


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges using "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    # normalize img
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # _____________________
    # CV implementation:
    # ---------------------
    cv_canny = cv2.Canny(img.astype(np.uint8), thrs_1, thrs_2)

    # _____________________
    # my implementation:
    # ---------------------
    # 1. Smooth the image with a Gaussian
    img = cv2.GaussianBlur(img, (5, 5), 1)

    # 2. Compute the partial derivatives Ix, Iy
    Ix = cv2.Sobel(img, -1, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, -1, 0, 1, ksize=3)

    # 3. Compute magnitude and direction of the gradient
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)  # Mag(G)
    theta = np.arctan2(Iy, Ix)  # direction

    # 4. Quantize the gradient directions
    theta_q = np.degrees(theta)
    theta_q[theta_q < 0] += 180

    # 5. Perform non-maximum suppression
    _nms = NMS(magnitude, theta_q)

    # 6. Hysteresis
    my_canny = hysteresis(_nms, thrs_1, thrs_2)

    return cv_canny, my_canny


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """
    # normalize img
    img = normalizeImg(img)

    radius_by_shape = min(img.shape[0], img.shape[1]) // 2
    max_radius = min(radius_by_shape, int(max_radius))

    # Get each pixels gradients direction
    Iy = cv2.Sobel(img, -1, 0, 1, ksize=3)
    Ix = cv2.Sobel(img, -1, 1, 0, ksize=3)
    angle = np.arctan2(Iy, Ix)
    angle = np.radians(angle * 180 / np.pi)

    # We will create an H matrix representing the Hough space
    radius_diff = max_radius - min_radius
    circle_hough = np.zeros((img.shape[0], img.shape[1], radius_diff + 1))

    # Get Edges
    canny_edges, _, = edgeDetectionCanny((img * 255).astype(np.uint8), 550, 100)

    # List of circles
    detected_circles = list()

    for x in range(canny_edges.shape[0]):
        for y in range(canny_edges.shape[1]):
            if canny_edges[x, y] == 255:
                for radius in range(int(radius_diff)):
                    try:
                        alpha = angle[x, y] - np.pi / 2

                        a1, a2 = int(x - radius * np.cos(alpha)), int(x + radius * np.cos(alpha))
                        b1, b2 = int(y + radius * np.sin(alpha)), int(y - radius * np.sin(alpha))

                        x_dimension = len(circle_hough)
                        y_dimension = len(circle_hough)

                        if 0 < a1 < x_dimension and 0 < b1 < y_dimension:
                            circle_hough[a1, b1, radius] += 1
                        if 0 < a2 < x_dimension and 0 < b2 < y_dimension:
                            circle_hough[a2, b2, radius] += 1

                    except IndexError as e:
                        pass

    thresh = np.max(circle_hough) / 2

    # Get the coordinates
    x, y, radius = np.where(circle_hough >= thresh)
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 0 and radius[i] == 0:
            continue
        detected_circles.append((y[i], x[i], radius[i]))

    return detected_circles

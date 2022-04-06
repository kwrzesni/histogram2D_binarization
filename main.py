import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram2D(im):
    kernel = np.ones([3, 3])/9
    g = cv2.filter2D(im, -1, kernel, borderType=cv2.BORDER_CONSTANT)[1:-1, 1:-1]
    f = im[1:-1, 1:-1]
    hist2D = np.zeros((256, 256))
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            hist2D[g[y, x], f[y, x]] += 1
    return hist2D


def calculate_entropy(hist2D):
    if not hist2D.shape[0] or not hist2D.shape[1]:
        return -float('inf')
    hist2D_sum = np.sum(hist2D)
    if hist2D_sum == 0:
        return -float('inf')
    p = hist2D/hist2D_sum
    #replace 0 in p to avoid divide by 0 in log calculation
    p[p == 0] = 1
    p_logs = np.log(p)
    return -np.sum(p*p_logs)


def calculate_cost(hist2D, point):
    s, t = point
    return calculate_entropy(hist2D[:s, :t]) + calculate_entropy(hist2D[s:, t:])


def brute_force(im):
    hist2D = calculate_histogram2D(im)
    cost = np.zeros((256, 256))
    for s in range(256):
        for t in range(256):
            cost[s, t] = calculate_cost(hist2D, (s, t))
    result = np.argmax(cost)
    print(result)
    s, t = result % 256, result // 256
    return t


def main():
    im = cv2.cvtColor(cv2.imread('rice.jpeg'), cv2.COLOR_BGR2GRAY)
    t = brute_force(im)
    _, im_bin = cv2.threshold(im, t, 255, cv2.THRESH_BINARY)
    cv2.imshow('result', im_bin)
    cv2.waitKey()


if __name__ == '__main__':
    main()


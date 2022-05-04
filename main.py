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


def calculate_cost_vec(hist2D, vec):
    n = vec.shape[1]
    f_vec = np.zeros(n)
    for i in range(n):
        f_vec[i] = calculate_cost(hist2D, vec[:, i])
    return f_vec


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


def pso(im, N_particles, N_iter):
    #params
    c1 = 1.3
    c2 = 1.3
    w = 0.8
    v_max = 2.5
    hist2D = calculate_histogram2D(im)
    params = (c1, c2, w, v_max, hist2D)
    X = np.random.randint(0, 255, (2, N_particles))
    V = np.random.randn(2, N_particles)*0.2

    # setup + 1st iter
    pb = X
    f_pb = calculate_cost_vec(hist2D, pb)
    gb = pb[:, f_pb.argmax()]
    f_gb = f_pb.max()

    r = np.random.rand(2)
    V = w * V + c1 * r[0] * (pb - X) + c2 * r[1] * (gb.reshape(-1,1) - X)
    V[(V > v_max)] = v_max
    X = X + V.astype('int32')
    #border condiction (space loop - naive)
    for idx in range(X.shape[0]):
        X[idx, (X[idx, :] >= 256)] -= 256
        X[idx, (X[idx, :] < 0)] += 256

    def iteration(V, X, pb, f_pb, gb, f_gb, params):
        (c1, c2, w, v_max, hist2D) = params
        f = calculate_cost_vec(hist2D, X)

        pb[:, (f > f_pb)] = X[:, (f > f_pb)]
        f_pb = calculate_cost_vec(hist2D, pb)
        gb = pb[:, f_pb.argmax()]
        f_gb = f_pb.max()

        r = np.random.rand(2)
        V = w * V + c1 * r[0] * (pb - X) + c2 * r[1] * (gb.reshape(-1, 1) - X)
        V[(V > v_max)] = v_max
        X = X + V.astype('int32')
        for idx in range(X.shape[0]):
            X[idx, (X[idx, :] >= 256)] -= 256
            X[idx, (X[idx, :] < 0)] += 256

    for i in range(N_iter):
        iteration(V, X, pb, f_pb, gb, f_gb, params)
    s, _ = gb

    return s


def pso_with_neighborhood(im, N_particles, N_iter_neigh, N_iter_glob, R):
    #params
    c1 = 1.3
    c2 = 1.3
    w = 0.8
    v_max = 2.5
    hist2D = calculate_histogram2D(im)
    params = (c1, c2, w, v_max, hist2D, R)
    X = np.random.randint(0, 255, (2, N_particles))
    V = np.random.randn(2, N_particles)*0.2

    # setup + 1st iter
    pb = X
    f_pb = calculate_cost_vec(hist2D, pb)
    gb = pb[:, f_pb.argmax()]
    f_gb = f_pb.max()

    r = np.random.rand(2)
    V = w * V + c1 * r[0] * (pb - X) + c2 * r[1] * (gb.reshape(-1, 1) - X)
    V[(V > v_max)] = v_max
    X = X + V.astype('int32')
    #border condiction (space loop - naive)
    for idx in range(X.shape[0]):
        X[idx, (X[idx, :] >= 256)] -= 256
        X[idx, (X[idx, :] < 0)] += 256

    def iteration(V, X, pb, f_pb, gb, f_gb, params, type):
        c1, c2, w, v_max, hist2D, R = params
        f = calculate_cost_vec(hist2D, X)

        pb[:, (f > f_pb)] = X[:, (f > f_pb)]
        f_pb = calculate_cost_vec(hist2D, pb)
        gb = pb[:, f_pb.argmax()]
        f_gb = f_pb.max()

        r = np.random.rand(2)
        if type == 'glob':
            V = w * V + c1 * r[0] * (pb - X) + c2 * r[1] * (gb.reshape(-1, 1) - X)
        elif type == 'neigh':
            for i in range(X.shape[1]):
                x = X[:, i]
                best_pos = x
                best = -float('inf')
                for j in range(X.shape[1]):
                    y = X[:, j]
                    if (x[0] - y[0])**2 + (x[1] - y[1])**2 <= R*R:
                        if f_pb[j] > best:
                            best = f_pb[j]
                            best_pos = y
                best_pos = np.array(best_pos)
                V[:, i] = w * V[:, i] + c1 * r[0] * (pb[:, i] - X[:, i]) + c2 * r[1] * (best_pos - X[:, i])
        V[(V > v_max)] = v_max
        X = X + V.astype('int32')
        for idx in range(X.shape[0]):
            X[idx, (X[idx, :] >= 256)] -= 256
            X[idx, (X[idx, :] < 0)] += 256

    for i in range(N_iter_neigh):
        iteration(V, X, pb, f_pb, gb, f_gb, params, 'neigh')
    for i in range(N_iter_glob):
        iteration(V, X, pb, f_pb, gb, f_gb, params, 'glob')
    s, _ = gb

    return s


def main(flag = 0):
    #np.random.seed(seed=49)
    im = cv2.cvtColor(cv2.imread('rice.png'), cv2.COLOR_BGR2GRAY)
    if flag:
        s = brute_force(im)
    else:
        #s = pso(im, N_particles=20, N_iter=70)
        s = pso_with_neighborhood(im, N_particles=100, N_iter_neigh=30, N_iter_glob=20, R=10)
    _, im_bin = cv2.threshold(im, s, 255, cv2.THRESH_BINARY)
    cv2.imshow('result', im_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


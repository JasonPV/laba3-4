import cv2
import numpy as np
import random
from sklearn.datasets import fetch_olivetti_faces
from scipy.fftpack import dct
from sklearn.metrics import accuracy_score


def get_data():
    data_images = fetch_olivetti_faces()
    return data_images['images'], data_images['target']


def get_histogram(image, param=30):
    hist, bins = np.histogram(image, bins=np.linspace(0, 1, param))
    return [hist, bins]


def get_dft(image, mat_side=13):
    f = np.fft.fft2(image)
    f = f[0:mat_side, 0:mat_side]
    return np.abs(f)


def get_dct(image, mat_side=13):
    c = dct(image, axis=1)
    c = dct(c, axis=0)
    c = c[0:mat_side, 0:mat_side]
    return c


def get_gradient(image, n=2):
    shape = image.shape[0]
    i, l = 0, 0
    r = n
    result = []

    while r <= shape:
        window = image[l:r, :]
        result.append(np.sum(window))
        i += 1
        l = i * n
        r = (i + 1) * n
    result = np.array(result)
    return result


def get_scale(image, scale=0.35):
    h = image.shape[0]
    w = image.shape[1]
    new_size = (int(h * scale), int(w * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def get_feature(data, method, parameter):
    result = []
    for element in data:
        if method == get_histogram:
            result.append(method(element, parameter)[0])
        else:
            result.append(method(element, parameter))
    return result


def distance(el1, el2):
    return np.linalg.norm(np.array(el1) - np.array(el2))



def classifier(train, data, method, param):
    assert method in [get_scale, get_histogram, get_dct, get_dft, get_gradient]
    feature_samples = get_feature(train[0], method, param)
    feature_data = get_feature(data[0], method, param)
    result = []
    for element in feature_data:
        mn = distance(element, feature_samples[0])
        cl = 0
        for i in range(1, len(feature_samples)):
            dist = distance(element, feature_samples[i])
            if dist < mn:
                mn = dist
                cl = i
        # print(mn)
        result.append(train[1][cl])
    # accuracy = accuracy_score(result, data[1])
    return result

def closest(train, test, m, p):
    feature_samples = get_feature(train[0], m, p)
    featrue_test = m(test, p)
    if m == get_histogram:
        featrue_test = featrue_test[0]
    buf = 0
    mn = distance(featrue_test, feature_samples[0])
    for i in range(1, len(feature_samples)):
        dist = distance(featrue_test, feature_samples[i])
        if dist < mn:
            mn = dist
            buf = i

    return buf



def get_best_params(train, test, method):
    assert method in [get_scale, get_histogram, get_dct, get_dft, get_gradient]
    if method == get_scale:
        params = np.arange(0.1, 1, 0.05)
    elif method == get_histogram:
        params = np.arange(13, 300, 1)
    elif method == get_gradient:
        params = np.arange(3, int(test[0][0].shape[0]/2 - 1), 1)
    else:
        params = np.arange(3, min(test[0][0].shape), 1)

    best_param = params[0]
    results = classifier(train, test, method, best_param)
    a = accuracy_score(results, test[1])
    for p in params:
        r = classifier(train, test, method, p)
        a_b = accuracy_score(r, test[1])
        if a_b >= a: #>=
            a = a_b
            best_param = p
    return best_param, a


def get_split_data(data, num=2):
    # random.seed(41)
    ran = [int(i) for i in range(10)]
    # random.shuffle(ran)
    ran = ran[:num]
    X = data[0]
    y = data[1]
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(0, 10):
        if i not in ran:
            for j in range(40):
                X_train.append(X[i+10*j])
                y_train.append(y[i+10*j])
    for i in range(0, 10):
        if i in ran:
            for j in range(40):
                X_test.append(X[i+10*j])
                y_test.append(y[i+10*j])

    data_train = [X_train, y_train]
    data_test = [X_test, y_test]
    # print(len(data_train[0]))
    # print(len(data_test[0]))
    return data_train, data_test


def get_samples(data):
    a = random.randint(0, 9)
    # a = 8
    arr1 = []
    arr2 = []
    for i in range(0, len(data[0]), 10):
        arr1.append(data[0][i+a])
        arr2.append(data[1][i+a])
    samples = np.array([arr1, arr2], dtype=np.ndarray)

    return samples, a


def get_cross(data, method):
    result = []
    params = []
    for i in range(9, 0, -1):
        data_train, data_test = get_split_data(data, i)
        p, a = get_best_params(data_train, data_test, method)
        params.append(p)
        result.append(a)
    return result, params

def get_vote(train, test):
    methods = ['get_histogram', 'get_dft', 'get_dct', 'get_gradient', 'get_scale']
    results = []
    params = []
    acs = []
    for m in methods:
        p, a = get_best_params(train, test, eval(m))
        results.append(classifier(train, test, eval(m), p))
        params.append(p)
        acs.append(a)
    results = np.array(results)
    res = results.transpose()
    res = res.tolist()
    res = list(map(lambda x: max(x, key=x.count), res))
    return res, acs, params

def get_vote_1(train, test, p):
    methods = ['get_histogram', 'get_dft', 'get_dct', 'get_gradient', 'get_scale']
    results = []
    k = 0
    for m in methods:
        if m == 'get_scale':
            par = float(p[k].text())
        else:
            par = int(p[k].text())
        results.append(classifier(train, test, eval(m), par))
        k+=1
    results = np.array(results)
    res = results.transpose()
    res = res.tolist()
    res = list(map(lambda x: max(x, key=x.count), res))
    return res


if __name__ == '__main__':
    data = get_data()
    # samples = get_samples(data)
    data_train, data_test = get_split_data(data, 0.2)

    best_p, accuracy = get_best_params(data_test, data_train, get_scale)
    print(best_p, accuracy)
    res, ac = classifier(data_test, data_train, get_scale, best_p)
    print(res)
    print(ac)
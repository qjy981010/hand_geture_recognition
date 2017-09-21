import random
import cv2
import math
import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName, labelnum):
    dataMat = []
    startlist = []
    endlist = []
    for i in range(labelnum):
        with open(fileName + str(i) + '.txt') as fp:
            lines = fp.readlines()
            if i:
                startlist.append(endlist[i-1])
            else:
                startlist.append(0)
            endlist.append(startlist[i] + len(lines))
            for line in lines:
                lineArr = line.strip().split()
                dataMat.append([float(x) for x in lineArr])
    return dataMat, startlist, endlist


def oldloadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(x) for x in lineArr[:-1]])
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i  # we want to select any J not equal to i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  # linear kernel
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # divide in NumPy is element-wise not matrix like Matlab
        K = np.exp(K / (-1 * kTup[1]**2))
    else:
        raise NameError(
            'Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct:

    def __init__(self, dataMat, labelMat, C, toler, kTup):
        self.X = dataMat
        self.labelMat = labelMat
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMat)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    fXk = np.float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k]) + oS.b
    Ek = fXk - np.float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    maxk = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxk = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxk, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) \
            or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * \
            (alphaJold - oS.alphas[j])  # update i by the same amount as j
        # added this for the Ecache                    #the update is in the
        # oppostie direction
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[
            i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[
            i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(
        classLabels).transpose(), C, toler, kTup)
    it = 0
    entireSet = True
    alphaPairsChanged = 0
    while (it < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            it += 1
        else:  # go over non-round (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
            it += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        # print("iteration number: %d" % it)
    return oS.b, oS.alphas


def calcWs(alphas, dataMat, labelMat):
    X = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def nothing():
    pass


def train(li, i, startlist, endlist, data, ks, errors):
    datMat = np.mat(data)
    labels = [-1] * startlist[i] + [1] * (endlist[i] - startlist[i]) + \
        [-1] * (endlist[-1] - endlist[i])
    b, alphas = smoP(data, labels, 200, 0.0001, 10000000, ('rbf', ks[i]))
    labelMat = np.mat(labels).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    judge = (datMat[svInd], labelMat[svInd], alphas[svInd], b ,i)
    errorcount = 0
    for j in range(datMat.shape[0]):
        kernelEval = kernelTrans(judge[0], datMat[j, :], ('rbf', ks[i]))
        predict = (kernelEval.T *
                   np.multiply(judge[1], judge[2]) + judge[3])[0][0]
        if np.sign(predict) != np.sign(labels[j]):
            errorcount += 1
    li.append(judge)
    errors.append((errorcount / datMat.shape[0], i, ks[i]))
    return judge


def mainTrain(resultfile, labelnum, data, startlist, endlist, ks):
    judge_feats = []
    manager = multiprocessing.Manager()
    li = manager.list()
    errors = manager.list()
    pool = multiprocessing.Pool(processes=6)
    # train(li, 4, startlist, endlist, data, ks, errors)
    # li.pop()
    # errors.pop()
    for i in range(labelnum):
        pool.apply_async(train, args=(li, i, startlist, endlist, data, ks, errors))
    pool.close()
    pool.join()
    judge_feats = li
    np.save(resultfile, judge_feats)
    print(errors)
    with open('error.txt', 'a') as f:
        f.write(str(ks) + ':\n' + str(errors) + '\n\n')
    return judge_feats


def get_video_guass_hsv(judge_feats, labelnum, startlist, endlist, ks):
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() is False):
        print("Error opening video stream or file")
        return
    count = [endlist[i] - startlist[i] for i in range(labelnum)]
    cv2.namedWindow('image')
    # create trackbars for color change
    cv2.createTrackbar('Hs', 'image', 0, 179, nothing)
    cv2.createTrackbar('Ss', 'image', 0, 255, nothing)
    cv2.createTrackbar('Vs', 'image', 0, 255, nothing)
    cv2.createTrackbar('He', 'image', 0, 179, nothing)
    cv2.createTrackbar('Se', 'image', 0, 255, nothing)
    cv2.createTrackbar('Ve', 'image', 0, 255, nothing)
    while(cap.isOpened()):
        ret, frame = cap.read()
        # Capture frame-by-frame
        if ret is True:
            cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 0)
            crop_img = frame[100:300, 100:300]
            blurred = cv2.GaussianBlur(crop_img, (17, 17), 0)
            # cv2.imshow('blurred', blurred)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
            lower_skin = np.array([90, 40, 50])
            upper_skin = np.array([125, 130, 255])
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            image, contours, hierarchy = cv2.findContours(mask.copy(),
                                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours):
                cnt = max(contours, key=lambda x: cv2.contourArea(x))
                hull = cv2.convexHull(cnt)
                drawing = np.zeros(crop_img.shape, np.uint8)
                epsilon = 0.03*cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
                cv2.drawContours(drawing, [approx], 0, (0, 255, 0), 0)
                hu = cv2.HuMoments(cv2.moments(cnt))[:2]
                hu1 = round(hu[0][0] * 10, 2)
                hu2 = round(hu[1][0] * 1000, 2)
                L = cv2.arcLength(cnt, True)
                S = cv2.contourArea(cnt)
                LdivideS = round(S / L, 2)
                Lhull = cv2.arcLength(hull, True)
                leftmost = cnt[cnt[:, :, 0].argmin()][0][0]
                rightmost = cnt[cnt[:, :, 0].argmax()][0][0]
                BdivideLh = round((L - Lhull) / (rightmost - leftmost), 2)
                hull = cv2.convexHull(cnt, returnPoints=False)
                defects = cv2.convexityDefects(cnt, hull)
                count_concave = 0
                count_convex = 0
                longest = 0
                shortest = 1000000
                lastfinger = [0, 0]
                longestdist = 0
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        a = (end[0] - start[0])**2 + (end[1] - start[1])**2
                        b = (far[0] - start[0])**2 + (far[1] - start[1])**2
                        c = (end[0] - far[0])**2 + (end[1] - far[1])**2
                        if a < b + c:
                            longest = max(longest, b, c)
                            shortest = min(shortest, b, c)
                            count_concave += 1
                for i in range(0, len(approx)):
                    pre = i - 1
                    nex = i + 1
                    if i == 0:
                        pre = len(approx) - 1
                    elif i == len(approx) - 1:
                        nex = 0
                    if approx[i][0][1] < approx[pre][0][1] and approx[i][0][1] < approx[nex][0][1]:
                        a = math.sqrt((approx[pre][0][0] - approx[nex][0][0])**2 + (approx[pre][0][1] - approx[nex][0][1])**2)
                        b = math.sqrt((approx[i][0][0] - approx[nex][0][0])**2 + (approx[i][0][1] - approx[nex][0][1])**2)
                        c = math.sqrt((approx[pre][0][0] - approx[i][0][0])**2 + (approx[pre][0][1] - approx[i][0][1])**2)
                        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                        if angle < 50:
                            count_convex += 1
                            if lastfinger[0]:
                                longestdist = max(longestdist, (approx[i][0][
                                                  0] - lastfinger[0])**2 + (approx[i][0][1] - lastfinger[1])**2)
                            lastfinger = approx[i][0]
                # print(hu1, hu2, count_concave, count_convex, longest / 1000, shortest / 1000, math.sqrt(longestdist) / 10)
                features = (hu1, hu2, count_concave, count_convex, longest /
                            1000, shortest / 1000, math.sqrt(longestdist) / 10)
                cv2.imshow('drawing', drawing)

                data = np.mat([list(features)])[0, :]
                maxlabel = 0
                maxpred = -0x7fff
                for i in range(labelnum):
                    judge = judge_feats[i]
                    # if judge[4] == 9:
                    #     continue
                    kernelEval = kernelTrans(judge[0], data, ('rbf', ks[judge[4]]))
                    predict = (kernelEval.T *
                               np.multiply(judge[1], judge[2]) + judge[3])[0][0]
                    if maxpred < predict:
                        maxpred = predict
                        maxlabel = judge[4]

                cv2.putText(frame, str(maxlabel), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            cv2.imshow('frame', frame)
            cv2.imshow('mask', mask)
            k = cv2.waitKey(10)
        if k == 27:
            break
        elif 47 < k and k < 58:
            index = k - 48
            print(index, count[index])
            with open('/home/qjy/ai/task/hand_geture_recognition/data/' + str(index) + '.txt', 'a') as fp:
                for i in range(7):
                    fp.write(str(features[i]))
                    if i == 6:
                        fp.write('\n')
                    else:
                        fp.write(' ')
            cv2.imwrite('/home/qjy/ai/task/hand_geture_recognition/data/img/' + str(index) + '/' +
                        str(count[index]) + '.jpg', mask)
            count[index] += 1
        elif k == 106:
            data = np.mat([list(features)])[0, :]
            maxlabel = 0
            maxpred = -0x7fff
            for i in range(labelnum):
                judge = judge_feats[i]
                # if judge[4] == 9:
                #     continue
                kernelEval = kernelTrans(judge[0], data, ('rbf', ks[judge[4]]))
                predict = (kernelEval.T *
                           np.multiply(judge[1], judge[2]) + judge[3])[0][0]
                print(judge[4], predict)
                if maxpred < predict:
                    maxpred = predict
                    maxlabel = judge[4]
            print(maxlabel)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    resultfile = 'result.npy'
    labelnum = 10
    ks = [9.3, 11.1, 12.9, 14.4, 13.5, 14.4, 11.100000000000005, 9.3, 11.0, 14.4]
    data, startlist, endlist = loadDataSet('/home/qjy/ai/task/hand_geture_recognition/data/', labelnum)
    # for k1 in np.arange(9, 15, 0.3):
    judge_feats = mainTrain(resultfile, labelnum, data, startlist, endlist, ks)
    # judge_feats = np.load(resultfile)
    print(len(judge_feats))
    get_video_guass_hsv(judge_feats, labelnum, startlist, endlist, ks)
    # time.sleep(30)

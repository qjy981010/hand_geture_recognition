import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_video():
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() is False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        # Capture frame-by-frame
        if ret is True:
            cv2.imshow('Frame', frame)
            k = cv2.waitKey(10)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def get_video_gray():
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() is False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        # Capture frame-by-frame
        if ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = np.sqrt(gray / float(np.max(gray)))
            cv2.imshow('Frame', gray)
            k = cv2.waitKey(10)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def getskin():
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() is False):
        print("Error opening video stream or file")
        return
    ret, frame = cap.read()
    if not ret:
        print("Error reading video stream or file")
        return
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    # 返回行数，列数，通道个数
    shape = ycbcr.shape

    Kl, Kh = 125, 188
    Ymin, Ymax = 16, 235
    Wlcb, Wlcr = 23, 20
    Whcb, Whcr = 14, 10
    Wcb, Wcr = 46.97, 38.76
    # 椭圆模型参数
    Cx, Cy = 109.38, 152.02
    ecx, ecy = 1.60, 2.41
    a, b = 25.39, 14.03
    Theta = 2.53 / np.pi * 180
    # 每行
    for row in range(shape[0]):
        # 每列
        for col in range(shape[1]):
            Y = ycbcr[row, col, 0]
            CbY = ycbcr[row, col, 1]
            CrY = ycbcr[row, col, 2]
            if Y < Kl or Y > Kh:
                # 求Cb, Cr的均值
                if Y < Kl:
                    # 公式(7)
                    CbY_aver = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)
                    # 公式(8)
                    CrY_aver = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
                    # 公式(6)
                    WcbY = Wlcb + (Y - Ymin) * (Wcb - Wlcb) / (Kl - Ymin)
                    WcrY = Wlcr + (Y - Ymin) * (Wcr - Wlcr) / (Kl - Ymin)
                elif Y > Kh:
                    # 公式(7)
                    CbY_aver = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh)
                    # 公式(8)
                    CrY_aver = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
                    # 公式(6)
                    WcbY = Whcb + (Ymax - Y) * (Wcb - Whcb) / (Ymax - Kh)
                    WcrY = Whcr + (Ymax - Y) * (Wcr - Whcr) / (Ymax - Kh)
                # 求Cb(Kh), Cr(Kh)的均值
                CbKh_aver = 108 + (Kh - Kh) * (118 - 108) / (Ymax - Kh)
                CrKh_aver = 154 + (Kh - Kh) * (154 - 132) / (Ymax - Kh)
                # 公式(5)
                Cb = (CbY - CbY_aver) * Wcb / WcbY + CbKh_aver
                Cr = (CrY - CrY_aver) * Wcr / WcrY + CrKh_aver
            else:
                # 公式(5)
                Cb = CbY
                Cr = CrY
            # Cb，Cr代入椭圆模型
            cosTheta = np.cos(Theta)
            sinTehta = np.sin(Theta)
            matrixA = np.array(
                [[cosTheta, sinTehta], [-sinTehta, cosTheta]], dtype=np.double)
            matrixB = np.array([[Cb - Cx], [Cr - Cy]], dtype=np.double)
            # 矩阵相乘
            matrixC = np.dot(matrixA, matrixB)
            x = matrixC[0, 0]
            y = matrixC[1, 0]
            ellipse = (x - ecx) ** 2 / a ** 2 + (y - ecy) ** 2 / b ** 2
            if ellipse <= 1:
                # 白
                ycbcr[row, col] = [255, 255, 255]
                # 黑
            else:
                ycbcr[row, col] = [0, 0, 0]
    # 绘图
    plt.subplot(111)
    plt.imshow(ycbcr)
    plt.title('New')
    plt.show()


def get_video_hcv():
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() is False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        # Capture frame-by-frame
        if ret is True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            lower_skin = np.array([100, 50, 0])
            upper_skin = np.array([125, 255, 255])

            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            cv2.imshow('Frame', mask)
            k = cv2.waitKey(10)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def get_video_blue_hcv():
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() is False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        # Capture frame-by-frame
        if ret is True:
            # 将图片从 BGR 空间转换到 HSV 空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 定义在HSV空间中蓝色的范围
            lower_blue = np.array([110, 50, 50])
            upper_blue = np.array([130, 255, 255])

            # 根据以上定义的蓝色的阈值得到蓝色的部分
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            res = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow('Frame', mask)
            # cv2.imshow('Frame', res)
            k = cv2.waitKey(10)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def get_video_guass_hsv():
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() is False):
        print("Error opening video stream or file")
    count = 100                        #############################################
    fp = open('gesture_data.txt', 'a')            #############################################
    while(cap.isOpened()):
        ret, frame = cap.read()
        # Capture frame-by-frame
        if ret is True:
            cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 0)
            crop_img = frame[100:300, 100:300]
            blurred = cv2.GaussianBlur(crop_img, (17, 17), 0)
            cv2.imshow('blurred', blurred)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

            lower_skin = np.array([90, 40, 50])
            upper_skin = np.array([125, 130, 255])
            # lower_skin = np.array([100, 50, 0])
            # upper_skin = np.array([125, 255, 255])

            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            hu = cv2.HuMoments(cv2.moments(mask))[:2]
            cv2.putText(frame, str(hu), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.imshow('frame', frame)
            cv2.imshow('Frame', mask)
            k = cv2.waitKey(10)
        if k == 27:
            break
        if k == ord(' '):
            print(count)
            print(hu)

            for i in range(7):
                fp.write(str(hu[i][0]))
                if i == 6:
                    fp.write(' 1\n') #############################################
                else:
                    fp.write(' ')
            cv2.imwrite('img/' + str(count) + '.jpg', mask) #############################################
            count += 1
    cap.release()
    cv2.destroyAllWindows()


def main():
    # get_video_gray()
    # getskin()
    # get_video_hcv()
    # get_video_blue_hcv()
    get_video_guass_hsv()
    # get_video()


if __name__ == '__main__':
    main()

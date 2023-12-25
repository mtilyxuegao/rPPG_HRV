import cv2
import dlib
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import sys
import os
from scipy import signal
from scipy.fftpack import fft, ifft
import matplotlib.mlab as mlab
from sklearn import preprocessing
from sklearn import impute
import pandas as pd

predictorPath = r"../shape_predictor_68_face_landmarks.dat"
predictorIdx = [[1, 2, 3, 4, 31, 36, 48], [12, 13, 14, 15, 35, 45, 54]]
medIdx = [27, 28, 29, 30]


def rect_to_bb(rect):
    """ Transform a rectangle into a bounding box
    Args:
        rect: an instance of dlib.rectangle
    Returns:
        [x, y, w, h]: coordinates of the upper-left corner
            and the width and height of the box
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return [x, y, w, h]


def shape_to_np(shape, dtype="int"):
    """ Transform the detection results into points
    Args:
        shape: an instance of dlib.full_object_detection
    Returns:
        coords: an array of point coordinates
            columns - x; y
    """
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def np_to_bb(coords, ratio=4, dtype="int"):
    """ Chooose ROI based on points and ratio
    Args:
        coords: an array of point coordinates
            columns - x; y
        ratio: the ratio of the length of the bounding box in each direction
            to the distance between ROI and the bounding box
        dtype: optional variable, type of the coordinates
    Returns:
        coordinates of the upper-left and bottom-right corner
    """
    roi = cv2.fitEllipse(coords)
    m_roi = list(map(int, [roi[0][0], roi[0][1], roi[1][0] / (ratio + 1), roi[1][1] / (ratio + 1), roi[2]]))
    return m_roi


def resize(image, width=1200):
    """ Resize the image with width
    Args:
        image: an instance of numpy.ndarray, the image
        width: the width of the resized image
    Returns:
        resized: the resized image
        size: size of the resized image
    """
    r = width * 1.0 / image.shape[1]
    size = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized, size


def coordTrans(imShape, oriSize, rect):
    """Transform the coordinates into the original image
    Args:
        imShape: shape of the detected image
        oriSize: size of the original image
        rect: an instance of dlib.rectangle, the face region
    Returns:
        the rect in the original image
    """

    left = int(rect.left() / oriSize[0] * imShape[1])
    right = int(rect.right() / oriSize[0] * imShape[1])
    top = int(rect.top() / oriSize[1] * imShape[0])
    bottom = int(rect.bottom() / oriSize[1] * imShape[0])

    left = int(round(rect.left() / oriSize[0] * imShape[1]))
    right = int(round(rect.right() / oriSize[0] * imShape[1]))
    top = int(round(rect.top() / oriSize[1] * imShape[0]))
    bottom = int(round(rect.bottom() / oriSize[1] * imShape[0]))

    return dlib.rectangle(left, top, right, bottom)


def dataplot(T, data, fn):
    plt.figure(fn)
    clr = ['b', 'g', 'r']
    for i in range(0, 3):
        plt.subplot(3, 2, 2 * i + 1)
        plt.plot(T, data[i], clr[i])
        plt.grid()
        plt.subplot(3, 2, 2 * i + 2)
        plt.psd(data[i], NFFT=256, Fs=25, window=mlab.window_none,
                scale_by_freq=True)


class Detector:
    """ Detect and calculate ppg signal
    roiRatio: a positive number, the roi gets bigger as it increases
    smoothRatio: a real number between 0 and 1,
         the landmarks get stabler as it increases
    """
    detectSize = 480
    clipSize = 540
    roiRatio = 2
    markSmoothRatio = 0.95
    Smthre = 0.9
    alpha = round(math.log(1 / Smthre) / 20, 4)

    def __init__(self, detectorPath=None, predictorPath=None, predictorIdx=None):
        """ Initialize the instance of Detector

        detector: dlib.fhog_object_detector
        predictor: dlib.shape_predictor
        rect: dlib.rectangle, face region in the last frame
        landmarks: numpy.ndarray, coordinates of face landmarks in the last frame
                columns - x; y

        Args:
            detectorPath: path of the face detector
            predictorPath: path of the shape predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictorPath)
        self.idx = predictorIdx
        self.rect = None
        self.face = None
        self.landmarks = None
        self.rois = None

    def __call__(self, image, t):
        """ Detect the face region and returns the ROI value

        Face detection is the slowest part.

        Args:
            image: an instance of numpy.ndarray, the image
        Return:
            val: an array of ROI value in each color channel
        """
        val = [0, 0, 0]
        # Resize the image to limit the calculation
        resized, detectionSize = resize(image, self.detectSize)
        # Perform face detection on a grayscale image
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # No need for upsample, because its effect is the same as resize
        rects = self.detector(gray, upsample_num_times=0)
        num = len(rects)  # there should be one face
        if num == 0:
            print("Time: ", '{:.3f}'.format(t), " No face in the frame!")
            if isinstance(self.landmarks, type(None)):
                return val
            else:
                landmarks = self.landmarks
                # Perfom landmarks smoothing
                if (self.rect != None):
                    distFM = self.distForMarks(self.landmarks[medIdx], landmarks[medIdx])
                    # print('\ndistFM:'+'{:.2f}'.format(distFM)+'\n')
                    # print(self.distForMarks(self.landmarks[medIdx], landmarks[medIdx]))
                    landmarks = self.smoothMarks(self.landmarks, landmarks, distFM)

                # ROI value
                rois = [np_to_bb(landmarks[idx], self.roiRatio) for idx in self.idx]
                mask = np.zeros([2, image.shape[0], image.shape[1]], dtype=int)
                val = [[], []]
                for i in range(0, len(rois)):
                    # print(i)
                    roi = rois[i]
                    cv2.ellipse(mask[i], (roi[0], roi[1]), (roi[2], roi[3]), roi[4], -180, 180, 255, -1)
                    mmsk = np.mean(image[mask[i] > 0], 0)
                    val[i].append(mmsk)
                height = image.shape[0]
                width = image.shape[1]
                iht = int(height / 4)
                iwt = int(width / 16)
                lbg = np.mean(np.mean(image[iht:height - iht, iwt:iwt * 2], 0), 0)
                rbg = np.mean(np.mean(image[iht:height - iht, width - 2 * iwt:width - iwt], 0), 0)
                return val, rois, [lbg, rbg]
        if num >= 2:
            print("More than one face!")
            return val
        rect = rects[0]
        # Perform landmark prediction on the face region
        face = coordTrans(image.shape, detectionSize, rect)
        # print(face)
        shape = self.predictor(image, face)
        landmarks = shape_to_np(shape)
        # Perfom landmarks smoothing
        if (self.rect != None):
            distFM = self.distForMarks(self.landmarks[medIdx], landmarks[medIdx])
            # print('\ndistFM:'+'{:.2f}'.format(distFM)+'\n')
            # print(self.distForMarks(self.landmarks[medIdx], landmarks[medIdx]))
            landmarks = self.smoothMarks(self.landmarks, landmarks, distFM)

        # ROI value
        rois = [np_to_bb(landmarks[idx], self.roiRatio) for idx in self.idx]
        mask = np.zeros([2, image.shape[0], image.shape[1]], dtype=int)
        val = [[], []]
        for i in range(0, len(rois)):
            # print(i)
            roi = rois[i]
            cv2.ellipse(mask[i], (roi[0], roi[1]), (roi[2], roi[3]), roi[4], -180, 180, 255, -1)
            mmsk = np.mean(image[mask[i] > 0], 0)
            val[i].append(mmsk)
        # print(val)
        # sys.exit(0)
        # plt.figure(1)
        # plt.subplot(1, 2, i + 1)
        # plt.imshow(mask[i])

        # plt.show()
        # cv2.waitKey()

        # vals = [np.mean(np.mean(image[roi[1]:roi[3], roi[0]:roi[2]], 0), 0) for roi in rois]
        # val = np.mean(vals, 0)
        # image 2160x3840,3
        height = image.shape[0]
        width = image.shape[1]
        iht = int(height / 4)
        iwt = int(width / 16)
        lbg = np.mean(np.mean(image[iht:height - iht, iwt:iwt * 2], 0), 0)
        rbg = np.mean(np.mean(image[iht:height - iht, width - 2 * iwt:width - iwt], 0), 0)

        self.rect = rect
        self.landmarks = landmarks
        self.face = face
        return val, rois, [lbg, rbg]

    def smoothMarks(self, landmarks1, landmarks2, distFM):
        smoothRatio = math.exp(-self.alpha * distFM)
        landmarks = smoothRatio * landmarks1 \
                    + (1 - smoothRatio) * landmarks2
        landmarks = np.array([[round(pair[0]), round(pair[1])]
                              for pair in landmarks])
        landmarks = landmarks.astype(int)
        return landmarks

    def distForMarks(self, mask1, mask2):
        """Calculate the distance between two rectangles for rectangle smoothing
        Arg:
            rect1, rect2: dlib.rectangle
        Return:
            distance between rectangles
        """
        dist = mask1 - mask2
        dist = np.sum(np.sqrt(np.sum(np.multiply(dist, dist), 1)))
        return dist


def VideoToTxt(videopath, PPGpath, subject, filename, startTime):
    # Initialization
    print('File ' + videopath + ' Extracting...')
    detect = Detector(predictorPath=predictorPath, predictorIdx=predictorIdx)
    times = []
    data = [[], [], []]
    data2 = [[], [], []]
    cbg = []
    rois_clt = []
    ED = []
    if os.path.exists(PPGpath):
        with open(PPGpath, 'r') as f:
            lines = f.readlines()
            ED = list(map(float, lines[1:]))

    video = cv2.VideoCapture(videopath)
    #    video = cv2.VideoCapture(0)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.set(cv2.CAP_PROP_POS_FRAMES, startTime * fps)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Handle frame one by one
    t = 0.0
    t_off = 1
    ret, frame = video.read()
    init_t = time.time()
    print('Video State: ', ret)

    while (video.isOpened()):
        t += 1.0 / fps
        # detect
        val = detect(frame, t)
        # print(type(val))
        if type(val) != list:
            value = val[0][0][0]
            value2 = val[0][1][0]
            rois = val[1]
            tc = val[2]

            # show result
            times.append(t)

            if type(value) != int:
                # print(value)
                for i in range(3):
                    data[i].append(value[i])
                    data2[i].append(value2[i])
                cbg.append(tc)
                rois_clt.append(rois)
            elif len(data[0]) > 1:
                for i in range(3):
                    data[i].append(data[i][-1])
                    data2[i].append(data2[i][-1])
                cbg.append(cbg[-1])
            # print(t,'Save mode ON:',savemode)
        # check stop or quit
        ret, frame = video.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:  # or t > t_off:
            break

    # release memory and destroy windows
    video.release()

    data = np.array(data)
    data2 = np.array(data2)
    nbg = np.array(cbg)
    #
    # if ED:
    #     yed = np.array(ED)
    #     np.savetxt(cutoutpath + '_ed.txt', yed, fmt='%f')
    nprc = np.array(rois_clt)
    return [data, data2, nbg[:, 0, :].T, nbg[:, 1, :].T, nprc[:, 0, :], nprc[:, 1, :]]


def ReadOut(subject):
    cofolder = ROUT + subject
    cutoutpath = cofolder + '/' + subject
    print(cutoutpath + '\nLoading file...\n')
    # data-3(BGR)*N
    data1 = np.loadtxt(cutoutpath + '_data1' + '.txt')
    data2 = np.loadtxt(cutoutpath + '_data2' + '.txt')
    # # bg-3(BGR)*N
    bg1 = np.loadtxt(cutoutpath + '_bg1' + '.txt')
    bg2 = np.loadtxt(cutoutpath + '_bg2' + '.txt')
    # # rc-N*5((x,y),(2b,2a),theta)
    rc1 = np.loadtxt(cutoutpath + '_roi1' + '.txt')
    rc2 = np.loadtxt(cutoutpath + '_roi2' + '.txt')
    # 8*N-1
    ed = []
    if os.path.exists(cutoutpath + '_ed.txt'):
        ed = np.loadtxt(cutoutpath + '_ed.txt')
    # return [data1, rc1, bg1], [data2, rc2, bg2], ed
    return data1, data2, bg1, bg2, rc1, rc2


def EDARead(subject):
    cofolder = './data_edav100/EDA/' + subject
    eda_data = np.loadtxt(cofolder + '_marked' + '.txt')
    return eda_data


if __name__ == "__main__":
    videofolder = r'D:\\MyResearch\Data\\MSE0501-02\\'
    videofolder = r'D:\\MyResearch\Data\\MSE0510\\'
    ExpDate = ['0509', '0510']
    ExpN = [4, 7]
    MVIN = 2
    ROUT = './data_edav100/'
    BOUT = './beat_split_edav64/'
    VIDEO_FIlE = []
    subjects = []
    for i in range(0, len(ExpDate)):
        for j in range(0, ExpN[i]):
            sub_file = [ExpDate[i] + '_' + '{:02d}'.format(j + 1) + '_' + '{:d}'.format(mvn + 1) + '.MP4' for mvn in
                        range(0, MVIN)]
            # subjects.append(ExpDate[i]+'_'+'{:02d}'.format(j+1))
            VIDEO_FIlE.append(sub_file)
    Video_Read = True
    Beats_Split = True
    Data_label = True
    Data_ob = False
    if Video_Read == False:
        for SF in VIDEO_FIlE:
            v_data1 = []
            v_data2 = []
            v_bkgd1 = []
            v_bkgd2 = []
            v_roi1 = []
            v_roi2 = []
            for vi in SF:
                fe = os.path.exists(videofolder + vi)
                print('File\n', videofolder + vi, '\nexisting: ', fe)
                subs = vi.split('_')
                subject = subs[0] + '_' + subs[1]
                print('subject: ', subject)
                # All_data: 3xn (RGBxframes), ROI:nx5
                if subs[2] == 1:
                    st = 60
                else:
                    st = 0
                [data1, data2, bg1, bg2, roi1, roi2] = VideoToTxt(videofolder + vi, '', subject, subject, st)
                v_data1.append(data1)
                v_data2.append(data2)
                v_bkgd1.append(bg1)
                v_bkgd2.append(bg2)
                v_roi1.append(roi1)
                v_roi2.append(roi2)
                print('File ', vi, ' Loading Finished')
            if v_data1 == []:
                continue

            av_data1 = np.concatenate(v_data1, 1)
            av_data2 = np.concatenate(v_data2, 1)
            av_bg1 = np.concatenate(v_bkgd1, 1)
            av_bg2 = np.concatenate(v_bkgd2, 1)
            av_roi1 = np.concatenate(v_roi1, 0)
            av_roi2 = np.concatenate(v_roi2, 0)
            cofolder = ROUT + subject
            cutoutpath = cofolder + '/' + subject
            if not os.path.exists(cofolder):
                os.makedirs(cofolder)

            np.savetxt(cutoutpath + '_data1' + '.txt', av_data1, fmt='%.4f')
            np.savetxt(cutoutpath + '_data2' + '.txt', av_data2, fmt='%.4f')

            np.savetxt(cutoutpath + '_bg1' + '.txt', av_bg1, fmt='%.4f')
            np.savetxt(cutoutpath + '_bg2' + '.txt', av_bg2, fmt='%.4f')

            np.savetxt(cutoutpath + '_roi1' + '.txt', av_roi1, fmt='%.4f')
            np.savetxt(cutoutpath + '_roi2' + '.txt', av_roi2, fmt='%.4f')

            print(SF, '\nFile Done.')
    if Video_Read == True and Beats_Split == False:
        # subjects = ['0501_01', '0501_02', '0501_03', '0502_01', '0502_03', '0502_04', '0502_05', '0502_06', '0502_07']
        subjects = ['0509_01', '0509_02', '0509_03', '0509_04',
                    '0510_01', '0510_02', '0510_03', '0510_04', '0510_05', '0510_06', '0510_07']
        # subjects = ['0510_01']
        # subjects = ['0501_02', '0501_03', '0502_02']
        fps = 50
        tt = 164  # -60 if it's 0509-0510
        ttf = tt * fps
        std = np.array([[164, 284], [298, 418], [436, 556], [570, 690], [708, 828], [840, 960]]) - tt
        cutoff1 = [0.5, 5]
        eda_ch = True
        for subject in subjects:
            # if subject != '0502_03':
            #     continue
            sdata1, sdata2, sbg1, sbg2, src1, src2 = ReadOut(subject)
            # data - PDA[0,1][0] 3(BGR)*N
            # rc - PDA[0,1][1] N*5 para((x,y),(2b,2a),theta)
            # bg - PDA[0,1][2] 3(BGR)*N
            # ed - PDA[2] (8*N-1,)

            if eda_ch == True:
                eda_data = EDARead(subject)
                eda_time = eda_data[:, 0]
                eda_conduct = eda_data[:, 1]
                eda_tonic = eda_data[:, 2]
                eda_tdata_sel = []
                eda_cdata_sel = []
                for sti in std:
                    ti_eda = (eda_time > sti[0]) & (eda_time < sti[1])
                    tempt = eda_tonic[ti_eda]
                    tempc = eda_conduct[ti_eda]
                    eda_tdata_sel.append(tempt)
                    eda_cdata_sel.append(tempc)
                eda_tscale = (eda_tonic - np.min(eda_tdata_sel)) / (np.max(eda_tdata_sel) - np.min(eda_tdata_sel))
                eda_cscale = (eda_conduct - np.min(eda_cdata_sel)) / (np.max(eda_cdata_sel) - np.min(eda_cdata_sel))
                eda_stage_scale = []

            o_data2 = sdata2[1][ttf:]
            my_imputer = impute.SimpleImputer()
            data_imputed = my_imputer.fit_transform(o_data2.reshape(1, -1))
            data2 = data_imputed.reshape(-1, order='C')
            bg2 = sbg2[1][ttf:]
            o_roi2 = src2[ttf:, :]
            # data2 = preprocessing.robust_scale(o_data2)

            times = np.array(range(0, data2.shape[0])) / fps
            b_but, a_but = signal.butter(6, [2 * cutoff1[0] / fps, 2 * cutoff1[1] / fps], 'bandpass')
            b_fir = signal.firwin2(512, [0, 2 * cutoff1[0] / fps, 2 * cutoff1[1] / fps, 1], [0, 1, 1, 0])
            cutoff2 = [0.7, 2]
            times = np.array(range(0, data2.shape[0])) / fps
            # ffd1 = signal.filtfilt(b_fir, 1, data2)
            fd2 = signal.filtfilt(b_but, a_but, data2)
            filt_data = signal.filtfilt(b_fir, 1, fd2)
            # smooth_filt_data = np.array(savgol(filt_data.tolist(), 7, 2))
            roi2 = o_roi2[0:filt_data.shape[0], :]

            # fig2 = plt.figure()
            # ax1 = fig2.add_subplot(221)
            # fb2 = signal.filtfilt(b_but, a_but, bg2)
            # fb2 = signal.filtfilt(b_fir, 1, fb2)
            # roi2_fbut = signal.filtfilt(b_but, a_but, roi2.T)
            # roi2_fbf = signal.filtfilt(b_fir, 1, roi2_fbut).T
            # ax1.plot(times, filt_data - np.mean(filt_data) + 2, label='face green channel', color='green')
            # # ax1.plot(times, fb2 - np.mean(fb2), label='backgroud', color='blue')
            # ax1.grid()
            # ax1.legend()
            # ax3 = fig2.add_subplot(223)
            # ax3.plot(times, data2)
            # ax3.grid()
            # ax4 = fig2.add_subplot(224)
            # ax4.psd(filt_data, NFFT=512, Fs=fps, window=mlab.window_none,
            #         scale_by_freq=True)
            # ax4.set_xlim([0, 5])
            # ax2 = fig2.add_subplot(222)
            # ax2.plot(times, roi2_fbf[:, 0] - np.mean(roi2_fbf[:, 0]) + 8, label='x')
            # ax2.plot(times, roi2_fbf[:, 1] - np.mean(roi2_fbf[:, 1]) + 6, label='y')
            # ax2.plot(times, roi2_fbf[:, 2] - np.mean(roi2_fbf[:, 2]) + 8, label='2b')
            # ax2.plot(times, roi2_fbf[:, 3] - np.mean(roi2_fbf[:, 3]) + 6, label='2a')
            # ax2.plot(times, roi2_fbf[:, 4] - np.mean(roi2_fbf[:, 4]) + 6, label='th')
            # ax2.grid()
            # ax2.legend()
            # sss
            # stage_beats_split(data2, subject, pfn)
            # sss
            fd_scale = []

            for sti in std:
                # print(sti)
                temp_data = filt_data[sti[0] * fps:sti[1] * fps]

                if eda_ch == True:
                    ti_eda = (eda_time > sti[0]) & (eda_time < sti[1])
                    tempt = eda_tscale[ti_eda]
                    tempc = eda_cscale[ti_eda]
                    eda_stage_scale.append(tempt)
                fd_scale.append(temp_data)
            # fd_scale = preprocessing.robust_scale(filt_data.reshape(-1, 1))[:].T[0]
            data_stage_scale = preprocessing.robust_scale(list(map(list, zip(*fd_scale))))
            if eda_ch == True:
                eda_stage_scale = np.array(eda_stage_scale).T
            b_c1, a_c1 = signal.butter(6, [2 * cutoff2[0] / fps, 2 * cutoff2[1] / fps], 'bandpass')
            for k in range(0, data_stage_scale.shape[1]):
                data_temp = data_stage_scale[:, k]
                if eda_ch == True:
                    eda_temp = eda_stage_scale[:, k]
                ffds = signal.filtfilt(b_c1, a_c1, data_temp)
                fpeak = signal.argrelextrema(ffds, np.greater)[0]
                data = np.array(data_temp)
                for i in range(0, len(fpeak)):
                    pkn = 3
                    if fpeak[i] + pkn >= len(data) - pkn or fpeak[i] - pkn < pkn:
                        continue
                    for j in range(0, pkn):
                        while data[fpeak[i] + pkn - j] > data[fpeak[i]]:
                            if fpeak[i] + pkn - j >= len(data) - pkn:
                                break
                            fpeak[i] = fpeak[i] + pkn - j
                        while data[fpeak[i] - pkn + j] > data[fpeak[i]]:
                            if fpeak[i] - pkn + j < pkn:
                                break
                            fpeak[i] = fpeak[i] - pkn + j
                rec_data = np.copy(data)
                normalized_para_A = np.zeros([len(fpeak) - 1, 1])
                normalized_para_T = np.zeros([len(fpeak) - 1, 1])
                if eda_ch == True:
                    normalized_EDA = np.zeros([len(fpeak) - 1, 1])
                availability = np.ones([len(fpeak) - 1, 1], dtype=int)
                normalized_para_beats = np.zeros([len(fpeak) - 1, 64])
                for nt in range(1, len(fpeak)):
                    temp = np.array(rec_data[fpeak[nt - 1]:(fpeak[nt] + 1)])
                    if len(temp) < 30 or len(temp) > 75:
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = 0
                        availability[nt - 1] = -1
                        print('????')
                        continue
                    t0 = temp[0]
                    ln = fpeak[nt] + 1 - fpeak[nt - 1]
                    normalized_para_T[nt - 1] = ln / fps
                    if eda_ch == True:
                        normalized_EDA[nt - 1] = np.mean(eda_temp[fpeak[nt - 1] * 2:fpeak[nt] * 2])
                    dy = (temp[-1] - t0) / (ln - 1)
                    for nj in range(0, ln):
                        temp[nj] = temp[nj] - dy * nj - t0
                    temp_peak_x = signal.argrelextrema(temp, np.greater)[0]
                    temp = -temp
                    if len(temp_peak_x) == 0:
                        temp_sth = temp.max()
                    else:
                        temp_peak_y = temp[temp_peak_x]
                        temp_sth = np.min(temp_peak_y)
                    if temp.min() < -0.05 or temp.max() > 6 or temp.max() < 0.4 or (
                            temp_sth < 0.2 * temp.max() and len(temp_peak_x) > 1):
                        availability[nt - 1] = 0
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = temp.min()
                        normalized_para_A[nt - 1] = temp.max()
                    else:
                        normalized_para_A[nt - 1] = temp.max()
                        temp = temp / normalized_para_A[nt - 1]
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = temp[0:-1]
                        xnew = np.linspace(fpeak[nt - 1], fpeak[nt], 64)
                        f = interpolate.interp1d(range(fpeak[nt - 1], fpeak[nt] + 1), temp, kind="slinear")
                        ynew = f(xnew)
                        normalized_para_beats[nt - 1] = ynew

                # plt.figure(1)
                # plt.subplot(2, 1, 1)
                # xt = np.array(range(0, rec_data.shape[0]))
                # xt = xt / 50
                # plt.plot(xt, rec_data, label='raw')
                # plt.plot(xt, data - np.mean(data) - 2, label='fdsc')
                # # plt.plot(xt, ffds - np.mean(ffds) + 4, label='filt')
                # avl = np.where(availability == 1)[0]
                # # plt.plot(fpeak / 50, rec_data[fpeak], linestyle=':', marker='.', label='pk-s')
                # avp = avl+1
                # plt.plot(fpeak[avp]/50, rec_data[fpeak[avp]], linestyle=':',marker='.', label='pk-e')
                # plt.legend()
                # plt.grid()
                # plt.subplot(2, 1, 2)
                # plt.psd(rec_data, NFFT=256, Fs=fps, window=mlab.window_none,
                #         scale_by_freq=True)
                # plt.xlim([0, 5])
                # plt.figure(2)
                # plt.plot(availability, label='A')
                # plt.show()
                # sss
                filename = subject
                cofolder = BOUT + subject
                cutoutpath = cofolder + '/' + filename
                if not os.path.exists(cofolder):
                    os.makedirs(cofolder)
                # split_path =
                print(filename + ' Saving Data_' + '{:02d}'.format(k) + '... ')
                np.savetxt(cutoutpath + '_rec_data_' + '{:02d}'.format(k) + '.txt', rec_data, fmt='%.4f')
                np.savetxt(cutoutpath + '_para_A_' + '{:02d}'.format(k) + '.txt', normalized_para_A, fmt='%.4f')
                np.savetxt(cutoutpath + '_para_T_' + '{:02d}'.format(k) + '.txt', normalized_para_T, fmt='%.4f')
                if eda_ch == True:
                    np.savetxt(cutoutpath + '_EDA_' + '{:02d}'.format(k) + '.txt', normalized_EDA, fmt='%.4f')
                np.savetxt(cutoutpath + '_para_beats_' + '{:02d}'.format(k) + '.txt', normalized_para_beats, fmt='%.4f')
                np.savetxt(cutoutpath + '_availability_' + '{:02d}'.format(k) + '.txt', availability, fmt='%d')
                np.savetxt(cutoutpath + '_fpeak_' + '{:02d}'.format(k) + '.txt', fpeak, fmt='%d')
    if Video_Read == True and Beats_Split == True and Data_label == False:
        # subjects = ['0501_01', '0501_02', '0501_03', '0502_01', '0502_02', '0502_03', '0502_04', '0502_05', '0502_06', '0502_07']
        subjects = ['0501_01', '0501_02', '0501_03',
                    '0502_01', '0502_02', '0502_03', '0502_04', '0502_05', '0502_06', '0502_07',
                    '0509_01', '0509_02', '0509_03', '0509_04',
                    '0510_01', '0510_02', '0510_03', '0510_04', '0510_06', '0510_07']
        # subjects = ['0501_02', '0501_03', '0502_02']
        # labels = np.array([[3, 1, 3, 2, 3, 0],
        #                    [3, 0, 3, 2, 3, 1],
        #                    ])
        labels = np.array([[3, 1, 4, 0, 5, 2],
                           [3, 1, 4, 2, 5, 0],
                           [3, 0, 4, 2, 5, 1],

                           [3, 0, 4, 2, 5, 1],
                           [3, 1, 4, 0, 5, 2],
                           [3, 0, 4, 2, 5, 1],
                           [3, 0, 4, 2, 5, 1],
                           [3, 2, 4, 0, 5, 1],
                           [3, 1, 4, 0, 5, 2],
                           [3, 1, 4, 2, 5, 0],

                           [3, 0, 4, 2, 5, 1],
                           [3, 1, 4, 2, 5, 0],
                           [3, 0, 4, 2, 5, 1],
                           [3, 0, 4, 2, 5, 1],

                           [3, 2, 4, 0, 5, 1],
                           [3, 1, 4, 2, 5, 0],
                           [3, 1, 4, 0, 5, 2],
                           [3, 1, 4, 2, 5, 0],
                           [3, 1, 4, 0, 5, 2],
                           [3, 1, 4, 0, 5, 2]])
        si = -1
        for subject in subjects:
            # if subject != subjects[1]:
            #     continue
            si = si + 1
            data = []
            fi = 0
            eda_ch = True
            cofolder = BOUT + subject
            cutoutpath = cofolder + '/' + subject
            while (True):
                rec_file = cutoutpath + '_rec_data_' + '{:02d}'.format(fi) + '.txt'
                para_A_file = cutoutpath + '_para_A_' + '{:02d}'.format(fi) + '.txt'
                para_T_file = cutoutpath + '_para_T_' + '{:02d}'.format(fi) + '.txt'
                if eda_ch == True:
                    EDA_file = cutoutpath + '_EDA_' + '{:02d}'.format(fi) + '.txt'
                para_beats_file = cutoutpath + '_para_beats_' + '{:02d}'.format(fi) + '.txt'
                avail_file = cutoutpath + '_availability_' + '{:02d}'.format(fi) + '.txt'
                fpeak_file = cutoutpath + '_fpeak_' + '{:02d}'.format(fi) + '.txt'

                if not os.path.exists(rec_file):
                    break
                else:
                    print(subject + ' loading Data_' + '{:02d}'.format(fi) + '... ')
                    rec_data = np.loadtxt(rec_file)  # 6000,
                    para_A = np.loadtxt(para_A_file)  # 147,
                    para_T = np.loadtxt(para_T_file)  # 147,
                    if eda_ch == True:
                        EDA = np.loadtxt(EDA_file)  # 147,
                    para_beats = np.loadtxt(para_beats_file)  # 147x64
                    availability = np.loadtxt(avail_file)
                    fpeak = np.loadtxt(fpeak_file)
                    avl = np.where(availability == 1)[0]
                    m_A = para_A[avl]  # signal.savgol_filter(para_A[avl],5,2)
                    m_T = para_T[avl]  # signal.savgol_filter(para_T[avl],5,2)
                    if eda_ch == True:
                        m_eda = EDA[avl]  #
                    nback = np.zeros(m_A.shape) + labels[si][fi]
                    if eda_ch == True:
                        m_data = np.vstack([m_A, m_T, para_beats[avl].T, m_eda, nback])  # 68x294
                    else:
                        m_data = np.vstack([m_A, m_T, para_beats[avl].T, nback])
                    m_data = m_data[:, 3:-3]
                    data.append(m_data)
                    print('data lenght: ' + str(len(data)))
                fi = fi + 1
            # rst = random.sample(range(10, len(fpeak) - 10), 30)
            # wn = 7
            # for ri in rst:
            #     while sum(availability[ri - wn:ri + wn]) < wn + 1:
            #         ri = random.randint(10, len(fpeak) - 10)
            #     temp_beats = np.array(para_beats[ri - wn:ri + wn, :])
            #     rbs = temp_beats[np.where(availability[ri - wn:ri + wn] == 1)[0], :].T

            # plt.figure(1)
            # plt.subplot(2,2,1)
            # plt.plot(para_A[avl])
            # plt.subplot(2, 2, 2)
            # plt.plot(m_A)
            # plt.subplot(2, 2, 3)
            # plt.plot(para_T[avl])
            # plt.subplot(2, 2, 4)
            # plt.plot(m_T)
            # plt.figure(2)
            # plt.plot(para_beats[avl, :].T)
            # plot_acf(rec_data).show()
            # # plot_pacf(rec_data).show()
            # rx = np.array(range(0,1000))
            # recy = np.sin(rx)+np.sin(2*rx)/2+np.sin(4*rx)/4
            # plot_acf(recy).show()
            # # plot_pacf(recy).show()
            # sss

            # plt.figure()
            # plt.subplot(2,2,1)
            # plt.plot(m_A)
            # plt.grid()
            # plt.title('para_A '+subject+' '+vc)
            # plt.subplot(2,2,2)
            # plt.plot(m_T)
            # plt.grid()
            # plt.title('para_T '+subject+' '+vc)
            # plt.subplot(2,2,3)
            # plt.plot(p1_rbs)
            # plt.grid()
            # plt.title('base-beat '+subject+' '+vc)
            # plt.subplot(2,2,4)
            # plt.plot(m_data[2:,:])
            # plt.grid()
            # plt.title('pca_data '+subject+' '+vc)
            s_data = np.concatenate(data, 1)
            ldfolder = './data_ATB64_back'
            ldpath = ldfolder + '/data_' + subject
            if not os.path.exists(ldfolder):
                os.makedirs(ldfolder)
            print('subject ' + subject + ' Saving Data... ')
            np.savetxt(ldpath + '.txt', s_data, fmt='%.4f')
            print('subject ' + subject + ' Done... ')
    if Data_ob == False:
        subjects = ['0501_01', '0501_02', '0501_03', '0502_01', '0502_02', '0502_03', '0502_04', '0502_05', '0502_06',
                    '0502_07']
        # subjects = ['0501_02', '0501_03']
        fps = 50
        tt = 164
        ttf = tt * fps
        std = np.array([[164, 284], [298, 418], [436, 556], [570, 690], [708, 828], [840, 960]]) - tt
        cutoff1 = [0.5, 5]
        for subject in subjects:
            if subject != '0501_02': #[3, 1, 4, 2, 5, 0],
                continue
            sdata1, sdata2, sbg1, sbg2, src1, src2 = ReadOut(subject)
            # data - PDA[0,1][0] 3(BGR)*N
            # rc - PDA[0,1][1] N*5 para((x,y),(2b,2a),theta)
            # bg - PDA[0,1][2] 3(BGR)*N
            # ed - PDA[2] (8*N-1,)
            eda_data = EDARead(subject)
            eda_time = eda_data[:, 0]
            eda_tdata = eda_data[:, 1]
            eda_cdata = eda_data[:, 2]
            data2 = sdata2[1][ttf:]
            bg2 = sbg2[1][ttf:]
            roi2 = src2[ttf:, :]

            times = np.array(range(0, data2.shape[0])) / fps
            b_but, a_but = signal.butter(6, [2 * cutoff1[0] / fps, 2 * cutoff1[1] / fps], 'bandpass')
            b_fir = signal.firwin2(256, [0, 2 * cutoff1[0] / fps, 2 * cutoff1[1] / fps, 1], [0, 1, 1, 0])
            cutoff2 = [0.7, 2]
            times = np.array(range(0, data2.shape[0])) / fps
            fd2 = signal.filtfilt(b_but, a_but, data2)
            filt_data = signal.filtfilt(b_fir, 1, fd2)
            # smooth_filt_data = np.array(savgol(filt_data.tolist(), 7, 2))

            # fig2 = plt.figure()
            # ax1 = fig2.add_subplot(221)
            # fd2 = signal.filtfilt(b_but, a_but, data2)
            # fd2 = signal.filtfilt(b_fir, 1, fd2)
            # fb2 = signal.filtfilt(b_but, a_but, bg2)
            # fb2 = signal.filtfilt(b_fir, 1, fb2)
            # roi2_fbut = signal.filtfilt(b_but, a_but, roi2.T)
            # roi2_fbf = signal.filtfilt(b_fir, 1, roi2_fbut).T
            # ax1.plot(times, fd2 - np.mean(fd2) + 2, label='face green channel', color='green')
            # ax1.plot(times, fb2 - np.mean(fb2), label='backgroud', color='blue')
            # ax1.grid()
            # ax1.legend()
            # ax3 = fig2.add_subplot(223)
            # logfd2 = np.log(fd2 + 10)
            # b_fir2 = signal.firwin2(512, [0, 2 * 0.5 / fps, 2 * 1 / fps, 1], [1, 1, 0, 0])
            # lfd2 = signal.filtfilt(b_fir2, 1, data2)
            # ax3.plot(times, data2)
            # ax3.plot(times, lfd2)
            # ax3.grid()
            # ax4 = fig2.add_subplot(224)
            #
            # ax4.psd(fd2, NFFT=512, Fs=fps, window=mlab.window_none,
            #         scale_by_freq=True)
            # ax4.set_xlim([0, 5])
            # ax2 = fig2.add_subplot(222)
            # ax2.plot(times, roi2_fbf[:, 0] - np.mean(roi2_fbf[:, 0]) + 8, label='x')
            # ax2.plot(times, roi2_fbf[:, 1] - np.mean(roi2_fbf[:, 1]) + 6, label='y')
            # ax2.plot(times, roi2_fbf[:, 2] - np.mean(roi2_fbf[:, 2]) + 8, label='2b')
            # ax2.plot(times, roi2_fbf[:, 3] - np.mean(roi2_fbf[:, 3]) + 6, label='2a')
            # ax2.plot(times, roi2_fbf[:, 4] - np.mean(roi2_fbf[:, 4]) + 6, label='th')
            # ax2.grid()
            # ax2.legend()
            # sss
            # stage_beats_split(data2, subject, pfn)
            # sss
            fd_scale = []
            eda_tdata_sel = []
            eda_cdata_sel = []
            for sti in std:
                ti_eda = (eda_time > sti[0]) & (eda_time < sti[1])
                tempt = eda_tdata[ti_eda]
                tempc = eda_cdata[ti_eda]
                eda_tdata_sel.append(tempt)
                eda_cdata_sel.append(tempc)
            eda_tscale = (eda_tdata - np.min(eda_tdata_sel)) / (np.max(eda_tdata_sel) - np.min(eda_tdata_sel))
            eda_cscale = (eda_cdata - np.min(eda_cdata_sel)) / (np.max(eda_cdata_sel) - np.min(eda_cdata_sel))
            eda_stage_scale = []
            for sti in std:
                # print(sti)
                temp_data = filt_data[sti[0] * fps:sti[1] * fps]
                ti_eda = (eda_time > sti[0]) & (eda_time < sti[1])
                tempt = eda_tscale[ti_eda]
                tempc = eda_cscale[ti_eda]
                eda_stage_scale.append(tempt)
                fd_scale.append(temp_data)
            # fd_scale = preprocessing.robust_scale(filt_data.reshape(-1, 1))[:].T[0]
            data_stage_scale = preprocessing.robust_scale(list(map(list, zip(*fd_scale))))
            eda_stage_scale = np.array(eda_stage_scale).T
            b_c1, a_c1 = signal.butter(6, [2 * cutoff2[0] / fps, 2 * cutoff2[1] / fps], 'bandpass')
            for k in range(0, data_stage_scale.shape[1]):
                if k != 3:
                    continue
                data_temp = data_stage_scale[:, k]
                eda_temp = eda_stage_scale[:, k]
                # f ,t ,Sxx = signal.spectrogram(data_temp, fs=50, window='', nperseg=1024, noverlap=512)
                # ssss
                ffds = signal.filtfilt(b_c1, a_c1, data_temp)
                fpeak = signal.argrelextrema(ffds, np.greater)[0]
                data = np.array(data_temp)
                for i in range(0, len(fpeak)):
                    pkn = 3
                    if fpeak[i] + pkn >= len(data) - pkn or fpeak[i] - pkn < pkn:
                        continue
                    for j in range(0, pkn):
                        while data[fpeak[i] + pkn - j] > data[fpeak[i]]:
                            if fpeak[i] + pkn - j >= len(data) - pkn:
                                break
                            fpeak[i] = fpeak[i] + pkn - j
                        while data[fpeak[i] - pkn + j] > data[fpeak[i]]:
                            if fpeak[i] - pkn + j < pkn:
                                break
                            fpeak[i] = fpeak[i] - pkn + j
                rec_data = np.copy(data)
                normalized_para_A = np.zeros([len(fpeak) - 1, 1])
                normalized_para_T = np.zeros([len(fpeak) - 1, 1])
                normalized_EDA = np.zeros([len(fpeak) - 1, 1])
                availability = np.ones([len(fpeak) - 1, 1], dtype=int)
                normalized_para_beats = np.zeros([len(fpeak) - 1, 64])
                for nt in range(1, len(fpeak)):
                    temp = np.array(rec_data[fpeak[nt - 1]:(fpeak[nt] + 1)])
                    if len(temp) < 20 or len(temp) > 80:
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = 0
                        availability[nt - 1] = -1
                        # print('{%d,%d}: ????' % (k, nt))
                        continue
                    t0 = temp[0]
                    ln = fpeak[nt] + 1 - fpeak[nt - 1]
                    normalized_para_T[nt - 1] = ln / fps
                    normalized_EDA[nt - 1] = np.mean(eda_temp[fpeak[nt - 1] * 2:fpeak[nt] * 2])
                    dy = (temp[-1] - t0) / (ln - 1)
                    for nj in range(0, ln):
                        temp[nj] = temp[nj] - dy * nj - t0
                    temp_peak_x = signal.argrelextrema(temp, np.greater)[0]
                    temp = -temp
                    if len(temp_peak_x) == 0:
                        temp_sth = temp.max()
                    else:
                        temp_peak_y = temp[temp_peak_x]
                        temp_sth = np.min(temp_peak_y)
                    if temp.min() < -0.05 or temp.max() > 4 or temp.max() < 0.4 or (
                            temp_sth < 0.3 * temp.max() and len(temp_peak_x) > 1):
                        availability[nt - 1] = 0
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = temp.min()
                        normalized_para_A[nt - 1] = temp.max()
                        print('{%.4f,%.4f, %.4f}: ????' % (temp[-1], temp.max(), temp.min()))
                    else:
                        normalized_para_A[nt - 1] = temp.max()
                        temp = temp / normalized_para_A[nt - 1]
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = temp[0:-1]
                        xnew = np.linspace(fpeak[nt - 1], fpeak[nt], 64)
                        f = interpolate.interp1d(range(fpeak[nt - 1], fpeak[nt] + 1), temp, kind="slinear")
                        ynew = f(xnew)
                        normalized_para_beats[nt - 1] = ynew

                plt.figure(1)
                plt.subplot(2, 1, 1)
                xt = np.array(range(0, rec_data.shape[0]))
                xt = xt / 50
                # plt.plot(xt, rec_data, label='rec')
                plt.plot(xt, - data + np.mean(data), label='FiltData')
                # plt.plot(xt, ffds - np.mean(ffds) + 1, label='filt')
                avl = np.where(availability == 1)[0]
                # plt.plot(fpeak / 50, rec_data[fpeak], linestyle=':', marker='.', label='pk-s')
                avp = avl + 1
                # plt.plot(fpeak[avp] / 50, rec_data[fpeak[avp]], linestyle=':', marker='.', label='pk-e')
                plt.legend()
                plt.grid()
                plt.subplot(2, 1, 2)
                plt.psd(rec_data, NFFT=256, Fs=fps, window=mlab.window_none,
                        scale_by_freq=True)
                plt.xlim([0, 5])
                plt.figure(2)
                plt.plot(availability, label='A')
                plt.show()
                sss

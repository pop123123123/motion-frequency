#!/usr/bin/env python

'''

USAGE: opt_flow.py [<video_source>]

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import signal as sig

from tqdm import tqdm
import code
import time

from scene_detect import getSceneList

import video


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def running_mean(x, N):
    if N > x.shape[0]:
        print("Warning: running_mean with window size > vector size. Returning vector instead")
        return x
    #cumsum = np.cumsum(np.insert(x, 0, 0))
    # return (cumsum[N:] - cumsum[:-N]) / float(N)
    conv = np.convolve(x, np.ones((N,))/N, mode='valid')
    return np.concatenate((conv, x[conv.shape[0] - x.shape[0]:]))

def reduce(image):
    while image.shape[0]*image.shape[1] > 40000:
        image = cv.pyrDown(image)
    return image

def main():
    import sys
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0
    try:
        compute_scene = len(sys.argv[2]) <= 0
    except IndexError:
        compute_scene = True

    if compute_scene:
        scenelist = getSceneList(fn)
    else:
        scenelist = [(0, float("inf"))]
    if scenelist is None or len(scenelist) == 0:
        scenelist = [(0, float("inf"))]
    cam = video.create_capture(fn)
    nb_frames = int(cam.get(cv.CAP_PROP_FRAME_COUNT))
    rate = cam.get(cv.CAP_PROP_FPS)#/2
    ret, prev = cam.read()
    realh, realw = prev.shape[:2]
    prev = reduce(prev)
    #h, w = realh//3, realw//3
    h, w = prev.shape[:2]

    #prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)[w:2*w,h:2*h]
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    
    pbar = tqdm(total=nb_frames)

    frame = 0
    scene_results = []
    for scene_start, scene_end in scenelist:
        for i in range(frame, scene_start):
            ret, prev = cam.read()
        
        if scene_start - frame > 0:
            prev = reduce(prev)
            prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        frame = scene_start

        data = []
        while frame < scene_end:
            ret, img = cam.read()
            if not ret:
                break
            img = reduce(img)
            #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)[w:2*w,h:2*h]
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            flow = cv.calcOpticalFlowFarneback(
                prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prevgray = gray

            # data.append(np.absolute(flow).mean(axis=(0,1)))#Si plusieurs mouvements, ils peuvent s'annuler
            flow0 = flow.mean(axis=(0, 1))

            if len(data) > 0 and np.absolute(maxi).sum() < 12:#skip the next frame
                data.append(np.stack((data[-1], flow0)).mean(axis=0))
            data.append(flow0)
            maxi = flow.max(axis=(0, 1))

            pbar.update(1)
            frame += 1
            
            if frame < scene_end and np.absolute(maxi).sum() < 12:#skip the next frame
                pbar.update(1)
                frame += 1
                if frame == scene_end:
                    data.append(flow0)
                ret, img = cam.read()
            #cv.imshow('flow', draw_flow(gray, flow))
            #ch = cv.waitKey(5)

        # code.interact(local=locals())
        t = np.arange(len(data))
        freq = np.fft.rfftfreq(t.shape[-1], 1/rate)
        maxima = None
        freqs = None
        t = None
        
        for i in range(2):
            signal = np.array(data)[:, i]
            #data = np.linalg.norm(data, axis=1)
            #data = np.sum(data, axis=1)
            signal = running_mean(signal, 3)

            signal = np.array(signal)
            #signal -= signal.min()
            #signal /= signal.max()
            #signal = signal*2 - 1

            treshold = 1/2

            f, t, Zxx = sig.stft(signal, rate, boundary='even', nperseg=min(3*rate, signal.shape[0]), noverlap=min(3*rate-1, signal.shape[0]-1))
            #Zxx = Zxx * (f>=treshold)[:, np.newaxis]

            #plt.pcolormesh(t, f, np.abs(Zxx), vmin=0)
            #plt.title('STFT Magnitude')
            #plt.ylabel('Frequency [Hz]')
            #plt.xlabel('Time [sec]')
            #plt.show()
#scipy.special.erf(z)
            maxi = np.abs(Zxx).max(axis=0)
            if maxima is None:
                maxima = maxi
            else:
                maxima = np.stack((maxima, maxi))
            
            freq = np.take(f, np.abs(Zxx).argmax(axis=0))
            if freqs is None:
                freqs = freq
            else:
                freqs = np.stack((freqs, freq))
            #plt.show()

            #maxima[i] = np.absolute(sp[first:]).max()
            #indices[i] = np.unravel_index(np.absolute(
            #    sp[first:]).argmax(), sp.shape)[0] + first
        freq = maxima.argmax(axis=0).choose(freqs)
        freq = running_mean(freq, min(10, freq.shape[0]))
        #plt.plot(t, freq)
        #plt.show()
        scene_results.append((t + scene_start/rate, freq))

    pbar.close()
    #for t, freq in scene_results:
    #plt.plot(np.array([i/rate for i in range(nb_frames)]),
    fig, ax = plt.subplots()
    plt.plot(np.concatenate([t for t, freq in scene_results]),
        np.concatenate([freq for t, freq in scene_results]))

    formatter = ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax.xaxis.set_major_formatter(formatter)
    plt.show()
    #print(freq[indices[maxima.argmax()]])


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

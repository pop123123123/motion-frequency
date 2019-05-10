#!/usr/bin/env python

'''
example to show optical flow

USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang*(180/np.pi/2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res


def running_mean(x, N):
    #cumsum = np.cumsum(np.insert(x, 0, 0))
    # return (cumsum[N:] - cumsum[:-N]) / float(N)
    conv = np.convolve(x, np.ones((N,))/N, mode='valid')
    return np.concatenate((conv, x[conv.shape[0] - x.shape[0]:]))


def main():
    import sys
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    cam = video.create_capture(fn)
    rate = cam.get(cv.CAP_PROP_FPS)
    ret, prev = cam.read()
    realh, realw = prev.shape[:2]
    #h, w = realh//3, realw//3
    h, w = realh, realw

    #prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)[w:2*w,h:2*h]
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    data = []

    while True:
        ret, img = cam.read()
        if not ret:
            break
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)[w:2*w,h:2*h]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(
            prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        # data.append(np.absolute(flow).mean(axis=(0,1)))#Si plusieurs mouvements, ils peuvent s'annuler
        data.append(flow.mean(axis=(0, 1)))

        cv.imshow('flow', draw_flow(gray, flow))
        if show_hsv:
            cv.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv.imshow('glitch', cur_glitch)

        ch = cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])

    print('Done')
    import code
    import matplotlib.pyplot as plt
    # code.interact(local=locals())
    t = np.arange(len(data))
    freq = np.fft.rfftfreq(t.shape[-1], 1/rate)
    maxima = np.array([0, 0])
    indices = np.array([0, 0])
    for i in range(2):
        signal = np.array(data)[:, i]
        #data = np.linalg.norm(data, axis=1)
        #data = np.sum(data, axis=1)
        signal = running_mean(signal, 3)

        signal = np.array(signal)
        #signal -= signal.min()
        #signal /= signal.max()
        #signal = signal*2 - 1
        plt.plot(t, signal)
        plt.show()

        first = np.argmax(freq > 1/3)  # Can return 0
        #first = np.argmax(freq > 0)
        sp = np.fft.rfft(signal)
        maxima[i] = np.absolute(sp[first:]).max()
        # code.interact(local=locals())
        indices[i] = np.unravel_index(np.absolute(
            sp[first:]).argmax(), sp.shape)[0] + first

    #plt.plot(freq, sp.real, freq, sp.imag)
    # plt.show()
    print(freq[indices[maxima.argmax()]])


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

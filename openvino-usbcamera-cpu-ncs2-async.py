import sys
import os
import numpy as np
import cv2
from os import system
import io
import time
from os.path import isfile
from os.path import join
import re
import argparse
import platform
try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin
import multiprocessing as mp
from time import sleep
import threading
import heapq


def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    contours = None
    try:
        #OpenCV4.x
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        #OpenCV3.x
        _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


def getValidPairs(detected_keypoints, outputs, w, h):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7

    for k in range(len(mapIdx)):
        pafA = outputs[0, mapIdx[k][0], :, :]
        pafB = outputs[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (w, h))
        pafB = cv2.resize(pafB, (w, h))

        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            valid_pairs.append(valid_pair)
        else:
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


processes = []
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
lastresults = None

keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2,17], [5,16]]
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]
colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255], [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255], [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


def image_preprocessing(color_image, w, h, new_w, new_h):
    resized_image = cv2.resize(color_image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.full((h, w, 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    return canvas


def camThread(results, frameBuffer, camera_width, camera_height, vidfps, nPoints, w, h, new_w, new_h):
    global fps
    global detectfps
    global lastresults
    global framecount
    global detectframecount
    global time1
    global time2
    global cam
    global window_name

    cam = cv2.VideoCapture(0)
    if cam.isOpened() != True:
        print("USB Camera Open Error!!!")
        sys.exit(0)
    cam.set(cv2.CAP_PROP_FPS, vidfps)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    window_name = "USB Camera"
    wait_key_time = 1

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        t1 = time.perf_counter()

        # USB Camera Stream Read
        s, color_image = cam.read()
        if not s:
            continue
        if frameBuffer.full():
            frameBuffer.get()

        color_image = image_preprocessing(color_image.copy(), w, h, new_w, new_h)
        frameClone = np.uint8(color_image.copy())
        frameBuffer.put(color_image)

        if not results.empty():
            detected_keypoints, outputs, keypoints_list = results.get(False)
            detectframecount += 1

            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

            valid_pairs, invalid_pairs = getValidPairs(detected_keypoints, outputs, w, h)
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

            cv2.putText(frameClone, fps, (w-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
            lastresults = [detected_keypoints, outputs, keypoints_list]
        else:
            if not isinstance(lastresults, type(None)):
                detected_keypoints, outputs, keypoints_list = lastresults
                for i in range(nPoints):
                    for j in range(len(detected_keypoints[i])):
                        cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

                valid_pairs, invalid_pairs = getValidPairs(detected_keypoints, outputs, w, h)
                personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

                for i in range(17):
                    for n in range(len(personwiseKeypoints)):
                        index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                        if -1 in index:
                            continue
                        B = np.int32(keypoints_list[index.astype(int), 0])
                        A = np.int32(keypoints_list[index.astype(int), 1])
                        cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

        cv2.putText(frameClone, fps,       (w-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(frameClone, detectfps, (w-170,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, frameClone)

        if cv2.waitKey(wait_key_time)&0xFF == ord('q'):
            sys.exit(0)

        ## Print FPS
        framecount += 1
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime


# l = Search list
# x = Search target value
def searchlist(l, x, notfoundvalue=-1):
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue


def async_infer(ncsworker):

    #ncsworker.skip_frame_measurement()

    while True:
        ncsworker.predict_async()


class NcsWorker(object):

    def __init__(self, devid, device, model_xml, frameBuffer, results, camera_width, camera_height, number_of_ncs, vidfps, nPoints, w, h, new_w, new_h):
        self.devid = devid
        self.frameBuffer = frameBuffer
        self.model_xml = model_xml
        self.model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.threshold = 0.1
        self.nPoints = nPoints
        self.num_requests = 4
        self.inferred_request = [0] * self.num_requests
        self.heap_request = []
        self.inferred_cnt = 0
        self.plugin = IEPlugin(device=device)
        if "CPU" == device:
            if platform.processor() == "x86_64":
                self.plugin.add_cpu_extension("lib/libcpu_extension.so")
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(self.net.inputs))
        self.exec_net = self.plugin.load(network=self.net, num_requests=self.num_requests)
        self.results = results
        self.number_of_ncs = number_of_ncs
        self.predict_async_time = 250
        self.skip_frame = 0
        self.roop_frame = 0
        self.vidfps = vidfps
        self.w = w #432
        self.h = h #368
        self.new_w = new_w
        self.new_h = new_h


    def skip_frame_measurement(self):
            surplustime_per_second = (1000 - self.predict_async_time)
            if surplustime_per_second > 0.0:
                frame_per_millisecond = (1000 / self.vidfps)
                total_skip_frame = surplustime_per_second / frame_per_millisecond
                self.skip_frame = int(total_skip_frame / self.num_requests)
            else:
                self.skip_frame = 0


    def predict_async(self):
        try:

            if self.frameBuffer.empty():
                return

            self.roop_frame += 1
            if self.roop_frame <= self.skip_frame:
               self.frameBuffer.get()
               return
            self.roop_frame = 0

            prepimg = self.frameBuffer.get()
            reqnum = searchlist(self.inferred_request, 0)

            if reqnum > -1:
                prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
                prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW, (1, 3, 368, 432)
                self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: prepimg})
                self.inferred_request[reqnum] = 1
                self.inferred_cnt += 1
                if self.inferred_cnt == sys.maxsize:
                    self.inferred_request = [0] * self.num_requests
                    self.heap_request = []
                    self.inferred_cnt = 0
                heapq.heappush(self.heap_request, (self.inferred_cnt, reqnum))

            try:
                cnt, dev = heapq.heappop(self.heap_request)
            except:
                return

            if self.exec_net.requests[dev].wait(0) == 0:
                self.exec_net.requests[dev].wait(-1)

                detected_keypoints = []
                keypoints_list = np.zeros((0, 3))
                keypoint_id = 0

                outputs = self.exec_net.requests[dev].outputs["Openpose/concat_stage7"]
                for part in range(self.nPoints):
                    probMap = outputs[0, part, :, :]
                    probMap = cv2.resize(probMap, (self.w, self.h)) # (432, 368)
                    keypoints = getKeypoints(probMap, self.threshold)
                    keypoints_with_id = []

                    for i in range(len(keypoints)):
                        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                        keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                        keypoint_id += 1

                    detected_keypoints.append(keypoints_with_id)

                self.results.put([detected_keypoints, outputs, keypoints_list])
                self.inferred_request[dev] = 0
            else:
                heapq.heappush(self.heap_request, (cnt, dev))
        except:
            import traceback
            traceback.print_exc()


def inferencer(device, model_xml, results, frameBuffer, number_of_ncs, camera_width, camera_height, vidfps, nPoints, w, h, new_w, new_h):

    # Init infer threads
    threads = []
    for devid in range(number_of_ncs):
        thworker = threading.Thread(target=async_infer, args=(NcsWorker(devid, device, model_xml, frameBuffer, results, camera_width, camera_height, number_of_ncs, vidfps, nPoints, w, h, new_w, new_h),))
        thworker.start()
        threads.append(thworker)

    for th in threads:
        th.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, MYRIAD is acceptable. (Default=CPU)", default="CPU", type=str)
    parser.add_argument('-numncs','--numberofncs',dest='number_of_ncs',type=int,default=1,help='Number of NCS. (Default=1)')
    parser.add_argument("-b", "--boost", help="Setting it to True will make it run faster instead of sacrificing accuracy. (Default=False)", default=False, type=bool)
    args = parser.parse_args()

    device = args.device

    if "CPU" == device:
        number_of_ncs = 1
        if args.boost == False:
            model_xml = "models/train/test/openvino/mobilenet_v2_1.4_224/FP32/frozen-model.xml"
        else:
            model_xml = "models/train/test/openvino/mobilenet_v2_0.5_224/FP32/frozen-model.xml"

    elif "MYRIAD" == device:
        number_of_ncs = args.number_of_ncs
        if args.boost == False:
            model_xml = "models/train/test/openvino/mobilenet_v2_1.4_224/FP16/frozen-model.xml"
        else:
            model_xml = "models/train/test/openvino/mobilenet_v2_0.5_224/FP16/frozen-model.xml"

    elif "GPU" == device:
        number_of_ncs = 1
        if args.boost == False:
            model_xml = "models/train/test/openvino/mobilenet_v2_1.4_224/FP16/frozen-model.xml"
        else:
            model_xml = "models/train/test/openvino/mobilenet_v2_0.5_224/FP16/frozen-model.xml"

    else:
        print("Specify the target device to infer on; CPU, GPU, MYRIAD is acceptable.")
        sys.exit(0)

    camera_width = 320
    camera_height = 240
    vidfps = 30
    nPoints = 18
    w = 432 # Network size (Width)
    h = 368 # Network size (Height)
    new_w = int(camera_width * min(w/camera_width, h/camera_height))
    new_h = int(camera_height * min(w/camera_width, h/camera_height))

    try:

        mp.set_start_method('forkserver')
        frameBuffer = mp.Queue(4)
        results = mp.Queue()

        # Start detection MultiStick
        # Activation of inferencer
        p = mp.Process(target=inferencer, args=(device, model_xml, results, frameBuffer, number_of_ncs, camera_width, camera_height, vidfps, nPoints, w, h, new_w, new_h), daemon=True)
        p.start()
        processes.append(p)

        if device == "MYRIAD":
            sleep(number_of_ncs * 7)

        # Start streaming
        p = mp.Process(target=camThread, args=(results, frameBuffer, camera_width, camera_height, vidfps, nPoints, w, h, new_w, new_h), daemon=True)
        p.start()
        processes.append(p)

        while True:
            sleep(1)

    except:
        import traceback
        traceback.print_exc()
    finally:
        for p in range(len(processes)):
            processes[p].terminate()

        print("\n\nFinished\n\n")

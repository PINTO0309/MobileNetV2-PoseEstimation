import sys
import argparse
import numpy as np
import cv2
import time
#from edgetpu.detection.engine import DetectionEngine
from edgetpu.basic.basic_engine import BasicEngine


keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2,17], [5,16]]
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]
colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255], [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255], [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


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


def getValidPairs(outputs, w, h, detected_keypoints):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/train/test/tpu/mobilenet_v2_1.4_224/output_tflite_graph_edgetpu.tflite", help="Path of the inference model.")
    parser.add_argument("--usbcamno", type=int, default=0, help="USB Camera number.")
    parser.add_argument("--usbcamfps", type=int, default=30, help="USB Camera FPS.")
    args = parser.parse_args()

    camera_width = 320
    camera_height = 240

    fps = ""
    framecount = 0
    time1 = 0
    elapsedTime = 0

    h = 368
    w = 432

    new_w = int(camera_width * min(w/camera_width, h/camera_height))
    new_h = int(camera_height * min(w/camera_width, h/camera_height))

    threshold = 0.1
    nPoints = 18

    cap = cv2.VideoCapture(args.usbcamno)
    cap.set(cv2.CAP_PROP_FPS, args.usbcamfps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE)

    # Initialize engine.
    engine = BasicEngine(args.model)

    # Run inference.
    while True:
        t1 = time.perf_counter()

        ret, color_image = cap.read()
        if not ret:
            break

        resized_image = cv2.resize(color_image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
        prepimg = resized_image[:, :, ::-1].copy()
        canvas = canvas2 = np.full((h, w, 3), 128)
        canvas[(h - new_h)//2:(h - new_h)//2 + new_h,(w - new_w)//2:(w - new_w)//2 + new_w, :] = prepimg
        canvas2[(h - new_h)//2:(h - new_h)//2 + new_h,(w - new_w)//2:(w - new_w)//2 + new_w, :] = resized_image

        prepimg = np.uint8(canvas).flatten()
        #prepimg = canvas.flatten()

        #tinf = time.perf_counter()
        #ans = engine.DetectWithImage(prepimg, threshold=0.5, keep_aspect_ratio=True, relative_coord=False, top_k=10)

        #print("engine.required_input_array_size()=", engine.required_input_array_size()) #476928
        #print("prepimg.flatten()=", len(prepimg)) #476928
        ans = engine.RunInference(prepimg)
        #print("len(ans)=", len(ans)) #2
        #print("ans[0]=", ans[0]) #3.071000099182129
        print("ans[1]=", ans[1]) #[0.04705882 0.09411765 0.04705882 ... 0.07058824 0.23529412 0.        ]
        print("len(ans[1])=", len(ans[1])) #141588=1x46x54x57 or 1x57x46x54

        outputs = ans[1].reshape((1, 46, 54, 57)).transpose((0, 3, 1, 2)) #(1, 57, 46, 54)
        #outputs = outputs[np.newaxis, :, :, :]
        #outputs = ans[1].reshape((1, 57, 46, 54)) #(1, 57, 46, 54)

        #print(time.perf_counter() - tinf, "sec")
        #sys.exit(0)

        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        #print("outputs.shape()=", outputs.shape)
        #print("outputs.shape(outputs[0, 0, :, :])=", outputs[0, 0, :, :])

        for part in range(nPoints):
            probMap = outputs[0, part, :, :]
            probMap = cv2.resize(probMap, (w, h)) # (432, 368)
            keypoints = getKeypoints(probMap, threshold)
            keypoints_with_id = []

            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)

        #print("len(detected_keypoints)=", len(detected_keypoints))

        frameClone = np.uint8(canvas2.copy())
        for i in range(nPoints):
            #print("detected_keypoints[i]=", detected_keypoints[i])
            for j in range(len(detected_keypoints[i])):

                cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

        valid_pairs, invalid_pairs = getValidPairs(outputs, w, h, detected_keypoints)
        #print("valid_pairs, invalid_pairs=", valid_pairs, invalid_pairs)
        personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

        print("personwiseKeypoints=", personwiseKeypoints)

        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

        cv2.putText(frameClone, fps, (w-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)

        cv2.imshow("USB Camera" , frameClone)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        # FPS calculation
        framecount += 1
        if framecount >= 5:
            fps = "(Playback) {:.1f} FPS".format(time1/15)
            framecount = 0
            time1 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

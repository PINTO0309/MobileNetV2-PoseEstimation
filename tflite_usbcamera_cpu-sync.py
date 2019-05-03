import cv2, sys, time
import numpy as np
import tensorflow as tf
from PIL import Image

width  = 320
height = 240
fps = ""
elapsedTime = 0
index_void = 2
num_threads = 4

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cv2.namedWindow('FPS', cv2.WINDOW_AUTOSIZE)

#input_details  = [{'shape': array([ 1, 368, 432, 3], dtype=int32), 'quantization': (0.0078125, 128), 'index': 493, 'dtype': <class 'numpy.uint8'>, 'name': 'image'}]
#output_details = [{'shape': array([ 1, 46, 54, 57], dtype=int32), 'quantization': (0.0235294122248888, 0), 'index': 490, 'dtype': <class 'numpy.uint8'>, 'name': 'Openpose/concat_stage7'}]

interpreter = tf.lite.Interpreter(model_path="models/train/test/tflite/mobilenet_v2_1.4_224/output_tflite_graph.tflite")
interpreter.allocate_tensors()
interpreter.set_num_threads(int(num_threads))
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

h = input_details[0]['shape'][1] #368
w = input_details[0]['shape'][2] #432

while True:
    t1 = time.time()
    ret, color_image = cap.read()
    if not ret:
        break

    # Resize image
    colw = color_image.shape[1]
    colh = color_image.shape[0]
    new_w = int(colw * min(w/colw, h/colh))
    new_h = int(colh * min(w/colw, h/colh))
    resized_image = cv2.resize(color_image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.full((h, w, 3), 128)
    canvas[(h - new_h)//2:(h - new_h)//2 + new_h,(w - new_w)//2:(w - new_w)//2 + new_w, :] = resized_image
    prepimg = canvas
    prepimg = prepimg[np.newaxis, :, :, :] # Batch size axis add

    # Estimation
    interpreter.set_tensor(input_details[0]['index'], np.array(prepimg, dtype=np.uint8))
    st = time.perf_counter()
    interpreter.invoke()
    print("elapsedtime=", time.perf_counter()-st)
    outputs = interpreter.get_tensor(output_details[0]['index'])

    sys.exit(0)

    # View
    output = outputs[0]
    res = np.argmax(output, axis=2)
    if index_void is not None:
        res = np.where(res == index_void, 0, res)
    image = Image.fromarray(np.uint8(res), mode="P")
    image.putpalette(palette)
    image = image.convert("RGB")
    image = image.resize((width, height))
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(frame, 1, image, 0.9, 0)

    cv2.putText(image, fps, (width-180,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
    cv2.imshow('FPS', image)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    elapsedTime = time.time() - t1
    fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)
    print("fps = ", str(fps))

cap.release()
cv2.destroyAllWindows()


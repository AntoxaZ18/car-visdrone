import sys
from time import time
import numpy as np
import onnxruntime as ort
import cv2
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from trackers import BYTETracker
from argparse import Namespace


def get_providers():
    providers = [i for i in ort.get_available_providers() if any(val in i for val in ('CUDA', 'CPU'))]
    print(ort.get_available_providers())
    modes = {
        'CUDAExecutionProvider': 'gpu',
        'CPUExecutionProvider': 'cpu'
    }
    providers = [modes.get(i) for i in providers]
    return providers



class Result():
    conf = None
    xywh = None
    cls = None
class YoloONNX():
    def __init__(self, path: str, session_options=None, device='cpu', batch=1, confidence=0.5) -> None:

        sess_options = ort.SessionOptions()

        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # # sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

        sess_options.execution_mode  = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.inter_op_num_threads = 3

        sess_providers = ['CPUExecutionProvider']
        if device == 'gpu':
            sess_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.mode = 'gpu'
        else:
            self.mode = 'cpu'
        
        self.session = ort.InferenceSession(path, providers=sess_providers, sess_options=sess_options)    #'CUDAExecutionProvider',
        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape

        self.input_width = 640
        self.input_height = 640
        self.batch = batch

        self.iou = 0.8
        self.confidence_thres = confidence
        self.input_size = (640, 640)
        self.classes = ['cars', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.executor = ThreadPoolExecutor(max_workers=self.batch)

        self.tracker = BYTETracker(Namespace(track_buffer=30, track_high_thresh=0.6, track_low_thresh=0.2, fuse_score=0.6, match_thresh=0.8, new_track_thresh=0.5), frame_rate=10)

    def _image_batch_preprocess(self, image_batch: List):
        imgs = np.stack([self.letterbox(img)[0] for img in image_batch])

        imgs = imgs[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        imgs = np.ascontiguousarray(imgs)  # contiguous
        imgs = imgs.astype(float) / 255.0

        return imgs

    def _image_preprocess(self, rgb_frame) -> np.ndarray:
        """image preprocessing
        rgb_frame - image in rgb format
        including resizing to yolo input shape
        add batch dimension and normalized to 0...1 range
        convert from image to tensor view
        # """
        self.img_height, self.img_width = rgb_frame.shape[:2]

        # image_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        image_data, pad = self.letterbox(rgb_frame, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(image_data) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data, pad
    
    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int, track_id: int=0) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (List[float]): Detected bounding box coordinates [x, y, width, height].
            score (float): Confidence score of the detection.
            class_id (int): Class ID for the detected object.
        """
        # Extract the coordinates of the bounding box
        box = [int(i) for i in box]
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f} id:{track_id}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


    def postprocess(self, output: List[np.ndarray], pad: Tuple[int, int]) -> np.ndarray:
        """
        Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs.
        It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            output (List[np.ndarray]): The output arrays from the model.
            pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.
        """
        # Transpose and squeeze the output to match the expected shape

        outputs = np.transpose(np.squeeze(output[0]))

        # Calculate the scaling factors for the bounding box coordinates
        input_height = self.input_size[0]
        input_width = self.input_size[0]

        gain = np.float32(min(input_height / self.img_height, input_width / self.img_width))
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        #find max score for prediction
        max_scores = np.amax(outputs[:, 4:], axis=1)
        #filter prediction with result more than confidence thresh
        results = outputs[max_scores > self.confidence_thres]
        #find class ids
        class_ids = np.argmax(results[:, 4:], axis=1).tolist()
        scores = np.amax(results[:, 4:], axis=1).tolist()
        
        #convert coordinates of boxes
        x = results[:, 0]
        y = results[:, 1]
        w = results[:, 2]
        h = results[:, 3]

        left = ((x - w / 2) / gain).astype(int)
        top = ((y - h / 2) / gain).astype(int)
        width = (w / gain).astype(int)
        height = (h / gain).astype(int)

        boxes = np.column_stack((left, top, width, height)).tolist()

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou)

        predictions = []

        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            predictions.append([*box, score, class_id])

        return predictions



    def __call__(self, images: np.ndarray) -> np.ndarray:
        """return image of object if they are on image. Return only one object with highest score"""
        if self.mode == 'gpu':
            res = self.call_gpu(images)
            return res
        else:
            return self.call_cpu(images)
    
    def call_gpu(self, images:List[np.ndarray]):

        # futures = [self.executor.submit(self._image_preprocess, image) for image in images]
        # wait(futures, return_when=ALL_COMPLETED)
        # tensor_images = [f.result() for f in futures]

        tensor_images = [self._image_preprocess(image) for image in images]

        output_name = self.session.get_outputs()[0].name
        input_name = self.session.get_inputs()[0].name

        imgs = [img for (img, _) in tensor_images]
        pads = [pad for (_, pad) in tensor_images]


        batch = np.concatenate(imgs, axis=0)
        
        outputs = self.session.run([output_name], {input_name: batch})
        
        predictions = outputs[0]

        results = [self.postprocess(np.expand_dims(predictions[idx], axis=0), pad) for image, idx, pad in zip(images, iter(range(predictions.shape[0])), pads)]

        return results


    def call_cpu(self, images:List[np.ndarray]):


        futures = [self.executor.submit(self._image_preprocess, image) for image in images]

        # start = time()
        # self._image_batch_preprocess(images)
        # print(time() - start)

        wait(futures, return_when=ALL_COMPLETED)
        tensor_images = [f.result() for f in futures]
    
        output_name = self.session.get_outputs()[0].name
        input_name = self.session.get_inputs()[0].name

        futures = [self.executor.submit(self.session.run, [output_name], {input_name: image}) for image, _ in tensor_images]

        wait(futures, return_when=ALL_COMPLETED)

        model_outputs = [f.result() for f in futures]
        pads = [pad for (img, pad) in tensor_images]


        predictions = [self.postprocess(output, pad) for output, pad in zip(model_outputs, pads)]

        return self.draw(images, predictions)



    def draw(self, images:List[np.ndarray], predictions):
        
        out_images = []

        for img, img_preds in zip(images, predictions):

            dets = np.array(img_preds)

            x = Result()
            x.conf = dets[:, 4]
            x.xywh = dets[:, 0:4]
            x.cls = dets[:, 5]

            online_targets = self.tracker.update(x, (self.img_height, self.img_width))
            if len(online_targets):
                x1 = online_targets[:, 0]
                y1 = online_targets[:, 1]
                x2 = online_targets[:, 2]
                y2 = online_targets[:, 3]

                targets = np.column_stack([(x1+x2) / 2, (y1 + y2) / 2, x2-x1, y2-y1, online_targets[:, 5], online_targets[:, 6], online_targets[:, 4]]).tolist()

                for box in targets:
                    self.draw_detections(img, [box[0], box[1], box[2], box[3]], box[4], int(box[5]), int(box[6]))

            out_images.append(img)

        return images
    

        #     for img, img_preds in zip(images, predictions):
        #     for box in img_preds:
        #         byte_pred = [*box[0], box[1], box[2]]
        #         # online_targets = tracker.update(byte_pred, [self.img_height, self.img_width], [640, 640])
        #         self.draw_detections(img, *box)
        #     out_images.append(img)

        # return images




if __name__ == '__main__':



    batch_images = 1

    nano = 'yolo11n_5epoch_16batch640.onnx'
    small = 'y11_100ep16b640.onnx'

    model = YoloONNX(small, device='cpu', batch = batch_images)

    # cap = cv2.VideoCapture('VID_20250412_121719.mp4')

    # # for _ in range(1000):
    # #     ret, frame = cap.read()
    # #     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
    # #     images = [image_rgb] 
    # #     process_imgs = model(images)

    # img = cv2.imread('test.jpg')
    # image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # model(images)

    # sys.exit(0)





    frame = cv2.imread('test.jpg')
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    images = [image_rgb] 
    process_imgs = model(images)

    print('show')
    cv2.startWindowThread()

    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('frame', process_imgs[0])
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    # cv2.imwrite('out.jpg', process_imgs[0])
    # cv2.imshow('0', process_imgs[0])

    # sys.exit(0)



    # print('load ok')
    # def warmup(model, images, iterations=20):
    #     for i in range(iterations):
    #         _ = model(images)
    #     print('warmup ok')


    # def bench(image):
    #     start = time()
    #     range_iter = 50
    #     images = [image] * batch_images
    #     for _ in range(range_iter):
    #         frame_boxes = model(images)
    #     print(f'FPS: {1 / ((time() - start) / (range_iter * batch_images)):.3f}')


    # warmup(model, images)


    # print(len(results))
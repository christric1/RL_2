import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as TF
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, box_iou, scale_boxs
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.metrics import ap_per_class


class yolo:
    """
        I want to graduate...
    """
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = 'yolov7.pt'

        # iou threshold
        self.iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel() 

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(640, s=self.stride)  # check img_size

    def detectImg(self, image: Tensor, labels: Tensor) -> str:
        """
        Args
            image  : [1, height, width]
            targets : [label, xmin, ymin, xmax, ymax]

        Return 
            avg_iou : float 
            F1_score : float
        """
        # Resize and pad image while meeting stride-multiple constraints
        img_ = image.permute(1, 2, 0).numpy()
        img_ = (img_[:, :, ::-1] * 255).astype(np.uint8)    # to BGR
        Padded_image = letterbox(img_, self.imgsz, stride=self.stride)[0]
        Padded_image = TF.to_tensor(Padded_image)[[2, 1, 0], :, :].unsqueeze(dim=0).to(self.device)     # to tensor, to RGB, add dim=0

        # Get the prediction
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            out = self.model(Padded_image)[0]

        # Apply NMS
        NMS_out = non_max_suppression(out, conf_thres=0.25, iou_thres=0.45)

        # Process detections
        stats = []
        label_iou = []
        for si, pred in enumerate(NMS_out):
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            if len(pred) == 0:
                stats.append((torch.zeros(0, self.niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions : [xmin, ymin, xmax, ymax, obj, cls]
            predn = pred.clone()
            scale_coords(Padded_image.shape[2:], predn[:, :4], image.shape[1:])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool, device=self.device)
            detected = []  # target indices
            tcls_tensor = labels[:, 0].to(self.device)

            # target boxes
            tbox = labels[:, 1:].to(self.device)
            scale_boxs(tbox, image.shape[1:])

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                pi = (cls == predn[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                    label_iou.append(ious.cpu())

                    # Append detections
                    detected_set = set()
                    for j in (ious > self.iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > self.iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

            # Append statistics (correct, conf, pcls, tcls)           
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            
        # Evaluation metric
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        avg_iou = torch.cat(label_iou).numpy().mean() if len(label_iou) > 0 else 0

        return avg_iou, f1.mean()
        

def detectImg(image: Tensor, labels: Tensor) -> str:
    """
    Args
        image  : [1, height, width]
        targets : [label, xmin, ymin, xmax, ymax]

    Return 
        avg_iou : float 
        F1_score : float
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = 'yolov7.pt'

    # iou threshold
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel() 

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(640, s=stride)  # check img_size
    
    # Resize and pad image while meeting stride-multiple constraints
    img_ = image.permute(1, 2, 0).numpy()
    img_ = (img_[:, :, ::-1] * 255).astype(np.uint8)    # to BGR
    Padded_image = letterbox(img_, imgsz, stride=stride)[0]
    Padded_image = TF.to_tensor(Padded_image)[[2, 1, 0], :, :].unsqueeze(dim=0).to(device)     # to tensor, to RGB, add dim=0

    # Get the prediction
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        out = model(Padded_image)[0]
    
    # Apply NMS
    NMS_out = non_max_suppression(out, conf_thres=0.25, iou_thres=0.45)

    # Process detections
    for si, pred in enumerate(NMS_out):
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        # if len(pred) == 0:
        #     if nl:
        #         stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))

        # Predictions : [xmin, ymin, xmax, ymax, obj, cls]
        predn = pred.clone()
        scale_coords(Padded_image.shape[2:], predn[:, :4], image.shape[1:])  # native-space pred

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0].to(device)

            # target boxes
            tbox = labels[:, 1:].to(device)
            scale_boxs(tbox, image.shape[1:])

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                pi = (cls == predn[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break
            
            # Evaluation metric
            stats = [(correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)]
            stats = [np.concatenate(x, 0) for x in zip(*stats)]
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            print(f1)

    # return avgRecall, avgPrecision

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images/bus.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', default='True', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

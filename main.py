import gradio as gr
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import os

import sys
sys.path.append('lama-with-refiner')
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.trainers import load_checkpoint

def get_inpaint_model():
    predict_config = OmegaConf.load('lama-with-refiner/configs/prediction/default.yaml')
    predict_config.model.path = 'big-lama'
    predict_config.refiner.gpu_ids = '0'

    device = torch.device(predict_config.device)
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')

    train_config = OmegaConf.load(train_config_path)
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(predict_config.model.path, 
                                   'models', 
                                   predict_config.model.checkpoint)

    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    return model,predict_config

model = YOLO("yolov8x-seg.pt")
inpaint_model,predict_config = get_inpaint_model()

import gradio as gr
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import os
from scipy.ndimage import gaussian_filter

import sys
sys.path.append('lama-with-refiner')
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.trainers import load_checkpoint

id2objects = {}
segment_masks = None

def process_classes(results):
    global segment_masks
    global id2objects
    segment_masks = None
    segment_masks = results[0].masks.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    conf = results[0].boxes.conf.cpu().numpy()
    dec = results[0].names
    id2objects = {}
    
    for i in range(len(classes)):
        id2objects[i] = dec[classes[i]], round(conf[i], 2)
    s = ""
    for k, v in id2objects.items():
        s += f"{k}: {v[0]} {str(v[1])}\n"
    return s 

def get_mask(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_img.jpg", img)
    
    results = model.predict(img)

    img = results[0].plot({'line_width': False, 'boxes': False, 'conf': True, 'labels': True})
    return img, process_classes(results)

def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def remove(img, s):
    with torch.no_grad():
        ids = [int(x) for x in s.split(",") if x]

        
        img = image_resize(img, height = 720)
        img = img.astype('float32') / 255
        img = np.transpose(img, (2, 0, 1))

        print(np.unique(segment_masks))
        temp_masks = segment_masks[ids]
        print(temp_masks.shape)
        masks = np.zeros((img.shape[1], img.shape[2]), dtype = bool)
        for mask in temp_masks:
            print(mask.shape)
            mask = cv2.resize(mask, (img.shape[2], img.shape[1]), interpolation = cv2.INTER_NEAREST).astype(bool)
            masks = masks | mask
            
        kernel = np.ones((7, 7), np.uint8)
        masks = masks.astype(np.uint8) * 255
        masks = cv2.GaussianBlur(masks, (7,7), 0)
        masks = cv2.dilate(masks, kernel, iterations = 3)

        batch = dict(image=img, mask=masks[None, ...])

        batch['unpad_to_size'] = [torch.tensor([batch['image'].shape[1]]),torch.tensor([batch['image'].shape[2]])]
        batch['image'] = torch.tensor(pad_img_to_modulo(batch['image'], predict_config.dataset.pad_out_to_modulo))[None].to(predict_config.device)
        batch['mask'] = torch.tensor(pad_img_to_modulo(batch['mask'], predict_config.dataset.pad_out_to_modulo))[None].float().to(predict_config.device)

        cur_res = refine_predict(batch, inpaint_model, **predict_config.refiner)
        cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')

    return cur_res, masks

if __name__ == "__main__":
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="input image")
                run_btn = gr.Button(variant = "primary")
            output_mask = gr.Image(label="masks", image_mode = "RGB")
        with gr.Row():
            with gr.Column():
                list_of_objects = gr.Textbox(label="Objects id")
                objects_to_remove = gr.Textbox(label="Objects id to remove")
                remove_button = gr.Button("remove")
            with gr.Column():
                output_img = gr.Image(label = "result")
                black_white = gr.Image(label = "segment mask")
        run_btn.click(get_mask, [input_img], [output_mask, list_of_objects])
        remove_button.click(remove, [input_img,objects_to_remove], [output_img, black_white])
    print("sdfdsf")
    app.launch(share=True)

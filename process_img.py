import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from utils.normal_utils import norm_normalize, normal2img, camNormal2worldNormal


class BackgroundRemoval():
    def __init__(self, device='cuda'):

        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)

        image = self.interface([image])[0]
        image = np.array(image)

        return image


class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                                   torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


class DPT():
    def __init__(self, task='depth', device='cuda'):

        self.task = task
        self.device = device

        from utils.dpt import DPTDepthModel

        if task == 'depth':
            path = 'pretrained_models/omnidata_dpt_depth_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384')
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])

        else:  # normal
            path = './pretrained_models/omnidata_dpt_normal_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor()
            ])

        # load model
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

    @torch.no_grad()
    def __call__(self, image):
        # image: np.ndarray, uint8, [H, W, 3]
        H, W = image.shape[:2]
        image = Image.fromarray(image)

        image = self.aug(image).unsqueeze(0).to(self.device)

        if self.task == 'depth':
            depth = self.model(image).clamp(0, 1)
            depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False)
            depth = depth.squeeze(1).cpu().numpy()
            return depth
        else:
            normal = self.model(image).clamp(0, 1)
            normal = F.interpolate(normal, size=(H, W), mode='bicubic', align_corners=False)
            normal = normal.cpu().numpy()
            return normal


def preprocess_single_image(img_path, args):
    out_dir = os.path.dirname(args.output_path)
    out_rgba = os.path.join(out_dir, os.path.basename(img_path).split('.')[0] + '.png')
    out_depth = os.path.join(out_dir, os.path.basename(img_path).split('.')[0] + '_depth.png')
    out_normal = os.path.join(out_dir, 'normal', os.path.basename(img_path).split('.')[0] + '_normal.png')
    out_caption = os.path.join(out_dir, os.path.basename(img_path).split('.')[0] + '_caption.txt')
    if not os.path.exists(os.path.join(out_dir, 'normal')):
        os.makedirs(os.path.join(out_dir, 'normal'))

    # load image
    print(f'[INFO] loading image {img_path}...')

    # check the exisiting files
    if os.path.isfile(out_rgba) and os.path.isfile(out_depth) and os.path.isfile(out_normal):
        print(f"{img_path} has already been here!")
        return
    print(img_path)
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    carved_image = None

    if image.shape[-1] == 4:
        carved_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if carved_image is None:
        # carve background
        print(f'[INFO] background removal...')
        carved_image = BackgroundRemoval()(image)  # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # predict depth
    # print(f'[INFO] depth estimation...')
    # dpt_depth_model = DPT(task='depth')
    # depth = dpt_depth_model(image)[0]
    # depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
    # depth[~mask] = 0
    # depth = (depth * 255).astype(np.uint8)
    # del dpt_depth_model

    # predict normal
    print(f'[INFO] normal estimation...')
    dpt_normal_model = DPT(task='normal')
    normal = dpt_normal_model(image)[0].transpose(1, 2, 0)
    normal = norm_normalize(normal * 2 - 1)  # map to [-1,1]
    idx = 0
    file_name = f'./camera_poses/{idx}.txt'
    with open(file_name, 'r') as file:
        cam_pose = np.loadtxt(file_name)
    rot_c2w = cam_pose[:3, :3]
    axis_transform = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [-1,0, 0],
    ])

    # Apply the transformation
    c2w_my = np.dot(axis_transform, rot_c2w)
    normal= camNormal2worldNormal(c2w_my, normal)
    # def normal_blender2opencv(normal):
    #     H, W, C = np.shape(normal)  # Get the shape of the normal map
    #     # Create a new array for the converted normals
    #
    #     # R_bcam2cv = np.asarray([[0, 0, -1],
    #     #                       [0, 1, 0],
    #     #                       [-1,0, 0]], dtype=np.float32)
    #     R_bcam2cv = np.asarray([[0, 1, 0],
    #                             [0, 0, 1],
    #                             [1, 0, 0]], dtype=np.float32)
    #     normal_cv = normal @ R_bcam2cv[None, :]
    #     print(np.shape(normal_cv))
    #
    #     return normal_cv
    # normal = normal_blender2opencv(normal)

    normal= normal2img(normal)

    normal[~mask] = 255
    del dpt_normal_model

    # recenter
    if opt.recenter:
        print(f'[INFO] recenter...')
        final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)
        # final_depth = np.zeros((opt.size, opt.size), dtype=np.uint8)
        # final_normal = np.zeros((opt.size, opt.size, 3), dtype=np.uint8)
        final_normal = np.full((opt.size, opt.size, 3), 255, dtype=np.uint8)

        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        desired_size = int(opt.size * (1 - opt.border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (opt.size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (opt.size - w2) // 2
        y2_max = y2_min + w2
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2),
                                                              interpolation=cv2.INTER_AREA)
        # final_depth[x2_min:x2_max, y2_min:y2_max] = cv2.resize(depth[x_min:x_max, y_min:y_max], (w2, h2),
        #                                                       interpolation=cv2.INTER_AREA)
        final_normal[x2_min:x2_max, y2_min:y2_max] = cv2.resize(normal[x_min:x_max, y_min:y_max], (w2, h2),
                                                                interpolation=cv2.INTER_AREA)

    else:
        final_rgba = carved_image
        # final_depth = depth
        final_normal = normal

    # write output
    cv2.imwrite(out_rgba, cv2.cvtColor(final_rgba, cv2.COLOR_RGBA2BGRA))
    # cv2.imwrite(out_depth, final_depth)
    cv2.imwrite(out_normal, final_normal)

    if opt.do_caption:
        # predict caption (it's too slow... use your brain instead)
        print(f'[INFO] captioning...')
        blip2 = BLIP2()
        caption = blip2(image)
        with open(out_caption, 'w') as f:
            f.write(caption)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('output_path', type=str, help="path to output images")
    parser.add_argument('--size', default=256, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0., type=float, help="output border ratio")
    parser.add_argument('--recenter', action='store_true',
                        help="recenter, potentially not helpful for multiview zero123")
    parser.add_argument('--do_caption', action='store_true', help="do text captioning")

    opt = parser.parse_args()

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    if os.path.isdir(opt.path):
        img_list = sorted(os.path.join(root, fname) for root, _dirs, files in os.walk(opt.path) for fname in files)
        img_list = [img for img in img_list if
                    not img.endswith("rgba.png") and not img.endswith("depth.png") and not img.endswith("normal.png")]
        img_list = [img for img in img_list if img.endswith(".png")]
        for img in img_list:
            # try:
            preprocess_single_image(img, opt)
        # except:
        #     with open("preprocess_images_invalid.txt", "a") as f:
        #         print(img, file=f)
    else:  # single image file
        preprocess_single_image(opt.path, opt)
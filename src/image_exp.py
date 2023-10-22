import timm
import urllib
import torch
import os

import numpy as np

import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Saliency
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

import shap

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0")
model = timm.create_model("resnet34", pretrained=True).to(device).eval()
integrated_gradients = IntegratedGradients(model)
gradient_shap = GradientShap(model)
occlusion = Occlusion(model)
saliency = Saliency(model)

class ImageExplanability:
    def __init__(
            self, 
            imagenet_class_path: str = "data/imagenet_classes.txt",
            image_data_path: str = "data/raw"
            ) -> None:
        
        self.device = device

        # Model Loaded
        self.model = model

        # ImageNet Classes Loaded
        with open(os.path.join(os.getcwd(), imagenet_class_path), "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
        
        self.transform = T.Compose([
            T.ToTensor()
        ])

        self.transform_normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Image Lists
        self.image_lists = os.listdir(os.path.join(os.getcwd(), image_data_path))
        self.integrated_gradients = integrated_gradients
        self.gradient_shap = gradient_shap
        self.occlusion = occlusion
        self.saliency = saliency

        # clear all images from data/explained folder
        for image in os.listdir(os.path.join(os.getcwd(), f"data/explained/")):
            os.remove(os.path.join(os.getcwd(), f"data/explained/{image}"))
            print(f"Removed {image} from data/explained/")
        
        self.default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                    [(1, 1, 1),
                                                    (0, 0, 0)], N=256)
    
    def predict(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = torch.tensor(imgs)
        imgs = imgs.permute(0, 3, 1, 2)

        img_tensor = imgs.to(device)

        output = self.model(img_tensor)
        return output

    def get_shap_explanability(self, image_tensor: torch.Tensor) -> shap.Explanation:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        inv_transform= T.Compose([
            T.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            T.Normalize(
                mean = (-1 * np.array(mean) / np.array(std)).tolist(),
                std = (1 / np.array(std)).tolist()
            ),
            T.Lambda(lambda x: x.permute(0, 2, 3, 1)),
        ])
        topk = 10
        batch_size = 50
        n_evals = 10000    
        # define a masker that is used to mask out partitions of the input image.
        masker_blur = shap.maskers.Image("blur(128,128)", (224, 224, 3))

        # create an explainer with model and image masker
        explainer = shap.Explainer(self.predict, masker_blur, output_names=self.categories)

        img_tensor = image_tensor.permute(0, 2, 3, 1)

        shap_values = explainer(img_tensor, max_evals=n_evals, batch_size=batch_size,
                                outputs=shap.Explanation.argsort.flip[:topk])
        shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]
        shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

        return shap_values

    def get_sal_explanability(self, image_tensor: torch.Tensor, target_idx: int, image_name: str) -> None:
        image_tensor.requires_grad = True
        grads = saliency.attribute(image_tensor, target=target_idx)
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
        original_image = np.transpose((image_tensor.squeeze(0).cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
        _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                                show_colorbar=True, title="Overlayed Gradient Magnitudes")
        # save saliency image
        plt.savefig(os.path.join(os.getcwd(), f"data/explained/{image_name.split('.')[0]}_sal.png"))

        image_tensor.requires_grad = False

    def get_gradcam_explanability(self, image_tensor: torch.Tensor, target_idx: int, image_name: str) -> None:
        image_tensor.requires_grad = True

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        inv_transform = T.Compose([
            # T.Lambda(lambda x: x.permute(0, 2, 3, 1)),
            T.Normalize(
                mean = (-1 * np.array(mean) / np.array(std)).tolist(),
                std = (1 / np.array(std)).tolist()
            ),
            # T.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        ])

        target_layers = [self.model.layer4[-1]]

        cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=True)

        targets = [ClassifierOutputTarget(target_idx)]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        rgb_img = inv_transform(image_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
        visualization = Image.fromarray(show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True))

        # save gradcam image
        visualization.save(os.path.join(os.getcwd(), f"data/explained/{image_name.split('.')[0]}_gradcam.png"))

        campp = GradCAMPlusPlus(model=self.model, target_layers=target_layers, use_cuda=True)

        grayscale_campp = campp(input_tensor=image_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_campp = grayscale_campp[0, :]
        rgb_img = inv_transform(image_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
        visualization_gcpp = Image.fromarray(show_cam_on_image(rgb_img, grayscale_campp, use_rgb=True))

        # save gradcam image
        visualization_gcpp.save(os.path.join(os.getcwd(), f"data/explained/{image_name.split('.')[0]}_gradcampp.png"))

        image_tensor.requires_grad = False

    def save_explanability_img(self, image_name: str) -> None:

        img = Image.open(os.path.join(os.getcwd(), f'data/raw/{image_name}'))

        transformed_img = self.transform(img)

        img_tensor = self.transform_normalize(transformed_img).unsqueeze(0).to(self.device)

        # self.model.eval()

        with torch.no_grad():
            output = self.model(img_tensor)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)

        pred_label_idx.squeeze_()
        predicted_label = self.categories[pred_label_idx.item()]
        print('Predicted:', predicted_label, ' : ', pred_label_idx.item(), 'Prob: (', prediction_score.squeeze().item(), ')')
        
        # Integrated Gradients calculation for the image
        attributions_ig = self.integrated_gradients.attribute(img_tensor, target=pred_label_idx.item(), n_steps=100)

        plot = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                        method='heat_map',
                                        # cmap=self.default_cmap,
                                        show_colorbar=True,
                                        sign='positive',
                                        outlier_perc=1)
        
        # Saving Ingtegrated Gradients for the image
        plot[0].savefig(os.path.join(os.getcwd(), f"data/explained/{image_name.split('.')[0]}_ig.png"))

        # Integrated Gradients with Noise Tunnel calculation for the image
        noise_tunnel = NoiseTunnel(self.integrated_gradients)

        attributions_ig_nt = noise_tunnel.attribute(img_tensor, nt_samples=3, nt_type='smoothgrad_sq', target=pred_label_idx.item())
        plot = viz.visualize_image_attr(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                        method='heat_map',
                                        # cmap=self.default_cmap,
                                        show_colorbar=True,
                                        sign='positive',
                                        outlier_perc=1)
        
        # Saving Ingtegrated Gradients with Noise Tunnel for the image
        plot[0].savefig(os.path.join(os.getcwd(), f"data/explained/{image_name.split('.')[0]}_ig_nt.png"))

        # Gradient Shap calculation for the image
        rand_img_dist = torch.cat([img_tensor * 0, img_tensor * 1])

        attributions_gs = self.gradient_shap.attribute(img_tensor,
                                                n_samples=50,
                                                stdevs=0.0001,
                                                baselines=rand_img_dist,
                                                target=pred_label_idx)
        plot = viz.visualize_image_attr(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        method='heat_map',
                                        # cmap=self.default_cmap,
                                        show_colorbar=True,
                                        sign='positive',
                                        outlier_perc=1)
        
        # Saving Gradient Shap for the image
        plot[0].savefig(os.path.join(os.getcwd(), f"data/explained/{image_name.split('.')[0]}_gshap.png"))

        # Occlusion calculation for the image
        attributions_occ = self.occlusion.attribute(img_tensor,
                                            strides = (3, 3, 3),
                                            target=pred_label_idx,
                                            sliding_window_shapes=(3, 15, 15),
                                            baselines=0)
        plot = viz.visualize_image_attr(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        method='heat_map',
                                        # cmap=self.default_cmap,
                                        show_colorbar=True,
                                        sign='positive',
                                        outlier_perc=1)
        
        # Saving Occlusion for the image
        plot[0].savefig(os.path.join(os.getcwd(), f"data/explained/{image_name.split('.')[0]}_occlusion.png"))

        # Shap calculation for the image
        shap_values = self.get_shap_explanability(img_tensor)

        # Saving Shap for the image
        shap.image_plot(shap_values=shap_values.values,
                        pixel_values=shap_values.data,
                        labels=shap_values.output_names,
                        true_labels=[self.categories[pred_label_idx]],
                        show=False)

        plt.savefig(os.path.join(os.getcwd(), f"data/explained/{image_name.split('.')[0]}_shap.png"))

        # Saliency calculation for the image
        self.get_sal_explanability(img_tensor, pred_label_idx.item(), image_name)

        #  GradCam and GradCAmPlusPlus calculation for the image
        self.get_gradcam_explanability(img_tensor, pred_label_idx.item(), image_name)

    def run_explanability_img(self) -> None:
        for image_name in self.image_lists:
            self.save_explanability_img(image_name)

    




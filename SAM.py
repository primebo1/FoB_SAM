# Code modified from github repository: segment-anything,
import torch.nn as nn
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

    
class SAM(nn.Module):
    def __init__(self, sam_pretrained_path=".../sam_vit_h_4b8939.pth"):
        super().__init__()

        self.get_sam(sam_pretrained_path)
         
    def get_sam(self, checkpoint_path):
        model_type="vit_h" 
        print(f"Using model type {model_type}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).eval().cuda()
        self.predictor = SamPredictor(self.sam)
        self.sam.requires_grad_(False) # freeze the model

        
    
    def predict_w_points_bbox(self, sam_input_points, bboxes, sam_neg_input_points, qry_img, config=None, return_logits=False):
        """
        Predicts masks and scores for given input points and bounding boxes.
        Args:
            sam_input_points (list): List of input points.
            bboxes (list): List of bounding boxes.
            sam_neg_input_points (list): List of negative input points.
            qry_img (numpy.ndarray): Query image.
            return_logits (bool, optional): Whether to return logits. Defaults to False.
        Returns:
            tuple: A tuple containing masks and scores.
        """
        masks, scores = [], []
        self.predictor.set_image(qry_img)

        bbox_xyxy = None
        all_points = []
        all_labels = []

        for point in sam_input_points: # positive points
            assert qry_img.max() <= 255 and qry_img.min() >= 0 and qry_img.dtype == np.uint8
            if point is not None:
                all_points.append(point)
                all_labels.extend([1] * len(point)) 


        if sam_neg_input_points is not None: # negative points
            for neg_point in sam_neg_input_points:
                if neg_point is not None:
                    all_points.append(neg_point)
                    all_labels.extend([0] * len(neg_point))  


        if all_points:
            points = np.vstack(all_points)
            point_labels = np.array(all_labels)
        else:
            points = None
            point_labels = None



        mask, score, logits = self.predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box = bbox_xyxy if bbox_xyxy is not None else None,
            return_logits=return_logits,
            multimask_output=True
        )

        if config['dataset'] == 'isic':
            best_pred_idx = 1   # please change this to 1 when testing on Skin-DS
        else:
            best_pred_idx = 0
        masks.append(mask[best_pred_idx])
        scores.append(score[best_pred_idx])
    

        return masks, scores

    def pre_process(self, image):
        """
        Pre-processes the given image.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            numpy.ndarray: The pre-processed image as a NumPy array.
        """
        image = image.permute(1, 2, 0).cpu().numpy()
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        return image
    
    def forward(self, query_image, pos_point=None, neg_point=None, config=None, return_logits=False):
        """
        Applies forward pass on the given query image and returns the result.

        Args:
            query_image (numpy.ndarray): The input query image.
            pos_point (tuple, optional): The positive point. 
            neg_point (tuple, optional): The negative point. 
            return_logits (bool, optional): Whether to return logits. 

        Returns:
            numpy.ndarray: The result of the forward pass.
        """
        self.image_size = query_image.shape[:2]
        query_image = self.pre_process(query_image)
        mask, score = self.predict_w_points_bbox(pos_point, None, neg_point, query_image, config, return_logits=return_logits)
        result = mask[0]

        return result
    
    

        
        

import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms
from stft_core.modeling.utils import cat
from stft_core.structures.bounding_box import BoxList
from stft_core.structures.boxlist_ops import remove_small_boxes


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep

def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)
    return tensor


class STFTFCOSPostProcessor(torch.nn.Module):
    """
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        stft_bbox_std,
    ):
        super(STFTFCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.stft_bbox_std = stft_bbox_std

    def forward_for_single_image(self, shifts_per_image, box_cls, box_center, stft_box_cls, stft_box_delta, stft_based_box, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        stft_bbox_std = stft_based_box[0].new_tensor(self.stft_bbox_std)

        # Iterate over every feature level
        for shifts_i, box_cls_i, box_ctr_i, stft_box_cls_i, stft_box_reg_i, stft_based_box_i in zip(
                shifts_per_image, box_cls, box_center, stft_box_cls, stft_box_delta, stft_based_box):
            # (HxWxK,)
            box_cls_i = box_cls_i.sigmoid_()
            box_ctr_i = box_ctr_i.sigmoid_()
            stft_box_cls_i = stft_box_cls_i.sigmoid_()
            
            predicted_prob = (box_cls_i * box_ctr_i).sqrt()
            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.pre_nms_thresh
            predicted_prob = predicted_prob * stft_box_cls_i
            predicted_prob = predicted_prob[keep_idxs]
            # Keep top k top scoring indices only.
            num_topk = min(self.pre_nms_top_n, predicted_prob.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = predicted_prob.sort(descending=True)
            topk_idxs = topk_idxs[:num_topk]

            keep_idxs = keep_idxs.nonzero()
            keep_idxs = keep_idxs[topk_idxs]
            keep_box_idxs = keep_idxs[:, 0]
            classes_idxs = keep_idxs[:, 1] + 1

            predicted_prob = predicted_prob[:num_topk]
            stft_box_reg_i = stft_box_reg_i[keep_box_idxs]
            stft_based_box_i = stft_based_box_i[keep_box_idxs]

            det_wh = (stft_based_box_i[..., 2:4] - stft_based_box_i[..., :2])
            det_wh = torch.cat([det_wh, det_wh], dim=1)
            predicted_boxes = stft_based_box_i + (stft_box_reg_i * stft_bbox_std * det_wh)
            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob.sqrt())
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_thresh)
        boxes_all = boxes_all[keep]
        scores_all = scores_all[keep]
        class_idxs_all = class_idxs_all[keep]

        number_of_detections = len(keep)
        if number_of_detections > self.fpn_post_nms_top_n > 0:
            image_thresh, _ = torch.kthvalue(
                scores_all,
                number_of_detections - self.fpn_post_nms_top_n + 1
            )
            keep = scores_all >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            boxes_all = boxes_all[keep]
            scores_all = scores_all[keep]
            class_idxs_all = class_idxs_all[keep]

        h, w = image_size
        boxlist = BoxList(boxes_all, (int(w), int(h)), mode="xyxy")
        boxlist.add_field("labels", class_idxs_all)
        boxlist.add_field("scores", scores_all)
        boxlist = boxlist.clip_to_image(remove_empty=False)
        boxlist = remove_small_boxes(boxlist, self.min_size)
        return boxlist

    def forward(self, shifts, box_cls, box_center, stft_box_cls, stft_box_delta, stft_based_box, image_sizes):
        """
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]
        stft_box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in stft_box_cls]
        stft_box_delta = [permute_to_N_HWA_K(x, 4) for x in stft_box_delta]

        for img_idx, (shifts_per_image,image_size_per_image) in enumerate(zip(shifts, image_sizes)):
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_ctr_per_image = [box_ctr_per_level[img_idx] for box_ctr_per_level in box_center]
            stft_box_cls_per_image = [stft_box_cls_per_level[img_idx] for stft_box_cls_per_level in stft_box_cls]
            stft_box_reg_per_image = [stft_reg_per_level[img_idx] for stft_reg_per_level in stft_box_delta]
            stft_based_box_per_image = [box_loc_per_level[img_idx] for box_loc_per_level in stft_based_box]

            results_per_image = self.forward_for_single_image(shifts_per_image,
                box_cls_per_image, box_ctr_per_image, stft_box_cls_per_image,
                stft_box_reg_per_image, stft_based_box_per_image, tuple(image_size_per_image)
            )
            results.append(results_per_image)
        return results


def make_fcos_stft_postprocessor(config):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    num_classes=config.MODEL.FCOS.NUM_CLASSES - 1
    stft_bbox_std = config.MODEL.STFT.BBOX_STD

    box_selector = STFTFCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=num_classes,
        stft_bbox_std=stft_bbox_std,
    )

    return box_selector

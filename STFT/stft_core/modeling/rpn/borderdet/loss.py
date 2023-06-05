import os
import torch
from torch.nn import functional as F

from stft_core.modeling.utils import cat
from stft_core.structures.bounding_box import BoxList #infer
from stft_core.structures.boxlist_ops import boxlist_iou




INF = 100000000

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def sigmoid_focal_loss(
    inputs,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)  # type: torch.jit.ScriptModule

def iou_loss(inputs, targets, weight=None, box_mode="xyxy", loss_type="iou", reduction="none"):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
        targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                   - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                   - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect
    ious = area_intersect / area_union.clamp(min=eps)

    if loss_type == "iou":
        loss = -ious.clamp(min=eps).log()
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])
        g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1 - gious
    else:
        raise NotImplementedError
    if weight is not None:
        loss = loss * weight.view(loss.size())
        if reduction == "mean":
            loss = loss.sum() / max(weight.sum().item(), eps)
    else:
        if reduction == "mean":
            loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()

    return loss

def smooth_l1_loss(input,
                   target,
                   beta: float,
                   reduction: str = "none",
                   size_average=False):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:

                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,

    where x = input - target.
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

    if reduction == "mean" or size_average:
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor

def permute_all_cls_and_box_to_N_HWA_K_and_concat(
    box_cls, box_delta, box_center, border_cls, border_delta, num_classes=2):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the centerness
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]

    border_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in border_cls]
    border_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in border_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    box_center = cat(box_center_flattened, dim=1).view(-1, 1)

    border_cls = cat(border_cls_flattened, dim=1).view(-1, num_classes)
    border_delta = cat(border_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta, box_center, border_cls, border_delta


class Shift2BoxTransform(object):
    def __init__(self, weights):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dl, dt, dr, db) deltas.
        """
        self.weights = weights

    def get_deltas(self, shifts, boxes):
        """
        Get box regression transformation deltas (dl, dt, dr, db) that can be used
        to transform the `shifts` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, shifts)`` is true.

        Args:
            shifts (Tensor): shifts, e.g., feature map coordinates
            boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(shifts, torch.Tensor), type(shifts)
        assert isinstance(boxes, torch.Tensor), type(boxes)

        deltas = torch.cat((shifts - boxes[..., :2], boxes[..., 2:] - shifts),
                           dim=-1) * shifts.new_tensor(self.weights)
        return deltas

    def apply_deltas(self, deltas, shifts):
        """
        Apply transformation `deltas` (dl, dt, dr, db) to `shifts`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single shift shifts[i].
            shifts (Tensor): shifts to transform, of shape (N, 2)
        """
        assert torch.isfinite(deltas).all().item()
        shifts = shifts.to(deltas.dtype)

        if deltas.numel() == 0:
            return torch.empty_like(deltas)

        deltas = deltas.view(deltas.size()[:-1] + (-1, 4)) / shifts.new_tensor(self.weights)
        boxes = torch.cat((shifts.unsqueeze(-2) - deltas[..., :2],
                           shifts.unsqueeze(-2) + deltas[..., 2:]),
                          dim=-1).view(deltas.size()[:-2] + (-1, ))
        return boxes


class BorderLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES 
        self.object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF], 
        ]
        self.shift2box_transform = Shift2BoxTransform(
            weights=(1.0, 1.0, 1.0, 1.0))
        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA 
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA 
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE 
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS 
        self.border_iou_thresh = cfg.MODEL.BORDER.IOU_THRESH
        self.border_bbox_std = cfg.MODEL.BORDER.BBOX_STD


    @torch.no_grad()
    def get_ground_truth(self, shifts, targets, pre_boxes_list):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
            gt_centerness (Tensor):
                An float tensor (0, 1) of shape (N, R) whose values in [0, 1]
                storing ground-truth centerness for each shift.
            border_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            border_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.

        """
        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []

        border_classes = []
        border_shifts_deltas = []

        for shifts_per_image, targets_per_image, pre_boxes in zip(shifts, targets, pre_boxes_list):

            object_sizes_of_interest = torch.cat([
                shifts_i.new_tensor(size).unsqueeze(0).expand(
                    shifts_i.size(0), -1) for shifts_i, size in zip(
                    shifts_per_image, self.object_sizes_of_interest)
            ], dim=0)

            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.bbox
            area = targets_per_image.area()
            center = targets_per_image.center()

            deltas = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.unsqueeze(1))

            if self.center_sampling_radius > 0:
                centers = targets_per_image.center()
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((
                        torch.max(centers - radius, gt_boxes[:, :2]),
                        torch.min(centers + radius, gt_boxes[:, 2:]),
                    ), dim=-1)
                    center_deltas = self.shift2box_transform.get_deltas(
                        shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = deltas.min(dim=-1).values > 0

            max_deltas = deltas.max(dim=-1).values
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_deltas >= object_sizes_of_interest[None, :, 0]) & \
                (max_deltas <= object_sizes_of_interest[None, :, 1])

            gt_positions_area = targets_per_image.area().unsqueeze(1).repeat(
                1, shifts_over_all_feature_maps.size(0))
            gt_positions_area[~is_in_boxes] = INF
            gt_positions_area[~is_cared_in_the_level] = INF

            # if there are still more than one objects for a position,
            # we choose the one with minimal area
            positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)

            # ground truth box regression
            gt_shifts_reg_deltas_i = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, targets_per_image[gt_matched_idxs].bbox)

            labels_per_im = targets_per_image.get_field("labels")

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = labels_per_im[gt_matched_idxs]
                # Shifts with area inf are treated as background.
                gt_classes_i[positions_min_area == INF] = self.num_classes+1 #value is 2 for not gt
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs)+self.num_classes+1 #value is 2 for not gt

            # ground truth centerness
            left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
            top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
            gt_centerness_i = torch.sqrt(
                (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
            )

            gt_classes.append(gt_classes_i) 
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i) 
            gt_centerness.append(gt_centerness_i) 

            # border
            iou = boxlist_iou(BoxList(pre_boxes, targets_per_image.size, targets_per_image.mode), targets_per_image)
            (max_iou, argmax_iou) = iou.max(dim=1)
            invalid = max_iou < self.border_iou_thresh
            gt_target = gt_boxes[argmax_iou]

            border_cls_target = labels_per_im[argmax_iou]
            border_cls_target[invalid] = self.num_classes+1 

            border_bbox_std = pre_boxes.new_tensor(self.border_bbox_std) 
            pre_boxes_wh = pre_boxes[:, 2:4] - pre_boxes[:, 0:2]
            pre_boxes_wh = torch.cat([pre_boxes_wh, pre_boxes_wh], dim=1)
            border_off_target = (gt_target - pre_boxes) / (pre_boxes_wh * border_bbox_std)

            border_classes.append(border_cls_target)
            border_shifts_deltas.append(border_off_target)

        return (
            torch.stack(gt_classes),
            torch.stack(gt_shifts_deltas),
            torch.stack(gt_centerness),
            torch.stack(border_classes),
            torch.stack(border_shifts_deltas),
        )


    def __call__(self, shifts, pred_class_logits, pred_shift_deltas, pred_centerness, 
        targets, bd_based_box, border_box_cls, border_bbox_reg):

        (
            gt_classes,
            gt_shifts_deltas,
            gt_centerness,
            gt_classes_border,
            gt_deltas_border,
        ) = self.get_ground_truth(shifts, targets, bd_based_box)

        (
            pred_class_logits,
            pred_shift_deltas,
            pred_centerness,
            border_class_logits,
            border_shift_deltas,
        ) = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_shift_deltas, pred_centerness,
            border_box_cls, border_bbox_reg, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        # fcos
        gt_classes = gt_classes.flatten().long()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4) 
        gt_centerness = gt_centerness.view(-1, 1) 

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != (self.num_classes+1))
        num_foreground = foreground_idxs.sum()
        acc_centerness_num = gt_centerness[foreground_idxs].sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]-1] = 1

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        num_foreground_avg_per_gpu = max(reduce_sum(num_foreground).item() / float(num_gpus), 1.0) 
        acc_centerness_num_avg_per_gpu = max(reduce_sum(acc_centerness_num).item() / float(num_gpus), 1.0)

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_foreground_avg_per_gpu

        # regression loss
        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            gt_centerness[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="sum",
        ) / acc_centerness_num_avg_per_gpu

        # centerness loss
        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_centerness[foreground_idxs],
            gt_centerness[foreground_idxs],
            reduction="sum",
        ) / num_foreground_avg_per_gpu

        # borderdet
        gt_classes_border = gt_classes_border.flatten().long() 
        gt_deltas_border = gt_deltas_border.view(-1, 4) 

        valid_idxs_border = gt_classes_border >= 0.
        foreground_idxs_border = (gt_classes_border >= 0) & (gt_classes_border != (self.num_classes+1))
        num_foreground_border = foreground_idxs_border.sum()

        gt_classes_border_target = torch.zeros_like(border_class_logits)
        gt_classes_border_target[
            foreground_idxs_border, gt_classes_border[foreground_idxs_border]-1] = 1

        num_foreground_border = max(reduce_sum(num_foreground_border).item() / float(num_gpus), 1.0)

        # loss_border_cls
        loss_border_cls = sigmoid_focal_loss_jit(
            border_class_logits[valid_idxs_border],
            gt_classes_border_target[valid_idxs_border],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_foreground_border

        if foreground_idxs_border.numel() > 0:
            loss_border_reg = (
                smooth_l1_loss(
                    border_shift_deltas[foreground_idxs_border],
                    gt_deltas_border[foreground_idxs_border],
                    beta=0,
                    reduction="sum"
                ) / num_foreground_border
            )
        else:
            loss_border_reg = border_shift_deltas.sum()

        return loss_cls, loss_box_reg, loss_centerness, loss_border_cls, loss_border_reg


def make_border_loss_evaluator(cfg):
    loss_evaluator = BorderLossComputation(cfg)
    return loss_evaluator

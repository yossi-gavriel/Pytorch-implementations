import torch


def calc_box_corners_midpoint(boxes):

    box_x1 = boxes[..., 0:1] - boxes[..., 2:3] / 2
    box_y1 = boxes[..., 1:2] - boxes[..., 3:4] / 2
    box_x2 = boxes[..., 2:3] + boxes[..., 2:3] / 2
    box_y2 = boxes[..., 3:4] - boxes[..., 3:4] / 2

    return box_x1, box_y1, box_x2, box_y2


def calc_box_corners_corners(boxes):

    box_x1 = boxes[..., 0:1]
    box_y1 = boxes[..., 1:2]
    box_x2 = boxes[..., 2:3]
    box_y2 = boxes[..., 3:4]  # (N, 1)

    return box_x1, box_y1, box_x2, box_y2


def intersection_over_union(boxes_preds, boxes_acts, box_format="midpoint"):
    """
    Calculate intersection over union
    :param boxes_preds : (tensor): shape (N,4) N is the number of bboxes
    :param boxes_acts : (tensor): tensor shape (N,4) N is the number of labels
    :param box_format:
    :return tensor: intersection over union for all examples
    """
    if box_format == "midpoint":

        box1_x1, box1_y1, box1_x2, box1_y2 = calc_box_corners_midpoint(boxes_preds)
        box2_x1, box2_y1, box2_x2, box2_y2 = calc_box_corners_midpoint(boxes_acts)

    else:

        box1_x1, box1_y1, box1_x2, box1_y2 = calc_box_corners_corners(boxes_preds)
        box2_x1, box2_y1, box2_x2, box2_y2 = calc_box_corners_corners(boxes_acts)

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.max(box1_x2, box2_x2)
    y2 = torch.max(box1_y2, box2_y2)

    # the clamp section is for case that they don't intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box1_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

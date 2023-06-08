import logging
import numpy as np
import cv2
import os
import torch


from .cvcvideo_eval import cvcvideo_detection_eval, cvcvideo_localization_center_eval


def vid_cvcvideo_evaluation(dataset, predictions, output_folder, visulize, vis_thr, **_):
    logger = logging.getLogger("stft_core.inference")
    logger.info(" performing cvcvideo evaluation.")


    pred_boxlists = []
    gt_boxlists = []
    filename_lists = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        if prediction.size[0] != image_width or prediction.size[1] != image_height:
            prediction = prediction.resize((image_width, image_height))
        prediction = prediction.clip_to_image(remove_empty=True)
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)

        filename_lists.append(dataset.get_img_name(image_id))

    if output_folder:
        torch.save(pred_boxlists, os.path.join(output_folder, "pred_boxlists.pth"))
        torch.save(gt_boxlists, os.path.join(output_folder, "gt_boxlists.pth"))
        torch.save(filename_lists, os.path.join(output_folder, "filename_lists.pth"))
    
    score_thrs = np.arange(0.5, 0.6, 0.1)

    logger.info(" Polyp Detection Task:")
    det_evals_dict = {}
    det_evals, det_tp, det_fp, det_tn, det_fn = cvcvideo_detection_eval(pred_boxlists, gt_boxlists, score_thrs)
    det_metrics = ['Precision', 'Recall', 'Accuracy', 'Sepcificity', 'F1_score', 'F2_score']
    for i in range(score_thrs.shape[0]):
        pt_string = '\nscore_thr:{:.2f}'.format(score_thrs[i])
        for j in range(len(det_metrics)):
            pt_string += '  {}: {:.4f} '.format(det_metrics[j], det_evals[j][i])
            each_name = '{}/score_thr:{:.2f}'.format(det_metrics[j], score_thrs[i])
            each_iterm = det_evals[j][i]
            det_evals_dict[each_name] = each_iterm
        logger.info(pt_string)

    # logger.info(" Polyp Localization Task:")
    # loc_center_evals_dict = {}
    # loc_evals, loc_tp, loc_fp, loc_fn = cvcvideo_localization_center_eval(pred_boxlists, gt_boxlists, score_thrs)
    # loc_metrics = ['Precision', 'Recall', 'F1_score', 'F2_score']
    # for i in range(score_thrs.shape[0]):
    #     pt_string = '\nscore_thr:{:.2f}'.format(score_thrs[i])
    #     for j in range(len(loc_metrics)):
    #         pt_string += '  {}: {:.4f} '.format(loc_metrics[j], loc_evals[j][i])
    #         each_name = '{}/score_thr:{:.2f}'.format(loc_metrics[j], score_thrs[i])
    #         each_iterm = loc_evals[j][i]
    #         loc_center_evals_dict[each_name] = each_iterm
    #     logger.info(pt_string)

    if output_folder and visulize:
        for image_id, (gt_boxlist, pred_boxlist) in enumerate(zip(gt_boxlists, pred_boxlists)):
            img, target, filename = dataset.get_visualization(image_id)
            save_line = filename+' '

            gt_bbox = gt_boxlist.bbox.numpy()
            gt_label = gt_boxlist.get_field("labels").numpy()
            if gt_label.sum()==0:
                save_line += str(0)+' '
            else:
                save_line += str(1)+' '
                for gt_idx in range(gt_label.sum()):
                    cv2.rectangle(img,(int(gt_bbox[gt_idx][0]),int(gt_bbox[gt_idx][1])),(int(gt_bbox[gt_idx][2]),int(gt_bbox[gt_idx][3])),(0,255,0),2)

            pred_score = pred_boxlist.get_field("scores").numpy()
            pred_bbox = pred_boxlist.bbox.numpy()
            det_inds = pred_score >= vis_thr
            highscore_score = pred_score[det_inds]
            highscore_bbox = pred_bbox[det_inds]

            if highscore_bbox.shape[0]>0:
                for temp_idx, temp_bbox in enumerate(highscore_bbox):
                    cv2.rectangle(img,(int(temp_bbox[0]),int(temp_bbox[1])),(int(temp_bbox[2]),int(temp_bbox[3])),(0,255,255),2)
                    cv2.putText(img, '{:.2f}'.format(highscore_score[temp_idx]), (int(temp_bbox[0]+10), int(temp_bbox[1]+10)), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,255), 2)

            cv2.imwrite(output_folder+'/'+filename.split('/')[-1]+'.jpg', img)
            save_line += str(det_tp[image_id][0])+' '+str(det_fp[image_id][0])+' '+str(det_tn[image_id][0])+' '+str(det_fn[image_id][0])+' '
            save_line += str(loc_tp[image_id][0])+' '+str(loc_fp[image_id][0])+' '+str(loc_fn[image_id][0])+'\n'
            with open(output_folder+'/result.txt', 'a+') as save_file:
                save_file.write(save_line)

    # return {'Detection':det_evals_dict, 'LocalizationCenter':loc_center_evals_dict}
    return {'Detection': det_evals_dict}

import time

import torch.autograd

from utils import *
from loss_poly import *
import tqdm
from ssl_utils import get_current_consistency_weight, update_ema_variables, get_random_batch


def train_one_epoch(model_name, loader, model, criterion, optimizer, epoch, lr_scheduler, device, batch, ema_model=None, ema_loader=None, mode=None):
    # if mode != "val":
    #     for idx, (ema_batch, _) in enumerate(ema_loader):
    #         ema_inputs = ema_batch.to(device)
    #         noise = torch.clamp(torch.randn_like(
    #             ema_inputs) * 0.1, -0.2, 0.2)
    #         ema_inputs = ema_inputs + noise
    #
    #         out_image = model(ema_inputs)[1]
    #         with torch.no_grad():
    #             ema_output = ema_model(ema_inputs)[1]
    #         ema_output = (ema_output > 0.5).float()
    #         out_loss = criterion(out_image, ema_output)
    #
    #         for i, (image, annotation, segment_image) in enumerate(loader):
    #             image, segment_image = image.to(device), segment_image.to(device)
    #             out_image = model(image)[1]
    #             out_loss_real = criterion(out_image, segment_image.float())
    #             break
    #
    #         out_loss = out_loss + out_loss_real
    #         optimizer.zero_grad()
    #         out_loss.backward()
    #         optimizer.step()
    #         update_ema_variables(model, ema_model, 0.9999, epoch * 126)
            # print(f'{epoch}-{i}-train_loss===>>{out_loss.item()}')
            # if model_name in []:

    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    dices = []
    average_loss, train_dice_avg, train_iou_average = 0.0, 0.0, 0.0
    for i, (image, annotation, segment_image) in enumerate(loader):
        out_loss = None
        out_image = None
        if model_name == "UNet_ssl":
            if mode=="val":
                image, segment_image = image.to(device), segment_image.to(device)
                out_image = model(image)[1]
                out_loss = criterion(out_image, segment_image.float())

            else:
                # Use one unlabeled batch for the EMA consistency branch.
                for idx, (ema_batch, _) in enumerate(ema_loader):
                    ema_inputs = ema_batch.to(device)
                    break

                # print(ema_batch)
                # noise = torch.clamp(torch.randn_like(
                #     ema_batch) * 0.1, -0.2, 0.2)  # torch.Size([12, 3, 224, 224])
                # ema_inputs = (ema_batch + noise).to(device)
                image, segment_image = image.to(device), segment_image.to(device)
                batchsize1, batchsize2 = image.shape[0], ema_inputs.shape[0]
                # print(image.shape, ema_inputs.shape)
                # assert image.shape[1:] == ema_inputs.shape[1:]
                # Labeled samples precede unlabeled samples along the batch dimension.
                concatenated_tensor = torch.cat((image, ema_inputs), dim=0)
                concatenated_out_image = model(concatenated_tensor)[1]
                out_image = concatenated_out_image[:batchsize1]
                with torch.no_grad():
                    ema_output = ema_model(ema_inputs)[1]
                # loss
                out_loss = criterion(out_image, segment_image.float())
                # ema_output = (ema_output > 0.5).float().squeeze(0)
                # consistency_loss = criterion(concatenated_out_image[batchsize1:], ema_output)
                if epoch * 20 < 1000:
                    consistency_loss = 0
                else:
                    consistency_loss = torch.mean((concatenated_out_image[batchsize1:] - ema_output) ** 2)
                out_loss = out_loss + consistency_loss

        if model_name == "UNet_ssl":
            if mode=="val":
                image, segment_image = image.to(device), segment_image.to(device)
                out_image = model(image)[1]
                out_loss = criterion(out_image, segment_image.float())

            else:
                image, segment_image = image.to(device), segment_image.to(device)
                out_image = model(image)[1]
                out_loss = criterion(out_image, segment_image.float())

        if model_name == "UNetText_ssl":
            pass

        # Image-only segmentation branch.
        if model_name == "UNet":
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = model(image)[1]
            # out_image = model(image)
            # print(image.shape, segment_image.shape, out_image.shape)
            out_loss = criterion(out_image, segment_image.float())
            # print(out_loss)

        # Text-conditioned segmentation branch.
        if model_name == "UNetText":
            image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
            out_image = model(image, annotation)[1]
            # print(image.shape, annotation.shape, segment_image.shape, out_image.shape)
            # torch.Size([8, 3, 256, 256]) torch.Size([8, 30, 768]) torch.Size([8, 2, 256, 256]) torch.Size([8, 2, 256, 256])
            out_loss = criterion(out_image, segment_image.float())
            # print(out_loss)

            # Below MICCAI liss
            # out_image_0, out_image_1 = model(image, annotation)
            # loss0 = criterion(out_image_0, segment_image.float())
            # loss1 = criterion(out_image_1, segment_image.float())
            # # print("loss 0: {:.3f}, loss 1: {:.3f}".format(loss0.item(), loss1.item()))
            # out_loss = 0.55 * loss0 + 0.45 * loss1
            # # 1 / 2 * (loss0 + loss1)
            # out_image = out_image_0

        #     June
        #     _, out_image, out_image_2, kl_div = model(image, annotation)
        #     seg_loss = criterion(out_image, segment_image.float())
        #     seg_loss2 = criterion(out_image_2, segment_image.float())
        #     kl_loss = torch.mean(kl_div)
        #     beta = 1e-3 #1e3
        #     print(seg_loss, kl_loss)
        #     # out_loss = seg_loss+ beta * kl_loss
        #     out_loss = seg_loss + 0.5 * seg_loss2 + beta * kl_loss

        # MMPL-Seg main branch. The first returned tensor is the prediction.
        if model_name == "MMPLSeg":
            image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
            out_image = model(image, annotation)[0]
            out_loss = criterion(out_image, segment_image.float())

        # Alignment-aware text-conditioned segmentation branch.
        if model_name == "AlignUNetText":
            image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
            out_image = model(image, annotation)[1]
            # print(image.shape, annotation.shape, segment_image.shape, out_image.shape)
            # torch.Size([8, 3, 256, 256]) torch.Size([8, 30, 768]) torch.Size([8, 2, 256, 256]) torch.Size([8, 2, 256, 256])
            out_loss = criterion(out_image, segment_image.float())
            # print(out_loss)

        # Image-text contrastive branch.
        if model_name == "IMC":
            image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
            out_image, image_similarity, text_similarity = model(image, annotation, mode="train")
            # print(image_similarity, text_similarity)
            # if i == 0:
            #     draw_similarity(image_similarity.cpu().detach().numpy(), text_similarity.cpu().detach().numpy())
            out_loss = criterion(out_image, segment_image.float(), image_similarity, text_similarity)

        # Image-text alignment branch.
        if model_name == "Align":
            image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
            logits_per_image, logits_per_text = model(image, annotation, mode="align")
            out_loss = criterion(logits_per_image.float(), logits_per_text.float())
            if epoch % 1000 == 0 & i == 0:
                print("epoch{}".format(epoch))
                draw_similarity(logits_per_image.cpu().detach().numpy(), logits_per_text.cpu().detach().numpy())

        # Feature-level alignment branch.
        if model_name == "AlignV2feature":
            image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
            image, text = model(image, annotation, mode="align")
            out_loss = criterion(image.float(), text.float())

        # =============================================================
        #       Draw
        # =============================================================
        # if i % 200 == 0:
        #     sample_img, sample_mask, sample_out = image[0].cpu(), segment_image[0].cpu(), out_image[0].cpu().detach().numpy()
        #     draw_seg_result(sample_img, sample_mask, sample_out)


        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            # last_layer_parameters = list(model.parameters())[1]
            # print("Parameters of the final layer:")
            # print(last_layer_parameters)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # update_ema_variables(model, ema_model, 0.999, epoch * 126)
            # print(f'{epoch}-{i}-train_loss===>>{out_loss.item()}')

        calculate = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)

        # train_dice_0 = calculate._show_dice(out_image_0, segment_image.float())
        # train_dice_1 = calculate._show_dice(out_image_1, segment_image.float())
        # print("train dice 0: {:.3f}, train dice 1: {:.3f}".format(train_dice_0, train_dice_1))


        train_dice = calculate._show_dice(out_image, segment_image.float())
        train_iou = iou_on_batch(segment_image, out_image)
        dices.append(train_dice)

        batch_time = time.time() - end
        time_sum += len(image) * batch_time
        loss_sum += len(image) * out_loss
        iou_sum += len(image) * train_iou
        # acc_sum += len(image) * train_acc
        dice_sum += len(image) * train_dice

        if (i+1) == len(loader):
            average_loss = loss_sum / (batch * i + len(image))
            average_time = time_sum / (batch * i + len(image))
            train_iou_average = iou_sum / (batch * i + len(image))
            # train_acc_average = acc_sum / (batch * i + len(image))
            train_dice_avg = dice_sum / (batch * i + len(image))
        else:
            average_loss = loss_sum / ((i+1) * batch)
            average_time = time_sum / ((i+1) * batch)
            train_iou_average = iou_sum / ((i+1) * batch)
            # train_acc_average = acc_sum / ((i+1) * batch)
            train_dice_avg = dice_sum / ((i+1) * batch)
        # print(average_loss.item(), train_dice_avg.item(), train_iou_average.item())
        end = time.time()
        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    # print(f"Epoch {epoch + 1}, Current Learning Rate: {current_lr}")

    return average_loss, train_dice_avg, train_iou_average

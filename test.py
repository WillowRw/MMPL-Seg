import random

import cv2
import torch
import tqdm
from torch.utils.data import DataLoader
from data import *
import matplotlib.pyplot as plt
from loss_poly import *
from Nets.Double_UnetModel4 import Dy_Double_UNet
from Nets.UNet import UNet
from Nets.UNet_plus_bert import UNetText

from sklearn.manifold import TSNE
# import segmentation_models_pytorch as smp
# from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_torch(seed=1029):  # 1029
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Disable hash randomization for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    # cv2.imwrite(save_path, predict_save * 255)
    # plt.imshow(predict_save * 255,cmap='gray')
    # plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    # plt.axis("off")
    # remove the white borders
    # height, width = predict_save.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig(save_path, dpi=2000)
    # plt.close()
    return dice_pred, iou_pred

def draw_tsne(vector_data):
    # Project NumPy feature vectors with t-SNE.
    tsne = TSNE(n_components=3, perplexity=30, n_iter=300)
    embedded_data = tsne.fit_transform(vector_data)
    # Plot the two leading projected dimensions.
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
    plt.title('t-SNE Visualization')
    plt.show()

def get_text_from_query(input_tensor, key_folder_path='QaTa-COV19-v2/Train_Set/Ground-truths'):
    # Input and output tensors have shape [1, 10, 256].
    keys = torch.rand(0, 2560)
    for filename in os.listdir(key_folder_path):
        if filename.endswith('.npy'):
            # Build the full path to the embedding file.
            npy_file_path = os.path.join(key_folder_path, filename)
            key_annotation = np.squeeze(np.load(npy_file_path), axis=0)
            key_annotation = torch.tensor(key_annotation).view(-1, 2560)
            keys = torch.cat((keys, key_annotation), dim=0)  # torch.Size([139, 2560])
    # print(keys.size())
    keys, input_tensor = keys.to(device), input_tensor.to(device)
    # Compute cosine similarity.
    similarities = torch.cosine_similarity(keys[:1000], input_tensor.view(-1, 2560)[:1000], dim=1)
    # print(similarities)
    # Find the most similar key.
    max_similarity_index = torch.argmax(similarities)
    print(max_similarity_index.item())
    # Return the value associated with the most similar key.
    output_tensor = keys[0].view(-1, 10, 256)

    # # Compute Euclidean distance.
    # distances = torch.cdist(keys.view(-1, 2560)[:1000], input_tensor.view(-1, 2560)[:1000])
    # # Find the nearest key.
    # min_distance_index = torch.argmin(distances)
    # print(min_distance_index)
    # # Retrieve the nearest vector.
    # output_tensor = keys.view(-1, 2560)[min_distance_index].view(-1, 10, 256)
    return output_tensor

def get_list(path='QaTa-COV19-v2/Train_Set/Ground-truths'):
    mask_list = torch.rand(0, 224*224)
    name_list = []
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            # Build the full path to the mask file.
            mask_path = os.path.join(path, filename)
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (224, 224))
            mask[mask <= 0] = 0
            mask[mask > 0] = 1
            mask = to_long_tensor(mask).view(-1, 224*224)
            mask_list = torch.cat((mask_list, mask), dim=0)
            name_list.append(filename)
    return mask_list, name_list

def get_npy(input_tensor, mask_list, name_list,
            text_embedding_dir='QaTa-COV19-v2/Train_Set/Text_Emb'):
    mask_list, input_tensor = mask_list.to(device), input_tensor.to(device)
    # Compute cosine similarity.
    similarities = torch.cosine_similarity(mask_list, input_tensor.view(-1, 224*224), dim=1)
    # print(similarities)
    # Find the most similar key.
    max_similarity_index = torch.argmax(similarities)
    annotation = torch.zeros(1, 10, 256)
    if max(similarities) >= 0.6:  # 0.6
        print("item num {}, similarity {}".format(max_similarity_index.item(), max(similarities)))
        annotation_path = os.path.join(text_embedding_dir,
                                       name_list[max_similarity_index.item()].replace("png", "npy"))
        annotation = torch.tensor(np.squeeze(np.load(annotation_path), axis=0))
    else:
        print("Fail, similarity {}".format(max(similarities)))
    return annotation


if __name__ == '__main__':
    seed_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 1
    test_data_path = r'QaTa-COV19-v2/Val_Set'
    test_loader = DataLoader(MyDataset(test_data_path, mode="test"),
                             batch_size=1,
                             shuffle=False)
    test_dataset_length = len(test_loader)
    print("Total data in DataLoader:", test_dataset_length)

    # Select the evaluation configuration.
    # model_name = "IMCModel"
    # model_name = "UNetText"
    # model_name = "UNet"
    model_name = "MMPLSeg"
    # model_name = "AlignUNetText"
    # model_name = "ProbabilisticUnet"

    model, weights = None, None
    if model_name == "MMPLSeg":
        model = Dy_Double_UNet(
            in_features=2560,
            out_features=224 * 224,
            r=512,
            n_channels=3,
            n_classes=num_classes,
        ).to(device)
        weights = 'params/Best_MMPLSeg.pth'
    if model_name == "UNet":
        model = UNet(n_channels=3, n_classes=num_classes).to(device)
        weights = 'params/Best_UNet.pth'

        # model = UNet(n_channels=3, n_classes=num_classes).to(device)
        # model = SwinTransformerSys(img_size=224, patch_size=4, in_chans=3, num_classes=1).to(device)
        # weights = './params_fewdata/4k_Best_UNet_ssl_Try_5percent_Epoch25.pth'
        # weights = 'params/Best_July_UNet_Cutmix_Epoch10.pth'
        # weights = 'params/Best_SwinTrans_UNet.pth'
    if model_name == "UNetText":
        print("Initializing UNetText.")
        model = UNetText(n_channels=3, n_classes=num_classes).to(device)
        weights = 'params/Best_UNetText.pth'

        # model = UNetText(n_channels=3, n_classes=num_classes).to(device)
        # weights = 'params/Best_Prob_Epoch17.pth'
        # model = UNetText(n_channels=3, n_classes=num_classes).to(device)
        # weights = 'params/Best_July_UNet_plus_VAE_kl100again_Epoch20.pth'
        # model = FT_UnetModel(in_features=2560, out_features=224 * 224, r=256).to(device)
        # weights = 'params/Best_256lora_UNetText.pth'
        # Dynamic double-UNet variant.
        # model = Dy_Double_UNet(in_features=2560, out_features=224 * 224, r=1024, n_channels=3, n_classes=1).to(device)
        # weights = 'params/Best_Double_UnetModel3_UNetText.pth'


        # sam_checkpoint = "params/sam_vit_h_4b8939.pth"
        # model_type = "default"
        # model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
        # mask_generator = SamAutomaticMaskGenerator(model)

    if model_name == "IMCModel":
        from Nets import IMC_model
        model = IMC_model.IMCModel(num_classes).to(device)
        weights = 'params/Best_1230_IMC.pth'
    if model_name == "AlignUNetText":
        from Nets.Align_UNet_plus_bert import AlignUNetText
        model = AlignUNetText(n_channels=3, n_classes=num_classes).to(device)
        weights = 'params/Best_0109_FeatureAlignIce_AlignUNetText.pth'

    if model_name == "ProbabilisticUnet":
        from Nets import probabilistic_unet
        model = probabilistic_unet.ProbabilisticUnet(input_channels=3).to(device)
        weights = 'params/Temp/ProbabilisticUnet_Epoch63.pth'


    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device))
        print('Loaded weights from {}.'.format(weights))
    else:
        raise FileNotFoundError('Checkpoint not found: {}.'.format(weights))

    # alignmodel = AlignModelV2().to(device)
    # weights = 'params/Best_AlignV2feature.pth'
    # if os.path.exists(weights):
    #     alignmodel.load_state_dict(torch.load(weights))
    #     print('Best_0111Short_AlignV2feature Weight load successfully')
    # else:
    #     print('Best_0111Short_AlignV2feature No Weights, train')
    # for param in alignmodel.parameters():
    #     param.requires_grad = False

    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0

    # unetmodel = UNet(n_channels=3, n_classes=num_classes).to(device)
    # unetweights = 'params/Best_SwinTransformerSys_UNet.pth'
    # if os.path.exists(weights):
    #     unetmodel.load_state_dict(torch.load(unetweights))
    #     print('unetweights Weight load successfully')

    # mask_list, name_list = get_list()
    df = pd.DataFrame(columns=['dice'])
    with torch.no_grad():
        for i, (image, annotation, segment_image) in enumerate(tqdm.tqdm(test_loader)):
            # print(annotation.size())
            arr = image.numpy()
            arr = arr.astype(np.float32)
            lab = segment_image.data[0, :].numpy()
            img_lab = np.reshape(lab, (lab.shape[0], lab.shape[1])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")

            height, width = 224, 224
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig("./result" + str(1) + "_lab.png", dpi=300)
            plt.close()

            output = None
            input_img = torch.from_numpy(arr)

            if model_name == "UNet":
                image, segment_image = image.to(device), segment_image.to(device)
                output = model(image)[1]  #[1]
            if model_name == "MMPLSeg":
                image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
                output = model(image, annotation)[0]
            if model_name == "UNetText":

                image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)

                # Probabilistic variant.
                # _, output, output2, kl_div = model(image, annotation)

                # Missing-modality evaluation.
                # random_number_in_range = random.uniform(0, 1)
                # # print(random_number_in_range)
                # if random_number_in_range > 0:  # Probability of retaining the text modality.
                #     # annotation = torch.zeros([1, 18, 256]).to(device)
                #     annotation = torch.from_numpy(np.load("assets/empty_text_embedding.npy")).to(device)
                # # # annotation = torch.from_numpy(np.load("assets/empty_text_embedding.npy")).to(
                # #     device)
                # # At test time, use 0 for missing text and 1 when text is available.
                output = model(image, annotation)[1]
                # print(output.size())



                # output = mask_generator.generate(image.squeeze(0))['segmentation']
                # print(len(output))

                # image = image.to(device)
                # annotation_gen = alignmodel(image, text=None, mode="test").view(-1, 10, 256)
                # annotation = get_text_from_query(annotation_gen)
                # all_vectors = np.vstack((all_vectors, annotation))

                # Two-stage prediction.
                # image = image.to(device)
                # image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
                # output = unetmodel(image)[1]
                # new_annotation = get_npy(output, mask_list, name_list).to(device)
                # output = model(image, new_annotation)[1]

                # LoRA variant.
                # image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
                # output = model(image, annotation)

                # Two-stage prediction with LoRA.
                # image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
                # output = model(image, annotation).to(device)
                # new_annotation = get_npy(output, mask_list, name_list).to(device)
                # output = model(image, new_annotation)


            if model_name == "AlignUNetText":
                image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
                output = model(image, annotation)[0]
            if model_name == "IMCModel":
                image, annotation, segment_image = image.to(device), annotation.to(device), segment_image.to(device)
                output = model(image, annotation, mode="test")

            if model_name == "ProbabilisticUnet":
                patch, annotation, masks = image.to(device), annotation.to(device), segment_image.to(device)
                # print(patch.size(), masks.size())
                # The implementation requires masks in this call but does not use them for prediction.
                model.forward(patch, masks, training=False)
                num_preds = 1
                predictions = []
                for i in range(num_preds):
                    mask_pred = model.sample(testing=True)
                    mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                    mask_pred = torch.squeeze(mask_pred, 0)
                    predictions.append(mask_pred)
                    # print(mask_pred.size())
                predictions = torch.cat(predictions, 0)
                output = predictions


            predicted = (output > 0.5).float()
            predicted = post_processing(predicted)
            batch_dice, batch_iou, batch_acc = calculate_metrics_iteration(predicted, segment_image.to(device))

            temp_df = pd.DataFrame({'dice': [batch_dice]})
            # Append the current sample metric.
            df = pd.concat([df, temp_df], ignore_index=True)


            total_dice += batch_dice
            total_iou += batch_iou
            total_acc += batch_acc

            # Optional qualitative examples.
            # if i ==0 :
            #     sample_img, sample_mask, sample_out = image[0].cpu(), segment_image[0].cpu(), predicted.cpu().detach().numpy()
            #     print(sample_img.shape, sample_mask.shape, sample_out.shape)
            #     draw_seg_result(i, sample_img, sample_mask, sample_out, model_name)
            # if i ==512 :
            #     sample_img, sample_mask, sample_out = image[0].cpu(), segment_image[0].cpu(), predicted.cpu().detach().numpy()
            #     print(sample_img.shape, sample_mask.shape, sample_out.shape)
            #     draw_seg_result(i, sample_img, sample_mask, sample_out, model_name)

            if i ==203 :
                sample_img, sample_mask, sample_out = image[0].cpu(), segment_image[0].cpu(), predicted.cpu().detach().numpy()
                print(sample_img.shape, sample_mask.shape, sample_out.shape)
                draw_seg_result(i, sample_img, sample_mask, sample_out, model_name)
            if i ==204 :
                sample_img, sample_mask, sample_out = image[0].cpu(), segment_image[0].cpu(), predicted.cpu().detach().numpy()
                print(sample_img.shape, sample_mask.shape, sample_out.shape)
                draw_seg_result(i, sample_img, sample_mask, sample_out, model_name)



        torch.cuda.empty_cache()
    # draw_tsne(all_vectors)
    # df.to_excel('draw/per_sample_dice.xlsx', index=False)
    print("Test ACC ", total_acc / test_dataset_length)
    print("Test Dice", total_dice / test_dataset_length)
    print("Test IOU", total_iou / test_dataset_length)


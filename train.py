import tqdm
import openpyxl
from torch import optim
from torch.utils.data import DataLoader
from data import MyDataset, MyDataset_ema
from Nets.Double_UnetModel4 import Dy_Double_UNet
from Nets.UNet import UNet
from Nets.UNet_plus_bert import UNetText
from utils import *
from loss_poly import *
import time
import logging
from Train_one_epoch import train_one_epoch
from utils import WeightedDiceBCE, WeightedDiceCE
from torch.optim.lr_scheduler import CosineAnnealingLR
torch.autograd.set_detect_anomaly(True)
from ssl_utils import TwoStreamBatchSampler
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    model_name = "MMPLSeg"
    # model_name = "UNet"
    # model_name = "UNetText"
    # model_name = "AlignUNetText"
    # model_name = "IMC"
    # model_name = "Align"
    # model_name = "AlignV2feature"
    # model_name = "ProbabilisticUnet"
    # model_name = "UNet_ssl"
    # model_name = "UNetText_ssl"

    num_classes = 1
    img_size = 224
    batch = 24
    total_epoch = 150
    model = None
    criterion = None
    ema_model=None
    db_ema = None
    ema_loader = None

    # QaTa-COV19-v2
    train_data_path = r'QaTa-COV19-v2/Train_Set'
    val_data_path = r'QaTa-COV19-v2/Val_Set'

    # MoNuSeg
    # train_data_path = r'datasets/MoNuSeg/Train_Folder'
    # val_data_path = r'datasets/MoNuSeg/Val_Folder'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("We are using device {}".format(device))
    seed_torch()
    os.makedirs('params/Temp', exist_ok=True)
    os.makedirs('params_fewdata/Temp', exist_ok=True)

    db_train = MyDataset(train_data_path, mode="train")
    db_val = MyDataset(val_data_path, mode="test")
    train_loader = DataLoader(db_train, batch_size=batch, shuffle=True)
    # train_loader = DataLoader(db_train, batch_size=batch, shuffle=True)
    val_loader = DataLoader(MyDataset(val_data_path, mode="test"),
                            batch_size=batch,
                            shuffle=True)

    if model_name in ["UNet_ssl", "UNetText_ssl"]:
        # aug_path = r'QaTa-COV19-v2\output_img_4k'
        # db_ema = MyDataset_ema(aug_path, mode="train")
        # ema_loader = DataLoader(db_ema, batch_size=batch, shuffle=True)
        aug_path = r'QaTa-COV19-v2/output_img_4k'
        db_ema = MyDataset_ema(aug_path, mode="train")
        ema_loader = DataLoader(db_ema, batch_size=batch, shuffle=True)


    train_dataset_length, val_dataset_length = len(train_loader), len(val_loader)
    print("train dataset length: ", train_dataset_length, "\nval dataset length: ", val_dataset_length)

    # =============================================================
    #
    # =============================================================


    if model_name == "UNet":
        model = UNet(n_channels=3, n_classes=num_classes).to(device)
        # model = SwinTransformerSys(img_size=224, patch_size=4, in_chans=3, num_classes=1).to(device)
        criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    if model_name == "UNetText":
        model = UNetText(n_channels=3, n_classes=num_classes).to(device)
        # model = FT_UnetModel(in_features=2560, out_features=224*224, r=1024).to(device)
        # model = Double_UNet(in_features=2560, out_features=224*224, r=1024, n_channels=3, n_classes=1).to(device)
        # Dynamic double-UNet experiment.
        # model = Dy_Double_UNet(in_features=2560, out_features=224 * 224, r=512, n_channels=3, n_classes=1).to(device)
        criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    if model_name == "MMPLSeg":
        model = Dy_Double_UNet(
            in_features=2560,
            out_features=img_size * img_size,
            r=512,
            n_channels=3,
            n_classes=num_classes,
        ).to(device)
        criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    if model_name == "AlignUNetText":
        from Nets.Align_UNet_plus_bert import AlignUNetText
        model = AlignUNetText(n_channels=3, n_classes=num_classes).to(device)
        criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    if model_name == "IMC":
        from Nets import IMC_model
        model = IMC_model.IMCModel(num_classes).to(device)
        # model.unet_block.load_state_dict(torch.load('params/Best_UNet.pth'))
        poly_lambd = 10  # Loss coefficient.
        criterion = LossPoly(poly_lambd)
    if model_name == "Align":
        from Nets import Align
        model = Align.AlignModel().to(device)
        # criterion = SimLoss()
        criterion = ClipLoss()
    if model_name == "AlignV2feature":
        from Nets import AlignV2feature
        model = AlignV2feature.AlignModelV2().to(device)
        criterion = nn.MSELoss()
    if model_name == "ProbabilisticUnet":
        from Nets import probabilistic_unet
        model = probabilistic_unet.ProbabilisticUnet(input_channels=3).to(device)
    if model_name == "UNet_ssl":
        model = UNet(n_channels=3, n_classes=num_classes).to(device)
        model_path = './params_fewdata/4k_Best_UNet_ssl_Try_5percent_Epoch7.pth'
        model.load_state_dict(torch.load(model_path))
        ema_model = UNet(n_channels=3, n_classes=num_classes).to(device)
        ema_path = './params_fewdata/4k_Best_UNet_ssl_Try_5percent_Epoch25.pth'
        ema_model.load_state_dict(torch.load(ema_path))

        for param in ema_model.parameters():
            param.detach_()
        criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    if model_name == "UNetText_ssl":
        model = UNetText(n_channels=3, n_classes=num_classes).to(device)  # Student Model
        ema_model = UNetText(n_channels=3, n_classes=num_classes).to(device)  # Teacher Model (Equipped with EMA)
        for param in ema_model.parameters():
            param.detach_()
        criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)



    # =============================================================
    #       Resume training (optional)
    # =============================================================
    # weights = 'params/Best_UNet.pth'
    # weights = 'params/Best_July_UNet_plus_VAE_kl50again_Epoch13.pth'
    # weights = 'params_fewdata/Best_Unet_5percent_Epoch117.pth'
    # if os.path.exists(weights):
    #     model.load_state_dict(torch.load(weights))
    #     print('Weight load successfully')
    # else:
    #     print('No Weights, train')


    # SGD optimizer (optional).
    # SDGmomentum = 0.99
    # SDGweight_decay = 0.0001
    # SDGlearning_rate = 1e-4
    # optimizer = torch.optim.SGD(model.parameters(), lr=SDGlearning_rate, momentum=SDGmomentum, weight_decay=SDGweight_decay)

    # AdamW optimizer used in the reported configuration.
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)  # 3e-4 # 0.001
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=50,
                                     eta_min=1e-6)  # T_max may be tuned relative to the total number of epochs.

    # Set this to the previous best score when resuming.
    best_val_dice = 0.0
    best_epoch = 1
    smallest_val_dice = 999 # 0.004606745205819607

    for epoch in range(total_epoch):
        if model_name in ["UNet_ssl", "UNetText_ssl"]:
            print("-------- Segmentation Training --------")
            print("=" * 20, "EPOCH", epoch + 1, "=" * 20)
            model.train(True)
            train_avg_loss, train_avg_dice, train_avg_iou = \
                train_one_epoch(model_name, train_loader, model, criterion, optimizer, epoch, lr_scheduler, device, batch, ema_model, ema_loader)
            print("--------Train Epoch {} loss {} dice {} IOU {} --------".format(epoch + 1, train_avg_loss, train_avg_dice,
                                                                                  train_avg_iou))
            weight_path = 'params_fewdata/Temp/4k_Temp_{}_Try_5percent.pth'.format(model_name)
            torch.save(model.state_dict(), weight_path)
            print('save successfully!')
            with torch.no_grad():
                model.eval()
                val_avg_loss, val_avg_dice, val_avg_iou = \
                    train_one_epoch(model_name, val_loader, model, criterion, optimizer, epoch, lr_scheduler, device, batch, mode="val")
                print("--------Val Epoch {} loss {} dice {} IOU {} --------".format(epoch + 1, val_avg_loss, val_avg_dice,
                                                                                    val_avg_iou))
                # =============================================================
                #       Save best model
                # =============================================================
                if val_avg_dice > best_val_dice:
                    best_val_dice = val_avg_dice
                    best_epoch = epoch + 1
                    best_weight_path = 'params_fewdata/4k_Best_{}_Try_5percent_Epoch{}.pth'.format(model_name, epoch)
                    torch.save(model.state_dict(), best_weight_path)
                    print('Best model saved successfully! This is epoch {}'.format(best_epoch))

        if model_name in ["Align", "AlignV2feature"]:
            # print("-------- Process of Align --------")
            model.train(True)
            train_avg_loss = \
                train_one_epoch(model_name, train_loader, model, criterion, optimizer, epoch, lr_scheduler, device,
                                batch)
            weight_path = 'params/Temp/Temp_SwinTransformerSys_{}.pth.'.format(model_name)
            torch.save(model.state_dict(), weight_path)
            print('save successfully!')
            with torch.no_grad():
                model.eval()
                val_align_loss = \
                    train_one_epoch(model_name, train_loader, model, criterion, optimizer, epoch, lr_scheduler, device,
                                    batch)
                if val_align_loss < smallest_val_dice:
                    smallest_val_dice = val_align_loss
                    best_epoch = epoch + 1
                    best_weight_path = 'params/Best_SwinTransformerSys_{}.pth.'.format(model_name)
                    torch.save(model.state_dict(), best_weight_path)
                    print('Best Align model saved successfully! This is epoch {}, loss {}'.format(best_epoch, smallest_val_dice))

        if model_name in ["MMPLSeg", "UNet", "UNetText", "AlignUNetText", "IMC"]:
            print("-------- Segmentation Training --------")
            print("=" * 20, "EPOCH", epoch + 1, "=" * 20)
            model.train(True)
            train_avg_loss, train_avg_dice, train_avg_iou = \
                train_one_epoch(model_name, train_loader, model, criterion, optimizer, epoch, lr_scheduler, device, batch)
            print("--------Train Epoch {} loss {} dice {} IOU {} --------".format(epoch + 1, train_avg_loss, train_avg_dice,
                                                                                  train_avg_iou))
            weight_path = 'params/Temp/Temp_{}_Epoch{}.pth'.format(model_name, epoch + 1)
            torch.save(model.state_dict(), weight_path)
            print('save successfully!')

            # =============================================================
            #       evaluate on validation set
            # =============================================================
            with torch.no_grad():
                model.eval()
                val_avg_loss, val_avg_dice, val_avg_iou = \
                    train_one_epoch(model_name, val_loader, model, criterion, optimizer, epoch, lr_scheduler, device, batch)
                print("--------Val Epoch {} loss {} dice {} IOU {} --------".format(epoch + 1, val_avg_loss, val_avg_dice,
                                                                                    val_avg_iou))
                # =============================================================
                #       Save best model
                # =============================================================
                if val_avg_dice > best_val_dice:
                    best_val_dice = val_avg_dice
                    best_epoch = epoch + 1
                    best_weight_path = 'params/Best_{}_Epoch{}.pth'.format(model_name, epoch + 1)
                    torch.save(model.state_dict(), best_weight_path)
                    print('Best model saved successfully! This is epoch {}'.format(best_epoch))

        if model_name in ["ProbabilisticUnet"]:
            print("-------- Segmentation Training --------")
            print("=" * 20, "EPOCH", epoch + 1, "=" * 20)
            from Nets.utils4prob import l2_regularisation
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
            for step, (patch, anno,  mask) in enumerate(train_loader):
                # print(patch.size(), mask.size())
                patch = patch.to(device)
                mask = mask.to(device)
                mask = torch.unsqueeze(mask, 1)
                model.forward(patch, mask, training=True)
                elbo = model.elbo(mask)
                reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(
                    model.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss
                print("loss:", loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            weight_path = 'params/Temp/ProbabilisticUnet_Epoch{}.pth'.format(epoch)
            torch.save(model.state_dict(), weight_path)
            print('save successfully!')

            # =============================================================
            #       evaluate on validation set
            # =============================================================
            # with torch.no_grad():
            #     model.eval()
            #     val_avg_loss, val_avg_dice, val_avg_iou = \
            #         train_one_epoch(model_name, val_loader, model, criterion, optimizer, epoch, lr_scheduler, device, batch)
            #     print("--------Val Epoch {} loss {} dice {} IOU {} --------".format(epoch + 1, val_avg_loss, val_avg_dice,
            #                                                                         val_avg_iou))
            #     # =============================================================
            #     #       Save best model
            #     # =============================================================
            #     if val_avg_dice > best_val_dice:
            #         best_val_dice = val_avg_dice
            #         best_epoch = epoch + 1
            #         best_weight_path = 'params/Best_Double_UnetModel4_March_addconcat_Epoch{}.pth.'.format(epoch)
            #         torch.save(model.state_dict(), best_weight_path)
            #         print('Best model saved successfully! This is epoch {}'.format(best_epoch))


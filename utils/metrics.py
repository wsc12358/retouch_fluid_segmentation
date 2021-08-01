import torch
import torch.nn.functional as F

def one_hot(mask, num_class):
    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """
    shape = mask.size()
    # label = label.view(-1, 1)
    # num_elem = label.size()[0]
    one_hot = torch.FloatTensor(shape[0], num_class, shape[1], shape[2]).random_().cuda()
    one_hot.zero_()
    label = mask.view(shape[0], 1, shape[1], shape[2])
    one_hot = one_hot.scatter_(1, label, 1)
    return one_hot

def get_dc(SR,GT,num_classes=4):
    smooth = 1.
    SR=F.softmax(SR,dim=1).float()
    SR=torch.argmax(SR,dim=1)
    SR=F.one_hot(SR,num_classes=num_classes).permute((0,3,1,2)).float()
    IRF_dice=-1
    SRF_dice=-1
    PED_dice=-1
    num=GT.size(0)
    if(GT[:,1,:,:].sum().item()>0):
        IRF_m1=SR[:,1,:,:].view(num,-1)
        IRF_m2=GT[:,1,:,:].view(num,-1)
        IRF_intersection = (IRF_m1 * IRF_m2).sum()
        IRF_dice=((2. * IRF_intersection + smooth) / (IRF_m1.sum() + IRF_m2.sum() + smooth)).item()

    if (GT[:, 2, :, :].sum().item() > 0):
        SRF_m1 = SR[:, 2, :, :].view(num, -1)
        SRF_m2 = GT[:, 2, :, :].view(num, -1)
        SRF_intersection = (SRF_m1 * SRF_m2).sum()
        SRF_dice = ((2. * SRF_intersection + smooth) / (SRF_m1.sum() + SRF_m2.sum() + smooth)).item()

    if (GT[:, 3, :, :].sum().item() > 0):
        PED_m1 = SR[:, 3, :, :].view(num, -1)
        PED_m2 = GT[:, 3, :, :].view(num, -1)
        PED_intersection = (PED_m1 * PED_m2).sum()
        PED_dice = ((2. * PED_intersection + smooth) / (PED_m1.sum() + PED_m2.sum() + smooth)).item()

    return [IRF_dice,SRF_dice,PED_dice]

# def get_DC(SR, GT):
#     # DC : Dice Coefficient
#     SR = torch.argmax(SR, dim=1)
#     SR = one_hot(SR, num_class=4).long()


#     GT = one_hot(GT, num_class=4).long()
#     # GT = GT == torch.max(GT)
#     IRFInter = torch.sum((SR[:, 1, :, :] + GT[:, 1, :, :]) == 2)
#     SRFInter = torch.sum((SR[:, 2, :, :] + GT[:, 2, :, :]) == 2)
#     PEDInter = torch.sum((SR[:, 3, :, :] + GT[:, 3, :, :]) == 2)
#     if torch.sum(GT[:,1,:,:])+torch.sum(SR[:,1,:,:])==0:
#         IRFDC=-1
#     else:
#         IRFDC = float(2 * IRFInter) / (float(torch.sum(SR[:, 1, :, :]) + torch.sum(GT[:, 1, :, :])) + 1e-6)

#     if torch.sum(GT[:,2,:,:])+torch.sum(SR[:,2,:,:])==0:
#         SRFDC=-1
#     else:
#         SRFDC = float(2 * SRFInter) / (float(torch.sum(SR[:, 2, :, :]) + torch.sum(GT[:, 2, :, :])) + 1e-6)

#     if torch.sum(GT[:,3,:,:])+torch.sum(SR[:,3,:,:])==0:
#         PEDDC=-1
#     else:
#         PEDDC = float(2 * PEDInter) / (float(torch.sum(SR[:, 3, :, :]) + torch.sum(GT[:, 3, :, :])) + 1e-6)

#     return IRFDC, SRFDC, PEDDC

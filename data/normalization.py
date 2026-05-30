import numpy as np
import SimpleITK as stk
import os


imgs_pth = "/home/cs4007/data/zyz/CDFSMIS/data/ABD_MR/"
imgs_list = os.listdir(imgs_pth)
imgs = [imgs_pth + fid for fid in imgs_list]

save_pth = "/home/cs4007/data/zyz/CDFSMIS/data/ABD_MR_NORM/"

for file in imgs_list:
    pth = imgs_pth + file
    image = stk.GetArrayFromImage(stk.ReadImage(pth))
    max = np.max(image)
    min = np.min(image)
    image = 255. * (image - min) / (max - min) 
    image = stk.GetImageFromArray(image)
    s_pth = save_pth + file
    stk.WriteImage(image, s_pth)


# for pid in imgs:
#     image = stk.GetArrayFromImage(stk.ReadImage(pid))
#     max = np.max(image)
#     min = np.min(image)
#     image = 255. * (image - min) / (max - min) 
#     stk.WriteImage(image, pid)
    
    
    
    





import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import *
from hsi_setup import Engine, train_options, make_dataset
import time

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)
    

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.net.use_2dconv)

   
    target_transform = HSI2Tensor()

    """Test-Dev"""
    df=[]
    test_dir = opt.test_dir
    for filename in os.listdir(test_dir):
        print(filename)
        save_path=os.path.join('/data/HSI_Data/Hyperspectral_Project/imgs/'+filename+'/')
        dir=test_dir+'/'+filename
        mat_dataset = MatDataFromFolder(
            dir)
        if not engine.get_net().use_2dconv:
            mat_transform = Compose([
                LoadMatHSI(input_key='input', gt_key='gt',
                        transform=lambda x:x[ ...][None], needsigma=False),
            ])
        else:
            mat_transform = Compose([
                LoadMatHSI(input_key='input', gt_key='gt', needsigma=False),
            ])

        mat_dataset = TransformDataset(mat_dataset, mat_transform)



        mat_loader = DataLoader(
            mat_dataset,
            batch_size=1, shuffle=False,
            num_workers=0, pin_memory=opt.no_cuda
        )

        base_lr = opt.lr
        epoch_per_save = 5
        adjust_learning_rate(engine.optimizer, opt.lr)

        engine.epoch  = 0


        strart_time = time.time()



        result=engine.test_rotate(mat_loader, dir,save_path)
        result=pd.Series(result,name=filename)

        df.append(result)

        end_time = time.time()
        test_time = end_time-strart_time
        print('cost-time: ',(test_time/len(mat_dataset)))
    df=pd.concat(df,axis=1)
    if not os.path.exists('test_result/' + os.path.basename(test_dir) + '.xlsx'):
        df.to_excel('test_result/' + os.path.basename(test_dir) + '.xlsx')
    else:
        for i in range(1,1000):
            if not os.path.exists('test_result/' + os.path.basename(test_dir)+f'_{i}' +'.xlsx'):
                df.to_excel('test_result/' + os.path.basename(test_dir)+f'_{i}' + '.xlsx')
                break


    ##python hside_simu_test_batch.py -a sert_base -p sert_base_gaussian_test -r -rp checkpoints/sert_base/sert_base_gaussian/model_latest.pth --test-dir /data/HSI_Data/test_noise

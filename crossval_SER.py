import train_ser
from train_ser import parse_arguments
import sys
import pickle
import os
import time


repeat_kfold = 1 # to  perform 10-fold for n-times with different seed
localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

#------------PARAMETERS---------------#

features_file = 'F:/Other_Speech_model/speech_Mamba/IEMOCAP_multi345200.pkl'
#  leave-one-speaker-out
#IEMOCAP
# val_id = ['1M','1F','2M','2F','3M','3F','4M','4F','5M','5F'] 
# test_id = ['1M','1F','2M','2F','3M','3F','4M','4F','5M','5F'] 
#EMODB
# val_id = ['03M', '09M', '10F', '11M', '12F', '13M', '14F', '15M', '16F']
# test_id = ['03M', '09M', '10F', '11M', '12F', '13M', '14F', '15M', '16F']
# #RAVDESS
# val_id = ['01M', '02F', '03M', '04F', '05M', '06F',
#     '07M', '08F', '09M', '10F', '11M', '12F',
#     '13M', '14F', '15M', '16F', '17M', '18F',
#     '19M', '20F', '21M', '22F', '23M', '24F']
# test_id =['01M', '02F', '03M', '04F', '05M', '06F',
#     '07M', '08F', '09M', '10F', '11M', '12F',
#     '13M', '14F', '15M', '16F', '17M', '18F',
#     '19M', '20F', '21M', '22F', '23M', '24F']
# MELD
val_id = ["Chandler",'Phoebe','Monica','Ross','Joey','others']
test_id =["Chandler",'Phoebe','Monica','Ross','Joey','others']


num_epochs  = '100'
early_stop = '8'
batch_size  = '64'
lr          = '0.00001'
random_seed = 111
gpu = '1'
gpu_ids = ['0']
save_label = str_time#'0930_01'#'alexnet_pm_0704'
dataset_type = "IEMOCAP" 
'''IEMOCAP/EMODB/RAVDESS/MELD'''
 

#Start Cross Validation
all_stat = []

for repeat in range(repeat_kfold):

    random_seed +=  (repeat*100)
    seed = str(random_seed)

    for v_id, t_id in list(zip(val_id, test_id)):

        train_ser.sys.argv      = [
                        
                                  'train_ser.py', 
                                  features_file,
                                  '--repeat_idx', str(repeat),
                                  '--val_id',v_id, 
                                  '--test_id', t_id,
                                  '--gpu', gpu,
                                  '--gpu_ids', gpu_ids,
                                  '--num_epochs', num_epochs,
                                  '--early_stop', early_stop,
                                  '--batch_size', batch_size,
                                  '--lr', lr,
                                  '--seed', seed,
                                  '--save_label', save_label,#,
                                  '--pretrained',
                                  '--dataset_type', dataset_type
                                  ]

    
        stat = train_ser.main(parse_arguments(sys.argv[1:]))   
        all_stat.append(stat)       
        os.remove(save_label+'.pth')
    
    # with open('allstat_iemocap_'+save_label+'_'+str(repeat)+'.pkl', "wb") as fout:
    #     pickle.dump(all_stat, fout)

n_total = repeat_kfold * len(val_id)

total_best_epoch = 0
total_epoch = 0
total_loss = 0
total_wa = 0
total_ua = 0
total_f1 = 0

for i in range(n_total):
    best_epoch = all_stat[i][1]
    epoch      = all_stat[i][0]

    test_loss  = float(all_stat[i][10])  # string → float
    test_wa    = float(all_stat[i][11])
    test_ua    = float(all_stat[i][12])
    test_f1    = float(all_stat[i][13])

    print(i, ": ",
          best_epoch,
          epoch,
          test_loss,
          test_wa,
          test_ua,
          test_f1)

    total_best_epoch += best_epoch
    total_epoch += epoch
    total_loss += test_loss
    total_wa += test_wa
    total_ua += test_ua
    total_f1 += test_f1

print("\nAVERAGE:",
      total_best_epoch / n_total,
      total_epoch / n_total,
      total_loss / n_total,
      total_wa / n_total,
      total_ua / n_total,
      total_f1 / n_total)

print("\nAll stat records:")
print(all_stat)


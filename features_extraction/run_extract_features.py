'''
Author: Shihe Dong
Description: Run_extract_features
IEMOCAP/EMODB/RAVDESS/MELD
ref:https://github.com/Vincent-ZHQ/CA-MSER
'''
import os
import sys
import argparse
import numpy as np
import pickle
from features_util import extract_features
from collections import Counter
import pandas as pd
from database import SER_DATABASES
import random


def main(args):
    
    #Get spectrogram parameters
    params={'window'        : args.window,
            'win_length'    : args.win_length,
            'hop_length'    : args.hop_length,
            'ndft'          : args.ndft,
            'nfreq'         : args.nfreq,
            'nmel'          : args.nmel,
            'segment_size'  : args.segment_size,
            'mixnoise'      : args.mixnoise
            }
    
    dataset  = args.dataset
    features = args.features
    dataset_dir = args.dataset_dir
    mixnoise = args.mixnoise

    if args.save_dir is not None:
        out_filename = args.save_dir+dataset+'_'+args.save_label +'.pkl'
    else:
        out_filename = 'None'

    print('\n')
    print('*'*50)
    print('\nFEATURES EXTRACTION')
    print(f'\t{"Dataset":>20}: {dataset}')
    print(f'\t{"Features":>20}: {features}')
    print(f'\t{"Dataset dir.":>20}: {dataset_dir}')
    print(f'\t{"Features file":>20}: {out_filename}')
    print(f'\t{"Add noise version":>20}: {mixnoise}')
    print(f"\nPARAMETERS:")
    for key in params:
        print(f'\t{key:>20}: {params[key]}')
    print('\n')

    # Random seed
    seed_everything(111)

    if dataset == 'IEMOCAP':
        # This is the 4-class, improvised data set
        #emot_map = {'ang':0,'sad':1,'hap':2,'neu':3}
        #include_scripted = False
        
        # Some publication works combined 'happy' and 'excited' into one 'happy' class,
        #   enable below for the 5531 dataset
        emot_map = {'ang':0,'sad':1,'hap':2, 'exc':2, 'neu':3}
        include_scripted = True 
        #Initialize database
        database = SER_DATABASES[dataset](dataset_dir, emotions_map=emot_map,
                                        include_scripted = include_scripted)
    elif dataset == "EMODB":
        '''
        对EMODB数据集传入情感标签。
        'ang':0,'sad':1, 'hap':2,'neu':3,"fea":4,'dis': 5,'bor':6
        '''
        emot_map = {'ang':0,'sad':1, 'hap':2,'neu':3,"fea":4,'dis': 5,'bor':6}
        database = SER_DATABASES[dataset](dataset_dir,emotions_map=emot_map)
    
    elif dataset == "RAVDESS":
        '''
        对RAVDESS数据集传入情感标签。
        RAVDESS表示：neu=01, hap=02, sad=03, ang=04, fea=05, dis=06, sur=07, con=08
        因此映射成0-7即可
        即：neu:0, hap:1, sad:2, ang:3, fea:4, dis:5, sur:6, con:7
        '''
        emot_map = {'01':0,'02':1,'03':2,'04':3,'05':4,'06':5,'07':6,'08':7}
        database = SER_DATABASES[dataset](dataset_dir,emotions_map=emot_map)
    elif dataset == "MELD":
        '''
        对MELD数据集传入情感标签。
        emotion_map={'neu':0, 'hap':1, 'sad':2, 'ang':3, 'sur':4, 'fear':5, 'dis':6}
        '''
        emot_map={'neu':0, 'hap':1, 'sad':2, 'ang':3, 'sur':4, 'fea':5, 'dis':6}
        database = SER_DATABASES[dataset](dataset_dir,emotion_map=emot_map)

    #Get file paths and label in database
    speaker_files = database.get_files()

    #Extract features
    features_data = extract_features(speaker_files, features, params)
    # print(type(features_data["3M"]))
    
    #Save features
    if args.save_dir is not None:
        
        with open(out_filename, "wb") as fout:
                pickle.dump(features_data, fout)

    #Print classes statistic
        
    print(f'\nSEGMENT CLASS DISTRIBUTION PER SPEAKER:\n')
    classes = database.get_classes()
    n_speaker=len(features_data)
    n_class=len(classes)
    class_dist= np.zeros((n_speaker,n_class),dtype=np.int32)
    speakers=[]
    data_shape=[]
    for i,speaker in enumerate(features_data.keys()):
        #print(f'\tSpeaker {speaker:>2}: {sorted(Counter(features_data[speaker][2]).items())}')
        cnt = sorted(Counter(features_data[speaker]["seg_label"]).items())
        
        for item in cnt:
            #print(item)
            class_dist[i][item[0]]=item[1]
        #print(class_dist)
        speakers.append(speaker)
        if mixnoise == True:
            data_shape.append(str(features_data[speaker]["seg_spec"][0].shape))
        else:
            data_shape.append(str(features_data[speaker]["seg_spec"].shape))
    class_dist = np.vstack(class_dist)
    #print(class_dist)
    df = {"speakerID": speakers,
          "shape (N,C,F,T)": data_shape}
    
    for c in range(class_dist.shape[1]):
        df[classes[c]] = class_dist[:,c]
    
    class_dist_f = pd.DataFrame(df)
    class_dist_f = class_dist_f.to_string(index=False) 
    print(class_dist_f)
     
    print('\n')
    print('*'*50)
    print('\n')



def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #DATASET
    parser.add_argument('--dataset', type=str, default='MELD',
        help='Dataset to extract features. Options:'
             '  - IEMOCAP (default)'
             '  - EMODB'
             '  - RAVDESS'
             '  - MELD')
    
    parser.add_argument('--dataset_dir', type=str, default='G:\dsh_postgraduate\Datasets\MELD.Raw',
        help='Path to the dataset directory.'
        'IEMOCAP:G:\dsh_postgraduate\Datasets\IEMOCAP'
        'EMODB:G:\dsh_postgraduate\Datasets\EMODB'
        'RAVDESS:G:\dsh_postgraduate\Datasets\RAVDESS\\219ed-main'
        'MELD:G:\dsh_postgraduate\Datasets\MELD.Raw')
    
    #FEATURES
    parser.add_argument('--features', type=str, default='logspec',
        help='Feature to be extracted. Options:'
             '  - logspec (default) : (1 ch.)log spectrogram'
             '  - logmelspec        : (1 ch.)log mel spectrogram')
    
    parser.add_argument('--window', type=str, default='hamming',
        help='Window type. Default: hamming')

    parser.add_argument('--win_length', type=float, default=40,
        help='Window size (msec). Default: 40')

    parser.add_argument('--hop_length', type=float, default=10,
        help='Window hop size (msec). Default: 10')
    
    parser.add_argument('--ndft', type=int, default=800,
        help='DFT size. Default: 800')

    parser.add_argument('--nfreq', type=int, default=200,
        help='Number of lowest DFT points to be used as features. Default: 200'
             '  Only effective for <logspec, lognrevspec> features')
    
    parser.add_argument('--nmel', type=int, default=128,
        help='Number of mel frequency bands used as features. Default: 128'
             '  Only effectice for <logmel, logmeldeltaspec> features')
    
    parser.add_argument('--segment_size', type=int, default=300,
        help='Size of each features segment')

    parser.add_argument('--mixnoise', action='store_true',
        help='Set this flag to mix with noise.')
    

    #FEATURES FILE
    parser.add_argument('--save_dir', type=str, default='G:\dsh_postgraduate\Other_Speech_model\CMTNET_Experiment_Paper\different_SNR\\',
        help='Path to directory to save the extracted features.')
    
    parser.add_argument('--save_label', type=str, default='nodb',
        help='Label to save the feature')

    return parser.parse_args(argv)


# seeding function for reproducibility
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    args.dataset = 'MELD'
    main(parse_arguments(sys.argv[1:]))
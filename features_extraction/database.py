'''
Author: Shihe Dong
Description: Database for SER.
IEMOCAP/EMODB/RAVDESS/MELD
ref:https://github.com/Vincent-ZHQ/CA-MSER
'''
import os
from collections import defaultdict
import pandas as pd
import ffmpeg
from pathlib import Path


''' Database:IEMOCAP'''

# 定义IEMOCAP的情感标签。字典。
IEMOCAP_EMOTIONS = {
    'neu': ['neu','neutral'],
    'hap': ['hap','happy','happiness'],
    'sad': ['sad','sadness'],
    'ang': ['ang','angry','anger'],
    'sur': ['sur', 'surprise', 'surprised'],
    'fea': ['fea', 'fear'],
    'dis': ['dis', 'disgust', 'disgusted'],
    'fru': ['fru', 'frustrated', 'frustration'],
    'exc': ['exc', 'excited', 'excitement'],
    'oth': ['oth', 'other', 'others']
}

class IEMOCAP_Database():
    '''
    初始化：
    database_dir:数据集路径。
    emotions_map:情感标签映射字典。
    include_scripted:是否包含scripted数据。【数据集特性】
    '''
    def __init__(self,database_dir,emotions_map= {'ang':0,'sad':1,'hap':2,'exc':2,'neu':3},
                 include_scripted=False):
        
        #path
        self.database_dir=database_dir

        #Emotion map
        self.emotions_map=emotions_map

        #IEMOCAP Session name 数据集一般分Sessions
        self.sessions=['Session1','Session2','Session3','Session4','Session5']

        #IEMOCAP emotion class
        self.all_emotions= IEMOCAP_EMOTIONS.keys()

        #IEMOCAP包含scripted数据。
        self.include_scripted = include_scripted

    def get_speaker_id(self,session,gender):
        '''
        获取说话人ID。
        session:Session1-5
        gender:M Male /F Female
        最后训练的时候1M，1F，2M，2F，3M，3F，4M，4F，5M，5F
        便于分配测试集验证集。
        '''
        ## session1 2 3 4 5,[-1]取最后一个字符
        return session[-1]+gender
    
    def get_classes(self):
        '''
        获取情感类别列表。进行列表拼接。hap+exc
        emotion_map={'ang':0,'sad':1,'hap':2,'neu':3}
        keys：ang,sad,hap,neu
        value:0,1,2,3
        是一个从0建立情感列表的代码。
        '''
        classes={} #定义空的情感类别标签。
        for key,value in self.emotions_map.items():
            #拼接情感类别列表
            if value in classes.keys():
                #如果值存在在keys里了，就在原有的文字标签上，加上新的文字。
                #hap:2 exc:2 -> hap+exc:2
                classes[value] += '+'+ key
            else:
                #如果keys不存在，就映射到里面，从0开始建情感列表。
                classes[value] = key
        
        return classes
    
    def get_files(self):
        '''
        获取数据集中的音频文件，将音频文件与speaker_id进行映射。
        返回一个字典：
        keys->speaker ID
        values->音频文件列表(.wav filepath, label) 元组 一个人可能有好多句。
        '''
        
        #取得情感。
        emotions = self.emotions_map.keys()
        dataset_dir = self.database_dir
        all_speaker_files = defaultdict(list) #定义一个字典，值是列表形式。
        total_num_files = 0

        #开始遍历。遍历文件目录下的所有session文件夹。
        '''
        目录格式：
        G:/Datasets/IEMOCAP/
            Session1/
            Session2/
            Session3/
            Session4/
            Session5/
        '''
        for session_name in os.listdir(dataset_dir):
            
            #如果list的文件夹里不是Session1-5，就跳过。
            if session_name not in self.sessions:
                continue
            '''
            eg.Datasets\IEMOCAP\Session1\sentences\wav
            wav里是一堆对话文件夹，对话文件里面是wav音频。
            '''
            wav_dir = os.path.join(dataset_dir,session_name,'sentences/wav')
            '''
            eg.Datasets\IEMOCAP\Session1\dialog\EmoEvaluation
            该文件夹是一堆情感文件txt，等后期再遍历。
            '''
            label_dir = os.path.join(dataset_dir,session_name,'dialog/EmoEvaluation')

            #接下来开始分男女。
            M_wav , F_wav= [],[] #定义男女说话人文件夹列表。
            for conversation_folder in os.listdir(wav_dir):
                #遍历wav文件夹下的对话文件夹。
                # if conversation_folder.startswith('.'):
                #     continue
                if self.include_scripted == False:
                    #排除scripted数据。只使用即兴数据。对文件夹名进行处理。
                    #只要文件夹为impro开头的。如果为False就不处理scripted数据。
                    #循环跳过这里走下一个。
                    if conversation_folder[7:12] != "impro":
                        continue
                
                #拼wav.文件地址。
                '''
                eg.Datasets\IEMOCAP\Session1\sentences\wav\Ses01F_impro01_F000
                能一直引到文件夹。
                '''
                conversation_dir = os.path.join(wav_dir,conversation_folder)

                #获得标签的地址。
                '''
                eg.Datasets\IEMOCAP\Session1\dialog\EmoEvaluation\Ses01F_impro01.txt
                每个语段都对应着txt。 相当于一个同名对话文件夹对应一个同名.txt，对话文件夹
                里对应着这个对话的所有wav.
                '''
                label_path=os.path.join(label_dir,conversation_folder+'.txt')
                #进行标签提取。
                labels={}
                with open(label_path,"r") as fin:
                    for line in fin: #逐行读取txt文件。
                        '''
                        [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
                        C-E2:	Neutral;	()
                        C-E3:	Neutral;	()
                        C-E4:	Neutral;	()
                        C-F1:	Neutral;	(curious)
                        A-E3:	val 3; act 2; dom  2;	()
                        A-E4:	val 2; act 3; dom  3;	(mildly aggravated but staying polite, attitude)
                        A-F1:	val 3; act 2; dom  1;	()

                        [10.0100 - 11.3925]	Ses01F_impro01_F001	neu	[2.5000, 2.5000, 2.5000]
                        C-E2:	Neutral;	()
                        C-E3:	Neutral;	()
                        C-E4:	Neutral;	()
                        C-F1:	Neutral; Anger;	()
                        A-E3:	val 3; act 2; dom  2;	()
                        A-E4:	val 2; act 3; dom  3;	(guarded, tense, ready)
                        A-F1:	val 2; act 3; dom  2;	()
                        只需要提取neu。
                        '''
                        if line[0]=="[":
                            t=line.split() #将句子按空格分隔。
                            #建立映射：
                            # Ses01F_impro01_F000 neu
                            # Ses01F_impro01_F001 neu
                            labels[t[3]]= t[4]
                #接下来开始建立音频映射。
                wav_files=[]
                for wav_name in os.listdir(conversation_dir):
                    #遍历对话文件夹下的所有wav音频文件。
                    #异常检测。
                    # if wav_name.startswith('.'):
                    #     continue
                    name, ext = os.path.splitext(wav_name) #名字/后缀名。
                    if ext != '.wav':
                        continue
                    emotion=labels[name] #通过映射获得情感标签。
                    if emotion not in emotions:
                        continue

                    label=self.emotions_map[emotion] #label通过映射只剩0，1，2，3
                    
                    #wav_name包含后缀名。 “（文件路径,label）”
                    wav_files.append((os.path.join(conversation_dir,wav_name),label))
                
                #分男女存储。遍历wav_files，根据文件名判断男女。wav_files是元组，所以要遍历第0项的字母。
                F_wav.extend([emo_wav for emo_wav in wav_files if emo_wav[0][-8] == "F"])
                M_wav.extend([emo_wav for emo_wav in wav_files if emo_wav[0][-8] == "M"])

                #统计文件数量。1F里面是（文件名，label）
                all_speaker_files[self.get_speaker_id(session_name,"M")] = M_wav
                all_speaker_files[self.get_speaker_id(session_name,"F")] = F_wav

                total_num_files += len(M_wav) + len(F_wav)
        print(f"IEMOCAP Database: Total number of files: {total_num_files}")
        return all_speaker_files

'''EMODB数据集'''

EMODB_EMOTIONS = {
    'W': 'ang',     # Ärger
    'T': 'sad',   # Trauer
    'F': 'hap', # Freude
    'N': 'neu' ,   # Neutral
    'A': 'fea',      # Angst
    'E': 'dis',   # Ekel
    'L': 'bor',   # Langeweile
}

class EMODB_Database():
    def __init__(self,database_dir,emotions_map= {'ang':0,'sad':1, 'hap':2,'neu':3,"fea":4,'dis': 5,'bor':6}):
        #记录EMODB的7类情感
        self.database_dir=database_dir
        self.emotions_map=emotions_map
    
    def get_speaker_id(self,filename):
        '''
        获取说话人ID。
        filename:文件名。
        eg. 03a02W.wav -> 03 03是奇数则代表是男性说话人，偶数则代表女性说话人。
        '''
        if int(filename[:2])%2==0:
            return filename[:2]+'F'
        else:
            return filename[:2]+'M'
    
    def get_classes(self):
        '''
        获取情感类别列表。key:表情。value:0-6
        '''
        classes={} #定义空的情感类别标签。
        for key,value in self.emotions_map.items():
            #拼接情感类别列表
            if value in classes.keys():
                #如果值存在在keys里了，就在原有的文字标签上，加上新的文字。
                #hap:2 exc:2 -> hap+exc:2
                classes[value] += '+'+ key
            else:
                #如果keys不存在，就映射到里面，从0开始建情感列表。[其实EMODB不太需要]
                classes[value] = key
        
        return classes
    
    def get_files(self):
        '''
        获取数据集中的音频文件，将音频文件与speaker_id进行映射。
        keys->speaker ID
        values->音频文件列表(.wav filepath, label) 元组 一个人可能有好多句。
        '''
        emotions= self.emotions_map.keys()
        dataset_dir=os.path.join(self.database_dir,"wav") #EMODB的wav文件夹路径
        all_speaker_files = defaultdict(list) #定义一个字典，值是列表形式

        #遍历文件目录下的所有音频文件。
        '''
        eg.Datasets/EMODB/wav 里面有一堆音频文件。
        eg.03a02W.wav --> 03号说话人，W->anger
        ''' 
        
        for filename in os.listdir(dataset_dir):
            name,ext=os.path.splitext(filename) #名字/后缀名。
            if ext != '.wav':
                continue
            emotion_code=name[5] #情感代码是第6个字符。
            #检查情感是否在emodb_emotions中。
            if emotion_code not in EMODB_EMOTIONS:
                continue
            emotion=EMODB_EMOTIONS[emotion_code] #通过映射获得情感标签。
            if emotion not in emotions:
                        continue
            label=self.emotions_map[emotion]   #label通过映射只剩0,1,2,3,4,5,6
            #进行拼接 3M->[(filepath,label), (filepath,label)...]
            all_speaker_files[self.get_speaker_id(filename)].append((os.path.join(dataset_dir,filename),label))
        return all_speaker_files

'''RAVDESS数据集'''
'''RAVDESS表示：neu=01, hap=02, sad=03, ang=04, fea=05, dis=06, sur=07, con=08'''
'''因此映射成0-7即可'''
'''即：neu:0, hap:1, sad:2, ang:3, fea:4, dis:5, sur:6, con:7'''

RAVDESS_EMOTIONS_Map={
    '0':"neu",
    '1':"hap",
    '2':"sad",
    '3':"ang",
    '4':"fea",
    '5':"dis",
    '6':"sur",
    '7':"cal"
}
class RAVDESS_Database():
    def __init__(self,database_dir,emotions_map= {'01':0,'02':1,'03':2,'04':3,'05':4,'06':5,'07':6,'08':7}):
        #记录RAVDESS的8类情感
        self.database_dir=database_dir
        self.emotions_map=emotions_map
    def get_speaker_id(self,filename):
        '''
        获取说话人ID。
        filename:文件名。
        eg. 03-01-05-01-02-01-12.wav -> 05:愤怒 说话人：12
        '''
        parts=filename.split('-') #通过-分割文件名,03 01 05 01 02 01 12.wav
        speaker_id=parts[-1].split('.')[0] #最后一部分是说话人，把12.wav取到后，按照.分割成12 wav，取最前面的。
        if int(speaker_id)%2==0:
            return speaker_id+'F'
        else:
            return speaker_id+'M'

    def get_classes(self):
        '''
        获取情感类别列表。Key:情绪中文。value:0-7
        '0':"neu",
        '1':"hap",
        '2':"sad",
        '3':"ang",
        '4':"fea",
        '5':"dis",
        '6':"sur",
        '7':"con"
        emotions_map= {'01':0,'02':1,'03':2,'04':3,'05':4,'06':5,'07':6,'08':7}
        '''
        # 确保 RAVDESS_EMOTIONS_Map 是用数值作为键（0-7）
        RAVDESS_EMOTIONS_Map = {
            0: "neu",
            1: "hap",
            2: "sad",
            3: "ang",
            4: "fea",
            5: "dis",
            6: "sur",
            7: "con"
        }
        
        classes = {}  # 存储 {数值标签: 拼接后的情绪字符串}
        for key, value in self.emotions_map.items():
            # value 是 0-7 的数值（对应情绪标签），用它查 RAVDESS_EMOTIONS_Map
            emotion_label = RAVDESS_EMOTIONS_Map[value]
            
            if value in classes:  # 直接判断数值标签是否已在 classes 中
                # 若存在，拼接情绪标签（如多个编码对应同一数值时）
                classes[value] += '+' + emotion_label
            else:
                # 若不存在，新增键值对
                classes[value] = emotion_label
        
        # 按数值标签排序后返回列表（确保顺序为 0→1→2...）
        return [classes[val] for val in sorted(classes.keys())]
    
    def get_files(self):
        '''
        获取数据集中的音频文件，将音频文件与speaker_id进行映射。
        返回一个字典：
        keys->speaker ID
        values->音频文件列表(.wav filepath, label) 元组 一个人可能有好多句。
        G:\dsh_postgraduate\Datasets\RAVDESS\\219ed-main\\
          Actor 01\
          Actor 02\
        '''
        emotions= self.emotions_map.keys()
        dataset_dir=self.database_dir
        persons=["Actor_01","Actor_02","Actor_03","Actor_04","Actor_05","Actor_06","Actor_07","Actor_08","Actor_09","Actor_10",
                 "Actor_11","Actor_12","Actor_13","Actor_14","Actor_15","Actor_16","Actor_17","Actor_18","Actor_19","Actor_20",
                 "Actor_21","Actor_22","Actor_23","Actor_24"]
        all_speaker_files = defaultdict(list) #定义一个字典，值是列表形式。
        for person in persons:
            for filename in os.listdir(os.path.join(dataset_dir,person)):
                #G:\dsh_postgraduate\Datasets\RAVDESS\\219ed-main\\Actor_01 filename都是.wav
                name,ext=os.path.splitext(filename) #名字/后缀名。
                if ext!=".wav":
                    continue
                if name.split('-')[2] not in emotions:
                    continue
                label=self.emotions_map[name.split('-')[2]] #进行label映射。
                all_speaker_files[self.get_speaker_id(filename)].append((os.path.join(os.path.join(dataset_dir,person),filename),label))
        
        return all_speaker_files



'''MELD Datasets'''
''' MELD 情感映射（原始 7 类） '''
MELD_EMOTIONS = {
    'neutral': 'neu',
    'joy': 'hap',
    'sadness': 'sad',
    'anger': 'ang',
    'surprise': 'sur',
    'fear': 'fea',
    'disgust': 'dis'
}

'''
Download:https://github.com/declare-lab/MELD
'''


class MELD_Database():
    def __init__(self,database_dir,emotion_map={'neu':0, 'hap':1, 'sad':2, 'ang':3, 'sur':4, 'fea':5, 'dis':6}):
        '''
        G:\dsh_postgraduate\Datasets\MELD.Raw
        \\train_splits
        \\output_repeated_splits_test
        \\dev_splits_complete
        \\train_sent_emo.csv
        \\test_sent_emo.csv
        \\dev_sent_emo.csv
        '''
        self.database_dir = database_dir #G:\dsh_postgraduate\Datasets\MELD.Raw
        self.emotion_map = emotion_map
        
        '''CSV文件路径'''
        self.csv_train = os.path.join(database_dir,"train_sent_emo.csv")
        self.csv_test = os.path.join(database_dir,"test_sent_emo.csv")
        self.csv_dev = os.path.join(database_dir,"dev_sent_emo.csv")

        '''音频路径'''
        self.audio_train = os.path.join(database_dir,"train_splits")
        self.audio_test = os.path.join(database_dir,"output_repeated_splits_test")
        self.audio_dev = os.path.join(database_dir,"dev_splits_complete")

        # 新增：初始化样本计数器（全局唯一，统计所有样本）
        self.speaker_count = 0  
        # 新增：定义test样本的数量阈值
        self.test_sample_threshold = 548  


    def get_speaker_id_train_test(self,wav_dir):
        '''
        新增逻辑：前548个样本返回test，剩余返回train
        兜底逻辑：路径包含train_splits则返回train，否则test
        '''
        # 计数器自增（每个样本调用时+1）
        self.speaker_count += 1
        
        # 核心逻辑：按计数判断
        if self.speaker_count <= self.test_sample_threshold:
            return str("test")
        else:
            return str("train")
        
        # （可选保留）兜底逻辑：如果需要路径判断作为备用，取消注释
        # if "train_splits" in wav_dir:
        #     return str("train")
        # else:
        #     return str("test")
    
    def get_speaker_id(self,speaker):
        """
        判断输入的speaker是否为核心人物，非核心人物返回'others'
        
        Args:
            speaker: 输入的说话人名称（字符串）
            
        Returns:
            str: 核心人物返回原名称，非核心人物返回'others'
        """
        # 定义核心人物列表（与你指定的val_id/test_id一致）
        core_speakers = ['Chandler','Phoebe','Monica','Ross','Joey','Rachel']
        
        # 去除首尾空格（防止输入有多余空格导致判断错误），统一大小写不敏感（可选）
        speaker_clean = speaker.strip()
        
        # 判断是否为核心人物
        if speaker_clean not in core_speakers:
            return 'others'
        # 核心人物返回原名称（也可根据需求返回speaker_clean）
        return speaker_clean

    def get_classes(self):
        classes={}
        for key,value in self.emotion_map.items():
            if value in classes.keys():
                classes[value] += '+'+ key
            else:
                classes[value] = key
        return classes
    
    def get_files(self):
        '''返回字典格式'''
        '''eg.speaker:[(wav_path,label)]'''
        all_speaker_files = defaultdict(list)

        # 重置计数器（每次调用get_files时重新计数，避免累计）
        self.speaker_count = 0

        #读取csv文件。
        def load_csv(csv_path,audio_dir):
            df = pd.read_csv(csv_path) #读取csv
            '''CSV文件构造：
            Sr No.	Utterance	Speaker	Emotion	Sentiment	Dialogue_ID	Utterance_ID	Season	Episode	StartTime	EndTime
                1	also I was the point person on my company聮s transition from the KL-5 to GR-6 system.	Chandler	neutral	neutral	0	0	8	21	00:16:16,059	00:16:21,731
                2	You must聮ve had your hands full.	The Interviewer	neutral	neutral	0	1	8	21	00:16:21,940	00:16:23,442
                3	That I did. That I did.	Chandler	neutral	neutral	0	2	8	21	00:16:23,442	00:16:26,389
                4	So let聮s talk a little bit about your duties.	The Interviewer	neutral	neutral	0	3	8	21	00:16:26,820	00:16:29,572
            '''
            #按行遍历全部内容。
            for _, row in df.iterrows():
                emotion = row["Emotion"] #eg. anger
                speaker = row["Speaker"] #eg. Ross
                dialogue_id = row["Dialogue_ID"]
                utterance_id = row["Utterance_ID"]

                '''
                MELD文件名格式：
                dia{dialogue_id}_utt{utterance_id}.wav
                '''
                wav_name = f"dia{dialogue_id}_utt{utterance_id}.wav"
                wav_path = os.path.join(audio_dir,wav_name) #G:\dsh_postgraduate\Datasets\MELD.Raw\train_splits\

                if not os.path.isfile(wav_path):
                    # print(f"[跳过] 找不到 WAV 文件：{wav_path}")
                    continue

                #emotion映射为label id
                emo_key = MELD_EMOTIONS[emotion]
                label = self.emotion_map[emo_key]

                # 调用计数版的get_speaker_id
                speaker_id = self.get_speaker_id(speaker)
                all_speaker_files[speaker_id].append((wav_path,label))
            
        failed_files = []  # 记录失败文件列表

        def convert_mp4_to_wav(mp4_path, wav_path, sr=16000):
            """将单个 MP4 转成 WAV（损坏文件自动跳过）"""
            print(f"[正在转换] {mp4_path}")

            try:
                # 检测是否有音频轨道
                probe = ffmpeg.probe(str(mp4_path))
                audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
                if len(audio_streams) == 0:
                    print(f"⚠ 无音轨 → 跳过 {mp4_path}")
                    failed_files.append(mp4_path)
                    return

                (
                    ffmpeg
                    .input(str(mp4_path))
                    .output(str(wav_path), ac=1, ar=sr, loglevel="error")
                    .overwrite_output()
                    .run()
                )

            except Exception as e:
                print(f"转换失败 → 跳过 {mp4_path}")
                print(f"错误信息：{e}")
                failed_files.append(mp4_path)
                # 不 raise，不中断继续处理后续文件


        def batch_convert_mp4_to_wav(root_dir, sr=16000):
            """
            将 MELD 某个目录下所有 .mp4 批量转成 .wav
            root_dir: MELD.Raw/train_splits 等目录
            """
            root = Path(root_dir)
            mp4_files = list(root.rglob("*.mp4"))  # 支持递归子目录

            print(f"发现 {len(mp4_files)} 个 mp4 文件，开始转换...\n")

            for mp4_file in mp4_files:
                wav_file = mp4_file.with_suffix(".wav")
                convert_mp4_to_wav(mp4_file, wav_file, sr)

            print("\n转换完成！")

            # 写入失败日志
            if failed_files:
                log_path = root / "failed_convert_list.txt"
                with open(log_path, "w", encoding="utf-8") as f:
                    for item in failed_files:
                        f.write(str(item) + "\n")

                print(f"共 {len(failed_files)} 个文件失败，已保存日志：{log_path}")
            else:
                print("没有失败文件！")

        
        #加载Train/Dev/Test,加载前先转换。不会覆盖原有mp4文件。但是遇到没音频的.mp4会报错，只处理有音频的。
        # batch_convert_mp4_to_wav(self.audio_train)
        # batch_convert_mp4_to_wav(self.audio_dev)
        # batch_convert_mp4_to_wav(self.audio_test)
        load_csv(self.csv_train, self.audio_train)
        load_csv(self.csv_dev,self.audio_dev)
        load_csv(self.csv_test, self.audio_test)
        
        # 新增：打印计数统计，验证逻辑是否生效
        print(f"总计样本数：{self.speaker_count}")
        print(f"Test样本数：{len(all_speaker_files.get('test', []))}")
        print(f"Train样本数：{len(all_speaker_files.get('train', []))}")
        
        return all_speaker_files
    




#负责后续调用。
SER_DATABASES = {'IEMOCAP': IEMOCAP_Database,
                 'EMODB': EMODB_Database,
                 'RAVDESS':RAVDESS_Database,
                 'MELD': MELD_Database
                 }
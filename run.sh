#!/bin/bash

# 通过镜像源法产生多通道信号（GPURIR），并和噪声进行混合，最终转换为频带SRP-PHAT与STFT谱相位的组合特征
# 并训练
# 目录下需创建csv文件夹，内部放置语料的csv文件以及噪声的csv文件
# stage=1，产生房间冲击响应
# stage=2，vad切割语音
# stage=3 产生多通道混响信号并混合噪声
# stage=4，特征提取
# staeg=5，训练
# stage=6，测试
pathwd='/home/admin/AEC/NKF_train/'
datapath=
feapath=
thread=8
stage=3
# rir集合
csv_rir=$pathwd'csv/rir_generate_test.csv'
# rir保存路径
rir_path=$pathwd'rir/'
# data保存路径
data_path=$pathwd'data/'
# vad 录音保存路径
vad_save_path=$pathwd'data/vad/'
# 原始音频集合
csv_wav=$pathwd'csv/wav1mic_5w.csv'



# 切割音频集合
sig_csv=$pathwd'csv/vad1mic.csv'
# 原始语音路径
speech_wav_path=$datapath'speech/'
# 参考语音路径
ref_wav_path=$datapath'ref/'
# 训练语音集合
train_wav_path=$datapath'train_data/'
# 原始语音集合
speech_csv=$pathwd'csv2/speech_csv.csv'
# 回声语音集合
echo_csv=$pathwd'csv/echo_sig.csv'
# 参考语音集合
ref_csv=$pathwd'csv2/ref_sig.csv'
# 训练语音集合
train_csv=$pathwd'csv2/train.csv'

# 特征提取（提高CPU训练时运行效率）
fea_csv=$pathwd'csv2/train/fea/'
# 原始语音STFT特征
fea_speech_path=$feapath'speech/'
# 参考回路STFT特征
fea_ref_path=$feapath'ref/'
# 训练数据STFT特征
fea_train_path=$feapath'train/'


# 训练特征路径
fea_train_csv=$pathwd'csv2/train/fea/train.csv'
fea_train_val_csv=$pathwd'csv2/train/fea/train_val.csv'
# 测试特征集合
fea_speech_csv=$pathwd'csv2/train/fea/speech.csv'
fea_speech_val_csv=$pathwd'csv2/train/fea/speech_val.csv'
# 验证特征集合
fea_ref_csv=$pathwd'csv2/train/fea/ref_sig.csv'
fea_ref_val_csv=$pathwd'csv2/train/fea/ref_sig_val.csv'



if [ ! -z $1 ]; then
    stage=$1
fi
# 后来依靠随机噪声生成，和论文对齐（下面的代码已经删除，具体可参考fastrir）
# 产生房间冲击响应(需设置参数过多，在此不引出)
if [ $stage -le 0 ]; then
    echo "Step 1: Generate RIRs"
    if [ ! -d $rir_path  ]; then
      mkdir $rir_path
    else
      echo rir_path exist
    fi
    python generate_rir.py --csv_rir $csv_rir || exit 1;

fi

# 数据进行VAD切割
if [ $stage -le 1 ]; then
    echo "Step 2: VAD"
    if [ ! -d $data_path  ]; then
      mkdir $data_path
    else
      echo data_path exist
    fi
    
    if [ ! -d $vad_save_path  ]; then
      mkdir $vad_save_path
    else
      echo vad_save_path exist
    fi
    python ./vad/vad.py --vad_save_path $vad_save_path --csv_wav $csv_wav --csv_vad $csv_vad || exit 1;
fi  
    
    
# 信号混合，并加回声，暂时不加噪（因为GPURIR暂不支持多进程，所以在此只能使用多线程处理）现改成shell中多进程
# 代码中存在的采样频率fs，音频大小，信噪比大小进代码调整
if [ $stage -le 2 ]; then
    echo "Step 3: Generate datas" 
    
    if [ ! -d $ref_wav_path  ]; then
      mkdir $ref_wav_path
    else
      echo "ref_wav_path exist"
    fi
    
    if [ ! -d $speech_wav_path  ]; then
      mkdir $speech_wav_path
    else
      echo "speech_wav_path exist"
    fi
    
    if [ ! -d $train_wav_path  ]; then
      mkdir $train_wav_path
    else
      echo "train_wav_path exist"
    fi
    
    for i in `seq $thread`; do
    {
    python ./data_generator/data_generator_pfdata.py --save_path $datapath --start $i --thread $thread --sig_csv ${sig_csv} --speech_csv ${speech_csv} --echo_csv ${echo_csv} --ref_csv ${ref_csv} --train_csv ${train_csv} || exit 1;
    }&
    done 
    wait
fi

if [ $stage -le 3 ]; then
    echo "Step 4: Feature extract"
    
    if [ ! -d $feapath  ]; then
      mkdir $feapath
    else
      echo "feapath exist"
    fi

    if [ ! -d $fea_csv  ]; then
      mkdir $fea_csv
    else
      echo "fea_csv exist"
    fi
    
    if [ ! -d $fea_speech_path  ]; then
      mkdir $fea_speech_path
    else
      echo "fea_speech_path exist"
    fi
    
    if [ ! -d $fea_train_path  ]; then
      mkdir $fea_train_path
    else
      echo "fea_train_path exist"
    fi
    
    if [ ! -d $fea_ref_path  ]; then
      mkdir $fea_ref_path
    else
      echo "fea_ref_path exist"
    fi
    
    for i in `seq 4`; do
    {
    python ./feature_extract/feature_extract.py --index $i --train_csv $train_csv --ref_csv $ref_csv --speech_csv $speech_csv --feature_path $feapath || exit 1;
    }&
    done
    wait
    for i in {5..8}; do
    {
    python ./feature_extract/feature_extract.py --index $i --train_csv $train_csv --ref_csv $ref_csv --speech_csv $speech_csv --feature_path $feapath || exit 1;
    }&
    done
    wait
    # 前8个csv代表train，第9个csv代表val,test
    python ./feature_extract/combine_utils.py --csv_number 8 --fea_train_csv $fea_train_csv --fea_val_csv $fea_val_csv --fea_speech_train_csv $fea_speech_train_csv --fea_speech_val_csv $fea_speech_val_csv --fea_ref_train_csv $fea_ref_train_csv --fea_ref_val_csv $fea_ref_val_csv || exit 1;
    
fi

# if [ $stage -le 4 ]; then
#     echo "Step 5: Train"
#     python ./train/train_nkf.py  --train_csv $fea_train_csv --train_val_csv $fea_train_val_csv --speech_csv $fea_speech_csv --speech_val_csv $fea_speech_val_csv --ref_val_csv $fea_ref_val_csv --ref_csv $fea_ref_csv|| exit 1;
    
# fi

# if [ $stage -le 5 ]; then
#     echo "Step 6: Test"
#     python test_srp_multi.py --model_dir $model_dir --fea_test_csv $fea_test_csv || exit 1;
    
# fi

# if [ $stage -le 6 ]; then
#     echo "Step 7: Quant"
#     python quantization.py --model_dir $model_dir  --train_list $fea_train_csv --dev_list $fea_val_csv --lr 0.0004 --epoch $quant_epoch --batch_size $batch_size || exit 1
    
# fi




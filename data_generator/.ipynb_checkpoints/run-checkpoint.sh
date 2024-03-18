#!/bin/bash

# 通过镜像源法产生多通道信号（GPURIR），并和扫地机噪声进行混合，最终转换为频带SRP-PHAT与STFT谱相位的组合特征
# 并训练
# 目录下需创建csv文件夹，内部放置语料的csv文件以及噪声的csv文件
# stage=1，产生房间冲击响应
# stage=2，vad切割语音
# stage=3 产生多通道混响信号并混合噪声
# stage=4，特征提取
# staeg=5，训练
# stage=6，测试

datapath='/mnt/users/daihuangyu/AEC_Challenge/NKF_Data_From_AEC_Challenge/double_talk_B12_hard/'
#datapath='/mnt/users/daihuangyu/AEC_Challenge/NKF_Data_From_AEC_Challenge/synthetic4/'
echopath=$datapath'echo_signal/'
farendpath=$datapath'farend_speech/'
nearendspeechpath=$datapath'nearend_speech/'
nearendmicpath=$datapath'nearend_mic_signal'
thread=8
stage=1




if [ ! -z $1 ]; then
    stage=$1
fi


# 信号混合，并加回声，暂时不加噪（因为GPURIR暂不支持多进程，所以在此只能使用多线程处理）现改成shell中多进程
# 代码中存在的采样频率fs，音频大小，信噪比大小进代码调整
if [ $stage -le 1 ]; then
    echo "Step 1: Generate datas" 
    
    if [ ! -d $datapath  ]; then
      mkdir $datapath
    else
      echo "data path exist"
    fi    
    
    if [ ! -d $echopath  ]; then
      mkdir $echopath
    else
      echo "echo path exist"
    fi
    
    if [ ! -d $farendpath  ]; then
      mkdir $farendpath
    else
      echo "farend speech path exist"
    fi
    
    if [ ! -d $nearendspeechpath  ]; then
      mkdir $nearendspeechpath
    else
      echo "nearend speech path exist"
    fi

    if [ ! -d $nearendmicpath  ]; then
      mkdir $nearendmicpath
    else
      echo "nearend mic path exist"
    fi
    
    for i in `seq $thread`; do
    {
    python ./data_generator_sameaspaper.py --index $i --thread $thread || exit 1;
    }&
    done 
#     for i in `seq $thread`; do
#     {
#     python ./data_generator_single_talk.py --index $i --thread $thread || exit 1;
#     }&
#     done 
    wait
fi


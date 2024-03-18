import pandas as pd
import argparse
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='df combine')
    parser.add_argument('--csv_number', dest='csv_number',type=int, required=False, 
                        default=8 ,help='number noise wav csv')
    parser.add_argument('--fea_train_csv', dest='fea_train_csv',type=str, required=False, 
                        default='../csv2/train/fea/train.csv' ,help='train_csv') 
    parser.add_argument('--fea_val_csv', dest='fea_val_csv',type=str, required=False, 
                        default='../csv2/train/fea/train_val.csv' ,help='val_csv')  
    parser.add_argument('--fea_speech_train_csv', dest='fea_speech_train_csv',type=str, required=False, 
                        default='../csv2/train/fea/speech.csv' ,help='fea_speech_train_csv')  
    parser.add_argument('--fea_speech_val_csv', dest='fea_speech_val_csv',type=str, required=False, 
                        default='../csv2/train/fea/speech_val.csv' ,help='fea_speech_val_csv') 
    parser.add_argument('--fea_ref_train_csv', dest='fea_ref_train_csv',type=str, required=False, 
                        default='../csv2/train/fea/ref_sig.csv' ,help='fea_ref_train_csv')  
    parser.add_argument('--fea_ref_val_csv', dest='fea_ref_val_csv',type=str, required=False, 
                        default='../csv2/train/fea/ref_sig_val.csv' ,help='fea_ref_val_csv')  
    
    args = parser.parse_args()
    
    dir_path, file_name = os.path.split(args.fea_train_csv)
    file_name, extension = os.path.splitext(file_name)
    file_path = dir_path+'/'+file_name+'_p'+str(1)+extension
    df_train = pd.read_csv(file_path)
    for i in range(2,args.csv_number):
        file_path = dir_path+'/'+file_name+'_p'+str(i)+extension
        df = pd.read_csv(file_path)
        df_train = pd.concat([df_train,df], join="inner")

    df_train.to_csv(args.fea_train_csv,index=False,sep=',')
    
    val_ori_name = dir_path+'/'+file_name+'_p'+str(args.csv_number)+extension
    df_val = pd.read_csv(val_ori_name)
    df_val.to_csv(args.fea_val_csv,index=False,sep=',')

    # ref
    dir_path, file_name = os.path.split(args.fea_ref_train_csv)
    file_name, extension = os.path.splitext(file_name)
    file_path = dir_path+'/'+file_name+'_p'+str(1)+extension

    df_ref_train = pd.read_csv(file_path)
    for i in range(2,args.csv_number):
        file_path = dir_path+'/'+file_name+'_p'+str(i)+extension
        df = pd.read_csv(file_path)
        df_ref_train = pd.concat([df_ref_train,df], join="inner")

    df_ref_train.to_csv(args.fea_ref_train_csv,index=False,sep=',')
    
    val_ori_name = dir_path+'/'+file_name+'_p'+str(args.csv_number)+extension
    df_ref_val = pd.read_csv(val_ori_name)
    df_ref_val.to_csv(args.fea_ref_val_csv,index=False,sep=',')
    
    # speech
    dir_path, file_name = os.path.split(args.fea_speech_train_csv)
    file_name, extension = os.path.splitext(file_name)
    file_path = dir_path+'/'+file_name+'_p'+str(1)+extension
    df_speech_train = pd.read_csv(file_path)
    for i in range(2,args.csv_number):
        file_path = dir_path+'/'+file_name+'_p'+str(i)+extension
        df = pd.read_csv(file_path)
        df_speech_train = pd.concat([df_speech_train,df], join="inner")

    df_speech_train.to_csv(args.fea_speech_train_csv,index=False,sep=',')
    
    val_ori_name = dir_path+'/'+file_name+'_p'+str(args.csv_number)+extension
    df_speech_val = pd.read_csv(val_ori_name)
    df_speech_val.to_csv(args.fea_speech_val_csv,index=False,sep=',')
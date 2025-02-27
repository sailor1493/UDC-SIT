python3 src/train_udcsit.py --arch Uformer_T --H 1792 --W 1280 --batch_size 4 --batch_size_val 8 --gpu '0,1,2,3' \
    --train_ps 1280 --train_dir /data/s0/udc/dataset/UDC_SIT/UDC-SIT/training --patch_size 1280 \
    --val_ps 1280 --val_dir /data/s0/udc/dataset/UDC_SIT/UDC-SIT/validation --env _neurips --dataset_name UDC_SIT \
    --test_dir /data/s0/udc/dataset/UDC_SIT/UDC-SIT/test \
    --mode motiondeblur --nepoch 2500 --checkpoint 50 --dataset UDC_SIT --warmup --dd_in 4 --save_img_iter 1 #\
    #--resume --pretrain_weights ./logs/UDC_SIT/Uformer_T_0518/models/model_latest.pth

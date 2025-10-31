export CUDA_VISIBLE_DEVICES=0


EthanolConcentration
python main.py --attention_type hybrid --output_dir experiments --comment "classification for AGF" --name EthanolConcentration_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/EthanolConcentration --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam --pos_encoding learnable --task classification --key_metric accuracy --model svd --eta 0.1 --poly_type jacobi --K 6 --alpha 1.5 --beta -1.5 --fixI

# FaceDetection
python main.py --attention_type hybrid --output_dir experiments --comment "classification for AGF" --name FaceDetection_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/FaceDetection --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model svd  --eta 0.1 --poly_type jacobi --K 9 --alpha 2.0 --beta 0.5 --fixI

# HandWriting
python main.py --attention_type hybrid --output_dir experiments --comment "classification for AGF" --name Handwriting_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/Handwriting --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model svd --eta 0.01 --poly_type jacobi --K 3

# HeartBeat
python main.py --attention_type hybrid --output_dir experiments --comment "classification for AGF" --name Heartbeat_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/Heartbeat --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model svd --eta 0.01 --poly_type chebyshev --K 6 

# JapaneseVowels
python main.py --attention_type hybrid --output_dir experiments --comment "classification for AGF" --name JapaneseVowels_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/JapaneseVowels --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model svd --eta 0.01 --poly_type legendre --K 4 

# PEMS-SF
python main.py --attention_type hybrid --output_dir experiments --comment "classification for AGF" --name PEMS-SF_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/PEMS-SF --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 400 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model svd --eta 0.1 --poly_type jacobi --K 5 --alpha 1.5 --beta 2.0 --fixI

# SelfRegulationSCP1
python main.py --attention_type hybrid --output_dir experiments --comment "classification for AGF" --name SelfRegulationSCP1_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/SelfRegulationSCP1 --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model svd --eta 0.1 --poly_type jacobi --K 9 --alpha 1.5 --beta 1.0 --fixI

# SelfRegulationSCP2
python main.py --attention_type hybrid --output_dir experiments --comment "classification for AGF" --name SelfRegulationSCP2_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/SelfRegulationSCP2 --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model svd --eta 0.1 --poly_type jacobi --K 7 --alpha 2.0 --beta 1.5 --fixI

# SpokenArabicDigits
python main.py --attention_type hybrid --output_dir experiments --comment "classification for AGF" --name SpokenArabicDigits_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/SpokenArabicDigits --data_class tsra  --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer Adam  --pos_encoding learnable  --task classification  --key_metric accuracy --model svd --eta 0.5 --poly_type jacobi --K 4 --alpha 1.0 --beta 0.0 --fixI

# UWave
python main.py --attention_type hybrid --output_dir experiments --comment "classification for AGF" --name UWaveGestureLibrary_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/UWaveGestureLibrary --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model svd --eta 0.1 --poly_type jacobi --K 10 --alpha 2.0 --beta 1.5 --fixI

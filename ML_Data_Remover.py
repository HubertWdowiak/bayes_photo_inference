import os
import random
path="/home/data/data/kodeiri/ML_project/train_aug_buffer/"
arr=os.listdir(path)
folder_name=arr[0]
img_list=os.listdir(path+folder_name)
target_range=random.randrange(4000,5000)
i=0

for i  in range (0,target_range):
    img_count=len(img_list)
    print(img_count)
    random_index=random.randrange(0,img_count)
    removed_file_name=img_list[random_index]
    os.remove(path+folder_name+"/"+removed_file_name)
    img_list.remove(removed_file_name)
    

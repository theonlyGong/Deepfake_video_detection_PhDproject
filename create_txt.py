# -*- coding: utf-8 -*-
# @Time : 2024/4/1 11:01
# @Author : Liangyu Gong
# @FileName: create_txt.py
# @Software: PyCharm

import os


if __name__ == "__main__":
    real_root = os.path.join(os.getcwd(), 'data', 'Real')
    fake_root = os.path.join(os.getcwd(), 'data', 'Fake')
    assert os.path.exists(real_root)
    assert os.path.exists(fake_root)

    for roots, dirs, files in os.walk(real_root):
        for video_name in files:
            video_path = os.path.join(roots, video_name)
            assert os.path.exists(video_path)
            # 转换为相对路径
            rel_path = os.path.relpath(video_path, os.getcwd())
            folder_name = os.path.basename(os.path.dirname(video_path))
            if folder_name == 'Train':
                with open('real_train.txt','a') as f:
                    f.write(rel_path+' 0'+'\n')
            elif folder_name == 'Validate':
                with open('real_val.txt','a') as f:
                    f.write(rel_path+' 0'+'\n')
            elif folder_name == 'Test':
                with open('real_test.txt','a') as f:
                    f.write(rel_path+' 0'+'\n')


    for roots, dirs, files in os.walk(fake_root):
        for video_name in files:
            video_path = os.path.join(roots, video_name)
            assert os.path.exists(video_path)
            # 转换为相对路径
            rel_path = os.path.relpath(video_path, os.getcwd())
            folder_name = os.path.basename(os.path.dirname(video_path))
            if folder_name == 'Train':
                with open('fake_train.txt','a') as f:
                    f.write(rel_path+' 1'+'\n')
            elif folder_name == 'Validate':
                with open('fake_val.txt','a') as f:
                    f.write(rel_path+' 1'+'\n')
            elif folder_name == 'Test':
                with open('fake_test.txt','a') as f:
                    f.write(rel_path+' 1'+'\n')


import os

def get_list(path) -> list:
    r"""Recursively read all files in root path"""
    # 清理路径中的空字符
    path = path.replace('\x00', '')
    image_list = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.split('.')[-1] in ['png', 'jpg', 'jpeg']:
                image_list.append(os.path.join(root, f))
    return image_list
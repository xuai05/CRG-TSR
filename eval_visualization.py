import h5py
import os
from matplotlib import pyplot as plt
import json
import cv2
from PIL import Image

def plot_visualization(images):
    n = len(images)
    fig = plt.figure()
    for i in range(n):
        fig.add_subplot(1,n,i+1)
        plt.imshow((images[i]))
    plt.show()

def save_visualization2video(images, path=None):
    fps = 4          
    size = (300, 300)
    if not path:
        video = cv2.VideoWriter("./output.avi", cv2.VideoWriter_fourcc('X','V','I','D'), fps, size)
    else:
        video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('X','V','I','D'), fps, size)
    for i in range(len(images)):
        img = cv2.resize(images[i],size)
        r,g,b = cv2.split(img)
        pic = cv2.merge([b,g,r])
        video.write(pic)
    video.release()
    cv2.destroyAllWindows()


y = 0.9009992
data_dir = 'AI2Thor_offline_data_2.0.2_images'
with open('visual_temp.json', 'r') as f:
    grid = json.load(f)

count = 0
for episode in grid:
    scene = episode['scene']
    target = episode['target'][0].split('|')[0]
    states = episode['states']
    is_success = episode['success']
    file_name = os.path.join(data_dir, scene, 'images.hdf5')
    f = h5py.File(file_name, 'r')
    
    images = []
    for i in range(len(states)):
        images.append(f[states[i]].value)
    f.close()
    video_path = os.path.join('./videos_exp1', str(is_success) + '_' + target + '_' + scene + '_' + str(count) + '.avi')
    save_visualization2video(images, video_path)
    count = count + 1
    if count % 100 == 0:
        print(count)

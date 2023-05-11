import numpy as np
import cv2
import os
import PIL

# path = r'C:\Users\Rajnish\Desktop\geeksforgeeks\geeks.png'
    
# # Reading an image in default mode
# image = cv2.imread(path)
    
# # Window name in which image is displayed
# window_name = 'Image'
  
# # font
# font = cv2.FONT_HERSHEY_SIMPLEX
  
# # org
# org = (50, 50)
  
# # fontScale
# fontScale = 1
   
# # Blue color in BGR
# color = (255, 0, 0)
  
# # Line thickness of 2 px
# thickness = 2
   
# # Using cv2.putText() method
# image = cv2.putText(image, 'OpenCV', org, font, 
#                    fontScale, color, thickness, cv2.LINE_AA)
   
# # Displaying the image
# cv2.imshow(window_name, image) 


def label_rescale(matriX):
    matriX = 360*matriX + 360*np.ones(np.shape(matriX))
    return matriX


input_npz = './MSP_videos_npz'
output_mp4 = './MSP_videos_mp4'
output_labels = './Custom_outs/'
# targets = './targetLabels/'
# fourcc = cv2.VideoWriter_fourcc('F', 'F', 'V', '1')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 25
circle = cv2.imread('MagicCircle_.png')
circle_ = cv2.resize(circle, (720, 720))
red_patch = np.zeros((32, 32, 3))
red_patch[:, :, 2] = 255
green_patch = np.zeros((32, 32, 3))
green_patch[:, :, 1] = 255

try:
    os.mkdir(output_mp4)
except:
    print('Output directory exists')
# selected_list = ['SAH_C6_S181_P362_VC1_000742_001590', 'SAL_C1_S025_P050_VC1_000701_001916', 'SSD_C2_S035_P069_VC1_002352_003021', 'SVH_C4_S107_P214_VC1_005501_006000', 'SVH_C5_S144_P287_VC1_003201_003701']
# selected_list = ['SAH_C1_S009_P018_VC1_004321_005367.npz','SAH_C2_S037_P074_VC1_002225_003088.npz']

# selected_list = os.listdir(output_labels)

for name in os.listdir(input_npz):
    
    # if name in selected_list:
    print(name)
    # for num in os.listdir(str(input_npz + '/' +name)):
    video = np.load(input_npz  + '/' + name +'/' + name +'.npz')['data']
    # targets_ar_raw = np.loadtxt(targets +name + '/' + name +'_Arousal_V_Aligned_0.csv', delimiter=',')[:, -1]
    # targets_val_raw = np.loadtxt(targets + name + '/' + name +'_Valence_V_Aligned_0.csv', delimiter=',')[:, -1]
    # targets_ar, targets_val = label_rescale(0.8*targets_ar_raw),  label_rescale(0.8*targets_val_raw)
    # targets_ar = 720*np.ones(np.shape(targets_ar)) -  targets_ar
    
    
    labels= np.load(output_labels + name+'.npz')['arr_0']
    labels_ = label_rescale(1 *labels)
    ar, val = 720*np.ones(np.shape(labels_[:, 20])) - labels_[:, 20], labels_[:, 41]
    ar_raw, val_raw = labels[:, 20], labels[:, 41]
    
    size = np.shape(video)[1], np.shape(video)[2]
    duration = int(np.shape(video)[0]/25)-1

    new_name = str(output_mp4 + '/' + name +'.mp4')
    out = cv2.VideoWriter(new_name, fourcc, fps, (1440, 720), True)
    for i in range(fps * duration):
        num = int(10*np.floor((i/10)))
        # print(num, ar[num])
        circle_dot = cv2.resize(circle, (720, 720))
        circle_dot = cv2.putText(circle_dot, "Pred: Ar:{:.2f}         Val:{:.2f}".format(ar_raw[num],val_raw[num]), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
        # circle_dot = cv2.putText(circle_dot, "True: Ar:{:.2f}         Val:{:.2f}".format(targets_ar_raw[num],targets_val_raw[num]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2, cv2.LINE_AA)
        
        try:
            
            circle_dot[int(ar[num]-16):int(ar[num]+16), int(val[num]-16):int(val[num]+16), :] = red_patch
            # circle_dot[int(targets_ar[num]-16):int(targets_ar[num]+16), int(targets_val[num]-16):int(targets_val[num]+16), :] = green_patch
                        
            video_frame = cv2.resize(video[i, :, :], (720, 720), interpolation = cv2.INTER_CUBIC)
            # video_frame =video[i, :, :].reshape((96, 96, 1))
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)
            # video_frame_ = np.stack((video_frame, video_frame, video_frame), axis=2)[:, :, :, 0]
            # video_frame_.astype('uint8')
            frame = np.concatenate((video_frame, circle_dot), axis = 1)
            # frame = np.concatenate((video_frame, video_frame), axis = 1)
            # frame = cv2.resize(frame, size)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # frame = PIL.Image.fromarray(frame)
            # print(np.shape(frame))
        except:
            video_frame = cv2.resize(video[i, :, :], (720, 720), interpolation = cv2.INTER_CUBIC)
            # video_frame =video[i, :, :].reshape((96, 96, 1))
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)
            # video_frame_ = np.stack((video_frame, video_frame, video_frame), axis=2)[:, :, :, 0]
            # video_frame_.astype('uint8')
            frame = np.concatenate((video_frame, circle_dot), axis = 1)
            # frame = cv2.putText(frame, "%0.2f" % (val[num],), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        out.write(frame)
    out.release()

from mmseg.apis import MMSegInferencer 
import os
import csv
import time
import cv2
import numpy as np
import torch
import func
image = 'https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG'
config = r'weights\202501078卡\pidnet-s_ditantuotan.py'
# checkpoint = r"D:\01item\20CITIC_code\out_models\3\epoch_51.pth"
# checkpoint = r"D:\01item\20CITIC_code\out_models\3\epoch_56.pth"
checkpoint = r"weights\202501078卡\epoch_590.pth"
inferencer = MMSegInferencer(model=config, weights=checkpoint, device='cuda')
def drawlabel(src,mask):
    
    classes = mask.max()
    classnames=['BT','QT']
    colors=[[0,255,0],[0,0,255],[255,255,0]]
    mat_show = draw_seg_result(
        classnames,src, mask, 3, colors, "GT_"
    )
    return mat_show
def qinference(image,input_shape,is_seg,out_x,out_y,classes,overlap):
    resultList=[]
    size = (
                (input_shape[3], input_shape[2])
                if isinstance(input_shape[2], int)
                else None
            )
    if not is_seg:
        # dst = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        # sm.preprocess_input(image, input_space=input_space)
        # image = np.expand_dims(image, axis=0)
        imager = cv2.resize(image, size)
        result = inferencer(imager)
        outImg=result['predictions']
        # outImg = outputData.squeeze().astype("uint8")
        if size:
            # 使用INTER_NEAREST缩放蒙版不会改变RGB值
            outImg = cv2.resize(
                outImg, (out_x, out_y), interpolation=cv2.INTER_NEAREST
            )
        # 去除忽略部分
        outImg = np.where(outImg == classes, 0, outImg)
        return outImg
        # resultList.append(outImg)
    else:
        result_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        st1 = time.time()
        for i, (x, y, subImg) in enumerate(
            func.get_sub_image(image, out_x, out_y, overlap)
        ):
            # image = sm.preprocess_input(dst, input_space=input_space)
            if size:
                subImg = cv2.resize(subImg, size)
            imager = cv2.dnn.blobFromImage(
                subImg,
                scalefactor=1,
                size=size,
                mean=None,
                swapRB=False,
                crop=False,
            )
            # img = np.expand_dims(image.astype('float32'), axis=0)
            result = inferencer(subImg)
            outputData=result['predictions']
            if size:
                result_image[
                    y : y + out_y, x : x + out_x
                ] = cv2.resize(
                    outputData.squeeze().astype("uint8"),
                    (out_x, out_y),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                result_image[
                    y : y + out_y, x : x + out_x
                ] = outputData.squeeze().astype("uint8")
        st2 = time.time()
        print("timehost is {} ms",(st2 - st1) * 1000)
        # 去除忽略部分
        result_image = np.where(result_image == classes, 0, result_image)
        return result_image
        # resultList.append(result_image)
def draw_seg_result(classesNames, src, mask, text_size, colors, prefix=""):
    res_mask = src.copy()
    # res_mask = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    boxes = {}
    classes = mask.max()
    for name in classesNames[:classes]:
        boxes.update({name: []})
    has_ignore = classes == 128
    true_classes = np.where(mask == 128, 0, mask).max()
    for i in range(classes):
        if i >= true_classes and i != 127:
            continue
        if i == 127 and not has_ignore:
            break
        mask_class = (mask == (i + 1)).astype("uint8")
        # cv_imwrite(f"C:/Users/BTW/Desktop/新建文件夹/{i}.png", np.where(mask_class > 0, 255, 0).astype('uint8'))
       
        mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
        mask_color[..., 0] = colors[i][0]
        mask_color[..., 1] = colors[i][1]
        mask_color[..., 2] = colors[i][2]
        # cv2.copyTo(mask_color, mask_class, res_mask)
        
        # cv2.addWeighted(mask_color, 0.7, mask_class, 0.3,0, res_mask)
        # contours, hierarchy = cv2.findContours(
        #     mask_class, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        # )[-2:]
        contours, hierarchy = cv2.findContours(
            mask_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]
        ctrs=[]
        for contour in contours:
            if cv2.contourArea(contour) >40000:
               rect = cv2.minAreaRect(contour)
               box = np.int64(cv2.boxPoints(rect))
            #    draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), thickness=2)

            #    boxes[classesNames[i]].append(cv2.boundingRect(contour))
               boxes[classesNames[i]].append(box)
               ctrs.append(contour)
        if len(ctrs)>0: 
            cv2.drawContours(
                res_mask, ctrs, -1, colors[i], 5
            )
    
        # contours, hierarchy = cv2.findContours(
        #     mask_class, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        # )[-2:]
        # cv2.drawContours(
        #     res_mask, contours, -1, colors[i], 3
        # )
        
        #绘制外接矩形
        # for i, (name, rects) in enumerate(boxes.items()):
        #     color = colors[i] if name != "忽略" else (155, 155, 155)
        #     for rect in rects:
        #         # x, y, w, h = rect
        #         res_mask = cv2.drawContours(res_mask, [box], -1, color, thickness=3)
        #         # cv2.rectangle(
        #         #     res_mask, (x, y), (x + w, y + h), color, 3
        #         # )
       
    return res_mask




# 设置图片文件夹路径
folder_path = r"F:\17jinhengData\02金相\脱碳\val"
 
# 创建一个列表来保存图片路径
image_paths = []
header = ['id', 'label']
# timestamp=time.time()

# #当前本地时间元组

# local_time=time.localtime()

# #当前格林尼治时间元组

# gm_time=time.gmtime()
timestr = time.strftime(r'%Y%m%d%H%M%S',time.localtime(time.time()))#把获取的时间转换成"年月日格式”
print(timestr) 

    # 遍历文件夹以获取所有图片路径
image_paths = []
image_paths=os.listdir(folder_path);
# image_paths.sort(key=lambda x:int(x.split('.')[0]))
classnames=['BT','QT']
colors=[[255,255,0],[0,255,255],[0,255,0]]
text_size=10
for imgpath in image_paths:
    if any(imgpath.endswith(extension) for extension in ['.jpg', '.jpeg', '.gif', '.bmp']):
        imgpath=os.path.join(folder_path, imgpath)
        # image_paths.append(imgpath)
        image = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)
        input_shape=[1,3,1024,1024]
        is_seg=False
        out_x=image.shape[1]
        out_y=image.shape[0]
        classes=2
        overlap=0.2
        # 进行分割
        timestr = time.strftime(r'%Y%m%d%H%M%S%MS',time.localtime(time.time()))#把获取的时间转换成"年月日格式”
        print(timestr) 
        result = qinference(image,input_shape,is_seg,out_x,out_y,classes,overlap)
        timestr = time.strftime(r'%Y%m%d%H%M%S%MS',time.localtime(time.time()))#把获取的时间转换成"年月日格式”
        print(timestr) 
        idname,ext=os.path.splitext(os.path.basename(imgpath))
        gpth=os.path.join(folder_path, "gray")
        gph = os.path.join(gpth,idname + ".jpg")
        gimage = cv2.imdecode(np.fromfile(gph, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        #绘制分割结果
        rcimage=draw_seg_result(classnames,gimage,result,text_size,colors,prefix="")
        #绘制标签
        
       
        ph = os.path.join(folder_path, idname + "_label.png")
        if os.path.exists(ph):
           maskimage = cv2.imdecode(np.fromfile(ph, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
           rcimage=drawlabel(rcimage, maskimage)
        
        #保存图像
        
        
        output_dir = os.path.join(folder_path, 'rsultimg1')
        if not os.path.exists(output_dir):
           os.makedirs(output_dir)
        output_path = os.path.join(output_dir, idname+'.jpg')
       # cv2.imwrite(output_path, image_convol)   
        cv2.imencode(".jpg", rcimage)[1].tofile(output_path) 
        
        
        #彩色图上绘制分割效果////////////////////
        
        #绘制分割结果
        rcimage=draw_seg_result(classnames,image,result,text_size,colors,prefix="")
        #绘制标签
        
       
        ph = os.path.join(folder_path, idname + "_label.png")
        if os.path.exists(ph):
        #    maskimage = cv2.imdecode(np.fromfile(ph, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
           rcimage=drawlabel(rcimage, maskimage)
        
        #保存图像
        
        
        output_dir = os.path.join(folder_path, 'RGBrsultimg')
        if not os.path.exists(output_dir):
           os.makedirs(output_dir)
        output_path = os.path.join(output_dir, idname+'.jpg')
       # cv2.imwrite(output_path, image_convol)   
        cv2.imencode(".jpg", rcimage)[1].tofile(output_path) 
        # print(idname,label)
          

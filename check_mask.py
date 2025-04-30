#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查图像掩码是否包含非零值并进行可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def check_and_visualize_mask(image_path):
    """
    检查图像掩码是否包含非零值并进行可视化
    
    Args:
        image_path (str): 图像路径
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件 {image_path} 不存在!")
        return
    
    # 读取图像
    print(f"正在读取图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return
    
    # 转换为RGB格式用于显示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检查是否为灰度图像或彩色图像
    if len(image.shape) == 3 and image.shape[2] == 3:
        # 彩色图像，转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("图像是彩色图像，已转换为灰度图进行分析")
    else:
        # 已经是灰度图
        gray_image = image
        print("图像是灰度图像")
    
    # 检查非零值
    non_zero_count = np.count_nonzero(gray_image)
    total_pixels = gray_image.size
    non_zero_percentage = (non_zero_count / total_pixels) * 100
    
    print(f"图像尺寸: {image.shape}")
    print(f"非零像素数量: {non_zero_count}")
    print(f"总像素数量: {total_pixels}")
    print(f"非零像素百分比: {non_zero_percentage:.2f}%")
    
    # 获取唯一值及其计数
    unique_values, counts = np.unique(gray_image, return_counts=True)
    print("\n像素值分布:")
    for value, count in zip(unique_values, counts):
        print(f"值 {value}: {count} 像素 ({(count/total_pixels)*100:.2f}%)")
    
    # 创建可视化
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(2, 2, 1)
    plt.title("原始图像")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    # 灰度图像
    plt.subplot(2, 2, 2)
    plt.title("灰度图像")
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    
    # 二值化显示非零区域
    plt.subplot(2, 2, 3)
    plt.title("非零区域 (二值化)")
    binary_mask = (gray_image > 0).astype(np.uint8) * 255
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    
    # 热力图显示像素值分布
    plt.subplot(2, 2, 4)
    plt.title("像素值热力图")
    plt.imshow(gray_image, cmap='viridis')
    plt.colorbar(label='像素值')
    plt.axis('off')
    
    # 保存可视化结果
    output_dir = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}_mask_analysis.png")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n可视化结果已保存至: {output_path}")
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    # 设置图像路径
    image_path = "CHASEDIR/results/Image_11R.png"
    
    # 检查并可视化掩码
    check_and_visualize_mask(image_path) 
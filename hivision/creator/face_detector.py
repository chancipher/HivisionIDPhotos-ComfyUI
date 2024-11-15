#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 19:32
@File: face_detector.py
@IDE: pycharm
@Description:
    人脸检测器
"""
try:
    from mtcnnruntime import MTCNN
except ImportError:
    raise ImportError(
        "Please install mtcnn-runtime by running `pip install mtcnn-runtime`"
    )

from .context import Context
from hivision.error import FaceError
import numpy as np
import cv2
from time import time

mtcnn = None

def detect_face_mtcnn(ctx: Context, scale: int = 2):
    """
    基于MTCNN模型的人脸检测处理器，只进行人脸数量的检测
    :param ctx: 上下文，此时已获取到原始图和抠图结果，但是我们只需要原始图
    :param scale: 最大边长缩放比例，原图:缩放图 = 1:scale
    :raise FaceError: 人脸检测错误，多个人脸或者没有人脸
    """
    global mtcnn
    if mtcnn is None:
        mtcnn = MTCNN()
    image = cv2.resize(
        ctx.origin_image,
        (ctx.origin_image.shape[1] // scale, ctx.origin_image.shape[0] // scale),
        interpolation=cv2.INTER_AREA,
    )
    # landmarks 是 5 个关键点，分别是左眼、右眼、鼻子、左嘴角、右嘴角，
    faces, landmarks = mtcnn.detect(image, thresholds=[0.8, 0.8, 0.8])

    # print(len(faces))
    if len(faces) != 1:
        # 保险措施，如果检测到多个人脸或者没有人脸，用原图再检测一次
        faces, landmarks = mtcnn.detect(ctx.origin_image)
    else:
        # 如果只有一个人脸，将人脸坐标放大
        for item, param in enumerate(faces[0]):
            faces[0][item] = param * 2
    if len(faces) != 1:
        raise FaceError("Expected 1 face, but got {}".format(len(faces)), len(faces))

    # 计算人脸坐标
    left = faces[0][0]
    top = faces[0][1]
    width = faces[0][2] - left + 1
    height = faces[0][3] - top + 1
    ctx.face["rectangle"] = (left, top, width, height)

    # 根据landmarks计算人脸偏转角度，以眼睛为标准，计算的人脸偏转角度，用于人脸矫正
    # 示例landmarks [106.37181  150.77415  127.21012  108.369156 144.61522  105.24723 107.45625  133.62355  151.24269  153.34407 ]
    landmarks = landmarks[0]
    left_eye = np.array([landmarks[0], landmarks[5]])
    right_eye = np.array([landmarks[1], landmarks[6]])
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_angle = np.degrees(np.arctan2(dy, dx))

    ctx.face["roll_angle"] = roll_angle
# -*- coding: utf-8 -*-
##########################################
# @Author: rtzhang
# @Date: 2025-03-15 14:48:24
# @LastEditors: rtzhang
# @LastEditTime: 2025-03-16 14:27:07
# @Description: 用于检查carla原始点云数据
##########################################

import sys
sys.path.append('.')
from options import options  # 导入参数设置
from sceneloader import SceneLoader  # 导入场景加载器
from utils import Visualizer  # 导入可视化工具


if __name__ == '__main__':
    # 设置参数, 也可以在命令行中设置（如python demo0.py --scene_id 1）或者使用options.py的默认参数
    opt = options()
    opt.dataset = 'raw_data'
    opt.scene_id = 0
    opt.preload = False  # 预加载
    opt.preload_begin = 0
    opt.preload_end = 100

    # 加载场景
    scene = SceneLoader(opt)
    print("场景帧数:", scene.frame_num)

    # 获取第0帧点云
    pcd_xyz, _ = scene.get_frame(frame_id=0)

    # 创建可视化工具
    visualizer = Visualizer(opt)

    # 可视化点云
    visualizer.draw_points(pcd_xyz)

    # 动态可视化整个场景
    # 可以使用 空格键 暂停/继续，在暂停状态下可以使用方向键 ← → 或 ↑ ↓ 来控制帧的前进和后退
    visualizer.play_scene(scene, begin=0, delay_time=0)

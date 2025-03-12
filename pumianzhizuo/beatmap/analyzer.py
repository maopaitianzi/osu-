#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
谱面分析器 - 用于分析OSU谱面文件
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional, Union
from PyQt5 import QtCore
import json


class BeatmapAnalyzer(QtCore.QObject):
    """
    谱面分析器类 - 提供OSU谱面分析功能
    
    特点:
    - OSU文件格式解析
    - 谱面难度分析
    - 谱面物件统计
    - 谱面模式识别
    - 物件分布可视化
    """
    
    # 定义信号
    analysis_progress = QtCore.pyqtSignal(int)  # 分析进度信号 (0-100)
    analysis_complete = QtCore.pyqtSignal(dict)  # 分析完成信号，发送结果字典
    analysis_error = QtCore.pyqtSignal(str)  # 分析错误信号
    
    def __init__(self):
        super().__init__()
        self.beatmap_path = ""
        self.beatmap_data = {}  # 存储谱面数据
        self.analysis_results = {}  # 存储分析结果
    
    def load_beatmap(self, file_path: str) -> bool:
        """
        加载OSU谱面文件
        
        参数:
            file_path: OSU文件路径
            
        返回:
            加载是否成功
        """
        try:
            self.beatmap_path = file_path
            self.beatmap_data = self.parse_osu_file(file_path)
            return True
        except Exception as e:
            self.analysis_error.emit(f"加载谱面文件失败: {str(e)}")
            return False
    
    def analyze(self) -> Dict:
        """
        执行完整的谱面分析
        
        返回:
            包含所有分析结果的字典
        """
        if not self.beatmap_data:
            self.analysis_error.emit("请先加载谱面文件")
            return {}
        
        try:
            # 重置分析结果
            self.analysis_results = {}
            
            # 发送进度信号
            self.analysis_progress.emit(5)
            
            # 提取基本信息
            self._extract_basic_info()
            self.analysis_progress.emit(20)
            
            # 分析物件分布
            self._analyze_hit_objects_distribution()
            self.analysis_progress.emit(40)
            
            # 分析难度参数
            self._analyze_difficulty()
            self.analysis_progress.emit(60)
            
            # 分析节奏密度
            self._analyze_rhythm_density()
            self.analysis_progress.emit(80)
            
            # 识别常见谱面模式
            self._identify_patterns()
            self.analysis_progress.emit(95)
            
            # 完成分析
            self.analysis_progress.emit(100)
            self.analysis_complete.emit(self.analysis_results)
            
            return self.analysis_results
            
        except Exception as e:
            self.analysis_error.emit(f"谱面分析过程中出错: {str(e)}")
            return {}
    
    def parse_osu_file(self, osu_path: str) -> Dict:
        """
        解析OSU谱面文件
        
        参数:
            osu_path: OSU文件路径
            
        返回:
            解析后的谱面数据字典
        """
        beatmap = {
            "General": {},
            "Editor": {},
            "Metadata": {},
            "Difficulty": {},
            "Events": [],
            "TimingPoints": [],
            "HitObjects": []
        }
        
        with open(osu_path, "r", encoding="utf-8", errors="ignore") as f:
            current_section = None
            for line in f:
                line = line.strip()
                
                # 跳过空行和注释
                if not line or line.startswith("//"):
                    continue
                
                # 检测区块
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1]
                    continue
                
                # 处理当前区块内容
                if current_section == "General" or current_section == "Editor" or current_section == "Metadata":
                    if ":" in line:
                        key, value = [x.strip() for x in line.split(":", 1)]
                        beatmap[current_section][key] = value
                
                elif current_section == "Difficulty":
                    if ":" in line:
                        key, value = [x.strip() for x in line.split(":", 1)]
                        beatmap[current_section][key] = float(value)
                
                elif current_section == "Events":
                    beatmap["Events"].append(line.split(","))
                
                elif current_section == "TimingPoints":
                    beatmap["TimingPoints"].append(line.split(","))
                
                elif current_section == "HitObjects":
                    parts = line.split(",")
                    if len(parts) >= 5:
                        obj = {
                            "x": int(parts[0]),
                            "y": int(parts[1]),
                            "time": int(parts[2]),
                            "type": int(parts[3]),
                            "hitSound": int(parts[4]),
                            "objectParams": parts[5:] if len(parts) > 5 else []
                        }
                        beatmap["HitObjects"].append(obj)
        
        return beatmap
    
    def _extract_basic_info(self) -> None:
        """提取谱面基本信息"""
        # 元数据信息
        metadata = self.beatmap_data.get("Metadata", {})
        self.analysis_results["metadata"] = {
            "title": metadata.get("Title", "未知标题"),
            "artist": metadata.get("Artist", "未知艺术家"),
            "creator": metadata.get("Creator", "未知作者"),
            "version": metadata.get("Version", "未知难度"),
            "tags": metadata.get("Tags", "").split(),
        }
        
        # 谱面基本信息
        general = self.beatmap_data.get("General", {})
        self.analysis_results["general"] = {
            "audio_filename": general.get("AudioFilename", ""),
            "mode": int(general.get("Mode", 0)),
            "preview_time": int(general.get("PreviewTime", -1))
        }
        
        # 统计物件信息
        hit_objects = self.beatmap_data.get("HitObjects", [])
        circle_count = 0
        slider_count = 0
        spinner_count = 0
        
        for obj in hit_objects:
            obj_type = obj.get("type", 0)
            if obj_type & 1:  # 圆圈
                circle_count += 1
            elif obj_type & 2:  # 滑条
                slider_count += 1
            elif obj_type & 8:  # 转盘
                spinner_count += 1
        
        self.analysis_results["objects_count"] = {
            "total": len(hit_objects),
            "circles": circle_count,
            "sliders": slider_count,
            "spinners": spinner_count
        }
        
        # 时长信息
        if hit_objects:
            first_obj_time = min(obj.get("time", 0) for obj in hit_objects)
            last_obj_time = max(obj.get("time", 0) for obj in hit_objects)
            duration_ms = last_obj_time - first_obj_time
            self.analysis_results["duration"] = {
                "first_object": first_obj_time,
                "last_object": last_obj_time,
                "total_ms": duration_ms,
                "total_seconds": duration_ms / 1000
            }
    
    def _analyze_hit_objects_distribution(self) -> None:
        """分析谱面物件分布"""
        hit_objects = self.beatmap_data.get("HitObjects", [])
        
        if not hit_objects:
            return
        
        # 按时间排序物件
        sorted_objects = sorted(hit_objects, key=lambda x: x.get("time", 0))
        
        # 提取时间点和位置
        times = [obj.get("time", 0) for obj in sorted_objects]
        positions = [(obj.get("x", 0), obj.get("y", 0)) for obj in sorted_objects]
        
        # 计算时间间隔
        time_intervals = []
        for i in range(1, len(times)):
            time_intervals.append(times[i] - times[i-1])
        
        # 计算位置移动距离
        distances = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distances.append(np.sqrt(dx*dx + dy*dy))
        
        # 存储结果
        self.analysis_results["distribution"] = {
            "time_intervals": time_intervals,
            "avg_time_interval": np.mean(time_intervals) if time_intervals else 0,
            "min_time_interval": min(time_intervals) if time_intervals else 0,
            "max_time_interval": max(time_intervals) if time_intervals else 0,
            "distances": distances,
            "avg_distance": np.mean(distances) if distances else 0,
            "max_distance": max(distances) if distances else 0
        }
        
        # 屏幕区域使用热图
        x_coords = [obj.get("x", 0) for obj in hit_objects]
        y_coords = [obj.get("y", 0) for obj in hit_objects]
        
        self.analysis_results["heatmap"] = {
            "x": x_coords,
            "y": y_coords
        }
    
    def _analyze_difficulty(self) -> None:
        """分析谱面难度参数"""
        difficulty = self.beatmap_data.get("Difficulty", {})
        
        # 提取核心难度参数
        ar = difficulty.get("ApproachRate", 5.0)
        od = difficulty.get("OverallDifficulty", 5.0)
        cs = difficulty.get("CircleSize", 4.0)
        hp = difficulty.get("HPDrainRate", 5.0)
        
        # 存储难度信息
        self.analysis_results["difficulty"] = {
            "AR": ar,
            "OD": od,
            "CS": cs,
            "HP": hp,
            "slider_multiplier": difficulty.get("SliderMultiplier", 1.4),
            "slider_tick_rate": difficulty.get("SliderTickRate", 1.0)
        }
        
        # 难度评级
        # 按照osu规则评定难度级别
        diff_rating = (ar + od + cs + hp) / 4
        
        if diff_rating < 3:
            difficulty_level = "Easy"
        elif diff_rating < 5:
            difficulty_level = "Normal"
        elif diff_rating < 7:
            difficulty_level = "Hard"
        elif diff_rating < 8:
            difficulty_level = "Insane"
        else:
            difficulty_level = "Expert"
        
        # 真实难度评级
        ar_rating = "超标" if ar > 9.5 else "偏高" if ar > 7 else "正常" if ar > 4 else "偏低"
        od_rating = "超标" if od > 9 else "偏高" if od > 7 else "正常" if od > 4 else "偏低"
        cs_rating = "超标" if cs > 7 else "偏高" if cs > 5 else "正常" if cs > 3 else "偏低"
        hp_rating = "超标" if hp > 9 else "偏高" if hp > 7 else "正常" if hp > 4 else "偏低"
        
        self.analysis_results["difficulty_rating"] = {
            "overall_level": difficulty_level,
            "numerical_rating": diff_rating,
            "ar_rating": ar_rating,
            "od_rating": od_rating,
            "cs_rating": cs_rating,
            "hp_rating": hp_rating
        }
    
    def _analyze_rhythm_density(self) -> None:
        """分析谱面节奏密度"""
        hit_objects = self.beatmap_data.get("HitObjects", [])
        timing_points = self.beatmap_data.get("TimingPoints", [])
        
        if not hit_objects or not timing_points:
            return
        
        try:
            # 提取主要BPM
            main_bpm = 0
            if timing_points and len(timing_points[0]) >= 2:
                ms_per_beat = float(timing_points[0][1])
                if ms_per_beat > 0:  # 确保是正值，才是真实BPM
                    main_bpm = 60000 / ms_per_beat
            
            # 如果无法提取BPM，则使用默认值
            if main_bpm <= 0:
                main_bpm = 120
            
            # 提取物件时间
            times = [obj.get("time", 0) for obj in hit_objects]
            times.sort()
            
            # 计算谱面总时长（秒）
            total_duration = (times[-1] - times[0]) / 1000 if times else 0
            
            # 计算密度指标
            objects_per_second = len(times) / total_duration if total_duration > 0 else 0
            
            # 转换为每拍物件数
            seconds_per_beat = 60 / main_bpm
            objects_per_beat = objects_per_second * seconds_per_beat
            
            # 根据密度分析难度
            if objects_per_beat < 1:
                density_level = "低密度"
            elif objects_per_beat < 2:
                density_level = "中等密度"
            elif objects_per_beat < 3:
                density_level = "高密度"
            else:
                density_level = "极高密度"
            
            # 计算连打段落
            stream_sections = []
            current_stream = []
            
            for i in range(1, len(times)):
                interval = times[i] - times[i-1]
                beat_fraction = interval / (60000 / main_bpm)
                
                # 如果时间间隔小于等于1/4拍，视为连打
                if beat_fraction <= 0.25:
                    if not current_stream:  # 新的连打段落
                        current_stream = [times[i-1], times[i]]
                    else:  # 延续当前连打
                        current_stream[1] = times[i]
                else:
                    # 当前连打段落结束
                    if current_stream and current_stream[1] - current_stream[0] >= 1000:  # 至少1秒的连打
                        stream_sections.append(current_stream)
                    current_stream = []
            
            # 处理最后一个可能的连打段落
            if current_stream and current_stream[1] - current_stream[0] >= 1000:
                stream_sections.append(current_stream)
            
            # 存储结果
            self.analysis_results["rhythm"] = {
                "bpm": main_bpm,
                "objects_per_second": objects_per_second,
                "objects_per_beat": objects_per_beat,
                "density_level": density_level,
                "stream_sections": stream_sections,
                "stream_sections_count": len(stream_sections)
            }
        
        except Exception as e:
            self.analysis_results["rhythm"] = {
                "error": f"节奏分析出错: {str(e)}"
            }
    
    def _identify_patterns(self) -> None:
        """识别谱面中的常见模式"""
        hit_objects = self.beatmap_data.get("HitObjects", [])
        
        if not hit_objects or len(hit_objects) < 4:
            return
        
        # 按时间排序物件
        sorted_objects = sorted(hit_objects, key=lambda x: x.get("time", 0))
        
        # 提取时间和位置
        times = [obj.get("time", 0) for obj in sorted_objects]
        positions = [(obj.get("x", 0), obj.get("y", 0)) for obj in sorted_objects]
        types = [obj.get("type", 0) for obj in sorted_objects]
        
        # 识别相关模式
        patterns = {
            "jumps": 0,          # 跳跃（两点间距离大）
            "streams": 0,         # 流串（连续密集点击）
            "sliders": 0,         # 滑条
            "spinners": 0,        # 转盘
            "stacks": 0,          # 堆叠（多个物件在同一位置）
            "triangles": 0,       # 三角形模式
            "squares": 0,         # 方形模式
            "back_and_forth": 0,  # 往返模式
            "zigzags": 0          # 之字形模式
        }
        
        # 扫描谱面识别模式
        # 识别跳跃和流串
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance = np.sqrt(dx*dx + dy*dy)
            time_diff = times[i] - times[i-1]
            
            # 跳跃: 距离大且时间较短
            if distance > 200:
                patterns["jumps"] += 1
            
            # 流串: 距离适中且时间非常短
            if distance < 100 and time_diff < 150:
                patterns["streams"] += 1
            
            # 堆叠: 几乎相同位置
            if distance < 5:
                patterns["stacks"] += 1
        
        # 识别常见几何形状模式 (至少需要3-4个点)
        for i in range(len(positions) - 3):
            p1, p2, p3, p4 = positions[i:i+4]
            
            # 三角形: 三个点形成的三边长度相似
            d12 = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            d23 = np.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
            d31 = np.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2)
            
            if abs(d12 - d23) < 30 and abs(d23 - d31) < 30 and abs(d31 - d12) < 30:
                patterns["triangles"] += 1
            
            # 方形: 四个点形成的四边和两个对角线长度特征
            d34 = np.sqrt((p3[0]-p4[0])**2 + (p3[1]-p4[1])**2)
            d41 = np.sqrt((p4[0]-p1[0])**2 + (p4[1]-p1[1])**2)
            d24 = np.sqrt((p2[0]-p4[0])**2 + (p2[1]-p4[1])**2)
            
            if abs(d12 - d34) < 30 and abs(d23 - d41) < 30:
                patterns["squares"] += 1
            
            # 之字形: 连续的转向
            v1 = (p2[0]-p1[0], p2[1]-p1[1])
            v2 = (p3[0]-p2[0], p3[1]-p2[1])
            v3 = (p4[0]-p3[0], p4[1]-p3[1])
            
            # 计算向量点积来判断转向
            dot1 = v1[0]*v2[0] + v1[1]*v2[1]
            dot2 = v2[0]*v3[0] + v2[1]*v3[1]
            
            # 如果两次转向都是锐角或钝角，则可能是之字形
            if (dot1 < 0 and dot2 < 0) or (dot1 > 0 and dot2 > 0):
                patterns["zigzags"] += 1
            
            # 往返: 从一点到另一点再返回
            backtrack = False
            if (abs(p1[0] - p3[0]) < 30 and abs(p1[1] - p3[1]) < 30) or \
               (abs(p2[0] - p4[0]) < 30 and abs(p2[1] - p4[1]) < 30):
                patterns["back_and_forth"] += 1
        
        # 统计滑条和转盘
        for obj_type in types:
            if obj_type & 2:  # 滑条
                patterns["sliders"] += 1
            if obj_type & 8:  # 转盘
                patterns["spinners"] += 1
        
        # 规范化模式计数
        total = sum(patterns.values())
        if total > 0:
            patterns_percentage = {k: v / total * 100 for k, v in patterns.items()}
        else:
            patterns_percentage = {k: 0 for k in patterns}
        
        # 存储结果
        self.analysis_results["patterns"] = {
            "counts": patterns,
            "percentage": patterns_percentage,
            # 确定主要模式
            "primary_pattern": max(patterns.items(), key=lambda x: x[1])[0] if any(patterns.values()) else "未确定"
        }
    
    def generate_heatmap(self, width: int = 800, height: int = 600) -> Figure:
        """
        生成谱面物件分布热图
        
        参数:
            width: 图像宽度
            height: 图像高度
            
        返回:
            matplotlib Figure对象
        """
        if "heatmap" not in self.analysis_results:
            return None
        
        heatmap_data = self.analysis_results["heatmap"]
        x_coords = heatmap_data.get("x", [])
        y_coords = heatmap_data.get("y", [])
        
        if not x_coords or not y_coords:
            return None
        
        fig = Figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        
        # 创建热图
        heatmap = ax.hexbin(x_coords, y_coords, gridsize=20, cmap='hot', alpha=0.7)
        
        # 设置图像标题和轴标签
        ax.set_title('谱面物件分布热图')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        # 添加颜色条
        fig.colorbar(heatmap, ax=ax, label='物件密度')
        
        # 设置坐标范围，与osu游戏区域一致
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 384)
        
        # 反转Y轴，因为osu坐标系的Y轴是向下增加的
        ax.invert_yaxis()
        
        fig.tight_layout()
        return fig
    
    def generate_timing_distribution(self, width: int = 800, height: int = 600) -> Figure:
        """
        生成谱面时间分布图
        
        参数:
            width: 图像宽度
            height: 图像高度
            
        返回:
            matplotlib Figure对象
        """
        if "distribution" not in self.analysis_results:
            return None
        
        distribution_data = self.analysis_results["distribution"]
        time_intervals = distribution_data.get("time_intervals", [])
        
        if not time_intervals:
            return None
        
        fig = Figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        
        # 创建时间间隔直方图
        ax.hist(time_intervals, bins=50, alpha=0.7, color='blue')
        
        # 设置图像标题和轴标签
        ax.set_title('物件时间间隔分布')
        ax.set_xlabel('时间间隔 (ms)')
        ax.set_ylabel('频率')
        
        # 添加均值线
        mean_interval = distribution_data.get("avg_time_interval", 0)
        ax.axvline(mean_interval, color='red', linestyle='dashed', linewidth=1)
        ax.text(mean_interval*1.05, ax.get_ylim()[1]*0.9, f'均值: {mean_interval:.1f}ms', color='red')
        
        fig.tight_layout()
        return fig
    
    def get_difficulty_summary(self) -> str:
        """
        获取谱面难度概要，作为人类可读的文本
        
        返回:
            难度概要文本
        """
        if "difficulty" not in self.analysis_results or "difficulty_rating" not in self.analysis_results:
            return "无法生成难度概要。谱面尚未分析。"
        
        difficulty = self.analysis_results["difficulty"]
        rating = self.analysis_results["difficulty_rating"]
        
        summary = f"""谱面难度概要:
标称难度等级: {rating['overall_level']}
难度数值评分: {rating['numerical_rating']:.2f}/10

参数评级:
- 接近速度(AR): {difficulty['AR']:.1f} - {rating['ar_rating']}
- 判定精度(OD): {difficulty['OD']:.1f} - {rating['od_rating']}
- 圆圈大小(CS): {difficulty['CS']:.1f} - {rating['cs_rating']}
- 生命消耗(HP): {difficulty['HP']:.1f} - {rating['hp_rating']}

对于难度{self.analysis_results['metadata']['version']}:
"""
        
        # 根据osu谱面分析标准，给出参数是否符合规范的建议
        version = self.analysis_results['metadata']['version'].lower()
        
        if "easy" in version:
            if difficulty['AR'] >= 5:
                summary += "• AR值偏高，Easy难度推荐AR<5\n"
            if difficulty['OD'] > 3:
                summary += "• OD值偏高，Easy难度推荐OD在1-3之间\n"
            if difficulty['HP'] > 3:
                summary += "• HP值偏高，Easy难度推荐HP在1-3之间\n"
        
        elif "normal" in version:
            if difficulty['AR'] < 4 or difficulty['AR'] > 6:
                summary += "• AR值不在推荐范围，Normal难度推荐AR在4-6之间\n"
            if difficulty['OD'] < 3 or difficulty['OD'] > 5:
                summary += "• OD值不在推荐范围，Normal难度推荐OD在3-5之间\n"
            if difficulty['HP'] < 3 or difficulty['HP'] > 5:
                summary += "• HP值不在推荐范围，Normal难度推荐HP在3-5之间\n"
        
        elif "hard" in version:
            if difficulty['AR'] < 6 or difficulty['AR'] > 8:
                summary += "• AR值不在推荐范围，Hard难度推荐AR在6-8之间\n"
            if difficulty['OD'] < 5 or difficulty['OD'] > 7:
                summary += "• OD值不在推荐范围，Hard难度推荐OD在5-7之间\n"
            if difficulty['HP'] < 4 or difficulty['HP'] > 6:
                summary += "• HP值不在推荐范围，Hard难度推荐HP在4-6之间\n"
        
        elif "insane" in version or "expert" in version:
            if difficulty['AR'] < 7:
                summary += "• AR值偏低，Insane/Expert难度推荐AR>=7\n"
            if difficulty['OD'] < 7:
                summary += "• OD值偏低，Insane/Expert难度推荐OD>=7\n"
        
        # 添加节奏密度信息
        if "rhythm" in self.analysis_results:
            rhythm = self.analysis_results["rhythm"]
            summary += f"\n节奏分析:\n- BPM: {rhythm.get('bpm', 0):.1f}\n"
            summary += f"- 密度等级: {rhythm.get('density_level', '未知')}\n"
            summary += f"- 每秒物件数: {rhythm.get('objects_per_second', 0):.2f}\n"
            summary += f"- 每拍物件数: {rhythm.get('objects_per_beat', 0):.2f}\n"
            summary += f"- 连打段落数: {rhythm.get('stream_sections_count', 0)}\n"
        
        # 添加物件统计
        if "objects_count" in self.analysis_results:
            objects = self.analysis_results["objects_count"]
            summary += f"\n物件统计:\n- 总数: {objects.get('total', 0)}\n"
            summary += f"- 圆圈: {objects.get('circles', 0)} ({objects.get('circles', 0)/max(1, objects.get('total', 1))*100:.1f}%)\n"
            summary += f"- 滑条: {objects.get('sliders', 0)} ({objects.get('sliders', 0)/max(1, objects.get('total', 1))*100:.1f}%)\n"
            summary += f"- 转盘: {objects.get('spinners', 0)} ({objects.get('spinners', 0)/max(1, objects.get('total', 1))*100:.1f}%)\n"
        
        # 添加模式信息
        if "patterns" in self.analysis_results and "primary_pattern" in self.analysis_results["patterns"]:
            patterns = self.analysis_results["patterns"]
            summary += f"\n主要模式: {patterns['primary_pattern']}\n"
        
        return summary
    
if __name__ == "__main__":
    # 简单测试
    analyzer = BeatmapAnalyzer()
    analyzer.load_beatmap("test.osu")
    analyzer.analyze()
    print(analyzer.get_difficulty_summary()) 
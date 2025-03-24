import os
import json
import random
import numpy as np
from datetime import datetime

class BeatmapGenerator:
    """谱面生成器，根据音频分析结果生成osu谱面"""
    
    def __init__(self):
        # 基础谱面参数
        self.ar = 5.0  # 接近速度
        self.od = 5.0  # 总体难度
        self.hp = 5.0  # 血量消耗
        self.cs = 4.0  # 圆圈大小
        self.slider_multiplier = 1.4  # 滑条速度倍数
        self.slider_tick_rate = 1  # 滑条点密度
        
        # 谱面元数据
        self.title = "未命名"
        self.artist = "未知艺术家"
        self.creator = "AI谱面生成器"
        self.version = "Normal"  # 难度名
        
        # 生成参数
        self.density = 10  # 谱面密度，1-10
        self.use_model = False  # 是否使用模型优化摆放
        self.model_path = None  # 模型文件路径
        
        # 事件选择概率
        self.beat_selection_probability = 0.3  # 节拍点选择概率（默认30%）
        self.onset_selection_probability = 0.7  # 起始点选择概率（默认70%）
        
        # 谱面数据
        self.hit_objects = []  # 谱面物件列表
        self.timing_points = []  # 时间点列表
        self.beat_length = 500  # 默认beat长度(ms)
        self.offset = 0  # 偏移量
        
        # 分析数据
        self.analysis_data_map = {}  # 各轨道的分析数据
        
        # 播放区域尺寸
        self.playfield_width = 512
        self.playfield_height = 384
        
        # 滑条类型映射
        self.slider_types = {
            'linear': 'L',  # 直线滑条
            'bezier': 'B',  # 贝塞尔滑条
            'perfect': 'P',  # 完美曲线滑条
            'catmull': 'C'   # Catmull曲线滑条
        }
        
        # 音效类型映射
        self.hitsound_types = {
            'normal': 0,    # 默认音效
            'whistle': 2,   # 口哨音效
            'finish': 4,    # 结束音效
            'clap': 8       # 拍手音效
        }
        
        # 节拍强度判定阈值
        self.strong_beat_threshold = 0.7  # 强拍阈值
        self.medium_beat_threshold = 0.4  # 次强拍阈值
        
    def set_metadata(self, title, artist, creator, version):
        """设置谱面元数据"""
        self.title = title
        self.artist = artist
        self.creator = creator
        self.version = version
        
    def set_difficulty(self, ar, od, hp, cs):
        """设置谱面难度参数"""
        self.ar = ar / 10.0
        self.od = od / 10.0
        self.hp = hp / 10.0
        self.cs = cs / 10.0
        
    def set_generation_params(self, density, use_model=False, model_path=None):
        """设置生成参数"""
        self.density = density
        self.use_model = use_model
        self.model_path = model_path
        
    def set_event_selection_probabilities(self, beat_prob=0.3, onset_prob=0.7):
        """
        设置事件选择概率
        
        参数:
            beat_prob: 节拍点选择概率 (0.0-1.0)
            onset_prob: 起始点选择概率 (0.0-1.0)
        """
        # 确保概率值在有效范围内
        self.beat_selection_probability = max(0.0, min(1.0, beat_prob))
        self.onset_selection_probability = max(0.0, min(1.0, onset_prob))
        
    def set_beat_strength_thresholds(self, strong_threshold=0.7, medium_threshold=0.4):
        """
        设置节拍强度判定阈值
        
        参数:
            strong_threshold: 强拍阈值 (0.0-1.0)，高于此值为强拍
            medium_threshold: 次强拍阈值 (0.0-1.0)，高于此值为次强拍
        """
        # 确保阈值在有效范围内
        self.strong_beat_threshold = max(0.0, min(1.0, strong_threshold))
        self.medium_beat_threshold = max(0.0, min(1.0, medium_threshold))
        # 确保强拍阈值大于次强拍阈值
        if self.medium_beat_threshold >= self.strong_beat_threshold:
            self.medium_beat_threshold = self.strong_beat_threshold * 0.6
        
    def load_analysis_data(self, analysis_data_map):
        """加载音频分析数据"""
        self.analysis_data_map = analysis_data_map
        
        # 提取BPM和偏移量信息
        for source_id, source_info in analysis_data_map.items():
            data = source_info["data"]
            
            # 获取BPM
            if "bpm" in data:
                bpm = data["bpm"]
                self.beat_length = 60000 / bpm  # 转换BPM为beat长度(ms)
            
            # 获取偏移量
            if "beat_times" in data and len(data["beat_times"]) > 0:
                self.offset = data["beat_times"][0] * 1000  # 转为毫秒
                
            # 找到一个有效源后就可以退出了，优先使用优先级高的源
            break
        
    def _get_random_position(self):
        """获取随机位置"""
        margin = 10  # 距离边缘的安全距离
        x = random.randint(margin, self.playfield_width - margin)
        y = random.randint(margin, self.playfield_height - margin)
        return x, y
    
    def _get_model_optimized_position(self, time_ms, prev_position=None):
        """使用模型优化物件位置（当前为随机位置，待实现）"""
        # TODO: 实现模型优化摆放
        return self._get_random_position()
    
    def _is_long_sound(self, onset_time, source_data):
        """判断是否为长音"""
        # 通过检查声音持续时间判断是否为长音
        # 可以根据强度或频谱特征进行判断
        
        # 如果有音符持续时间信息
        if "note_durations" in source_data:
            durations = source_data["note_durations"]
            for start, duration in durations:
                if abs(start - onset_time) < 0.05:  # 50ms以内认为是同一时间点
                    return duration > 0.15  # 持续时间大于150ms视为长音
        
        # 如果有频谱持续时间信息
        if "spectral_flux" in source_data:
            flux = source_data["spectral_flux"]
            onset_index = int(onset_time * 100)  # 假设采样率为100Hz
            if onset_index < len(flux) - 10:
                # 检查后续10个采样点是否有足够的能量
                sustained_energy = sum(flux[onset_index:onset_index+10]) > 0.5
                return sustained_energy
        
        # 默认短音
        return False
    
    def _get_beat_strength_type(self, time, source_data):
        """
        判断拍子强度类型，通过音量包络和节拍强度来确定
        返回: 
            - "strong": 强拍
            - "medium": 次强拍
            - "weak": 弱拍
        """
        # 默认为弱拍
        beat_type = "weak"
        
        # 优先使用鼓声音量包络进行判定（深粉色线）
        # 注意：确保使用的是音量包络（深粉色线），而非波形图（浅粉色线）
        drums_envelope = None
        
        # 首先查找专门的鼓声音量envelope
        if "drums_volume" in source_data:
            drums_envelope = source_data["drums_volume"]
        elif "drums_envelope" in source_data:
            drums_envelope = source_data["drums_envelope"]
        elif "spectral_drums" in source_data and "energy" in source_data["spectral_drums"]:
            drums_envelope = source_data["spectral_drums"]["energy"]
        
        # 如果没有找到鼓声专用的音量数据，再尝试使用常规音量包络
        if drums_envelope is None:
            if "volume_envelope" in source_data:
                drums_envelope = source_data["volume_envelope"]
            elif "spectral_contrast" in source_data:
                # 使用频谱对比度的低频段作为鼓声能量估计
                if isinstance(source_data["spectral_contrast"], dict) and "sub_band" in source_data["spectral_contrast"]:
                    drums_envelope = source_data["spectral_contrast"]["sub_band"]
        
        # 根据音量包络判断拍子强度
        if drums_envelope is not None:
            volume = None
            
            # 字典格式的音量包络（带时间和值）
            if isinstance(drums_envelope, dict) and "times" in drums_envelope and "values" in drums_envelope:
                times = drums_envelope["times"]
                values = drums_envelope["values"]
                
                # 找到最接近的时间点
                closest_idx = None
                min_diff = 0.1  # 100ms以内认为是同一时间点
                
                for i, env_time in enumerate(times):
                    diff = abs(time - env_time)
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = i
                
                # 获取对应的音量值
                if closest_idx is not None and closest_idx < len(values):
                    volume = values[closest_idx]
            
            # 列表格式的音量包络（均匀采样）
            elif isinstance(drums_envelope, list) and len(drums_envelope) > 0:
                if "duration" in source_data:
                    duration = source_data["duration"]
                    sample_rate = len(drums_envelope) / duration
                    
                    # 计算时间对应的索引
                    time_idx = int(time * sample_rate)
                    if 0 <= time_idx < len(drums_envelope):
                        volume = drums_envelope[time_idx]
            
            # 根据音量值判断拍子强度
            if volume is not None:
                # 使用可配置的阈值
                if volume > self.strong_beat_threshold:
                    beat_type = "strong"
                elif volume > self.medium_beat_threshold:
                    beat_type = "medium"
                # 否则保持弱拍
                
                # 打印调试信息，帮助用户确认判定结果
                # print(f"时间点: {time:.3f}s, 音量值: {volume:.5f}, 判定为: {beat_type}")
        
        # 如果没有音量包络或未能确定强度，回退到使用节拍强度信息
        if beat_type == "weak" and "beat_times" in source_data and "beat_strength" in source_data:
            beat_times = source_data["beat_times"]
            beat_strength = source_data["beat_strength"]
            
            # 找到最接近的节拍
            closest_idx = None
            min_diff = 0.1  # 100ms以内认为是同一拍
            
            for i, beat_time in enumerate(beat_times):
                diff = abs(time - beat_time)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            
            # 如果找到匹配的节拍，判断强度
            if closest_idx is not None and closest_idx < len(beat_strength):
                strength = beat_strength[closest_idx]
                
                # 使用可配置的阈值
                if strength > self.strong_beat_threshold:
                    beat_type = "strong"
                elif strength > self.medium_beat_threshold:
                    beat_type = "medium"
                # 其他情况保持弱拍
                
                # 也检查这个节拍是否在强拍列表中
                if "strong_beats" in source_data and closest_idx in source_data.get("strong_beats", []):
                    beat_type = "strong"
        
        return beat_type
    
    def _generate_slider(self, start_time, end_time, position, source=None):
        """生成滑条"""
        x, y = position
        
        # 滑条持续时间
        duration_beats = (end_time - start_time) / self.beat_length
        
        # 通过持续时间和滑条速度计算长度
        length = duration_beats * self.slider_multiplier * 100
        
        # 随机选择滑条类型，偏好贝塞尔曲线
        slider_type = random.choice([self.slider_types['bezier'], 
                                    self.slider_types['bezier'], 
                                    self.slider_types['linear'], 
                                    self.slider_types['perfect']])
        
        # 生成随机控制点
        if slider_type == self.slider_types['linear']:
            # 直线滑条，只需要一个终点
            end_x = min(max(x + random.randint(-100, 100), 0), self.playfield_width)
            end_y = min(max(y + random.randint(-100, 100), 0), self.playfield_height)
            control_points = f"|{end_x}:{end_y}"
        elif slider_type == self.slider_types['bezier']:
            # 贝塞尔曲线，需要2-3个控制点
            points_count = random.randint(2, 3)
            points = []
            for _ in range(points_count):
                point_x = min(max(x + random.randint(-100, 100), 0), self.playfield_width)
                point_y = min(max(y + random.randint(-100, 100), 0), self.playfield_height)
                points.append(f"{point_x}:{point_y}")
            control_points = "|" + "|".join(points)
        elif slider_type == self.slider_types['perfect']:
            # 完美曲线，需要2个点形成圆弧
            point1_x = min(max(x + random.randint(-80, 80), 0), self.playfield_width)
            point1_y = min(max(y + random.randint(-80, 80), 0), self.playfield_height)
            point2_x = min(max(x + random.randint(-80, 80), 0), self.playfield_width)
            point2_y = min(max(y + random.randint(-80, 80), 0), self.playfield_height)
            control_points = f"|{point1_x}:{point1_y}|{point2_x}:{point2_y}"
        
        # 设置音效
        hitsound = 0
        addition = "0:0:0:0:"
        
        # 根据来源和拍子强度设置音效
        if source == "drums":
            # 获取当前拍子的强度类型
            source_data = {}
            for src_id, src_info in self.analysis_data_map.items():
                if src_id == source:
                    source_data = src_info["data"]
                    break
            
            beat_type = self._get_beat_strength_type(start_time/1000, source_data)
            
            # 根据拍子强度设置不同的音效
            if beat_type == "strong":
                # 强拍: clap + whistle (移除finish音效)
                hitsound = self.hitsound_types['clap'] + self.hitsound_types['whistle']
                # 强拍音量100%
                addition = "0:0:0:100:"
            elif beat_type == "medium":
                # 次强拍: clap
                hitsound = self.hitsound_types['clap']
                # 次强拍音量50%
                addition = "0:0:0:80:"
            else:
                # 弱拍: whistle
                hitsound = self.hitsound_types['whistle']
        
        # 构建滑条对象
        # 格式：x,y,time,type,hitSound,curveType|curvePoints,slides,length,edgeHitsound,edgeAddition,hitSample
        hit_object = f"{x},{y},{int(start_time)},2,{hitsound},{slider_type}{control_points},1,{int(length)},0:0|0:0,{addition}"
        
        return hit_object
    
    def _generate_circle(self, time, position, source=None):
        """生成圆圈"""
        x, y = position
        
        # 设置音效
        hitsound = 0
        addition = "0:0:0:0:"
        
        # 根据来源和拍子强度设置音效
        if source == "drums":
            # 获取当前拍子的强度类型
            source_data = {}
            for src_id, src_info in self.analysis_data_map.items():
                if src_id == source:
                    source_data = src_info["data"]
                    break
            
            beat_type = self._get_beat_strength_type(time/1000, source_data)
            
            # 根据拍子强度设置不同的音效
            if beat_type == "strong":
                # 强拍: clap + whistle (移除finish音效)
                hitsound = self.hitsound_types['clap'] + self.hitsound_types['whistle']
                # 强拍音量100%
                addition = "0:0:0:100:"
            elif beat_type == "medium":
                # 次强拍: clap
                hitsound = self.hitsound_types['clap']
                # 次强拍音量50%
                addition = "0:0:0:50:"
            else:
                # 弱拍: whistle
                hitsound = self.hitsound_types['whistle']
            
        # 格式：x,y,time,type,hitSound,hitSample
        hit_object = f"{x},{y},{int(time)},1,{hitsound},{addition}"
        return hit_object
    
    def generate_beatmap(self):
        """生成谱面"""
        # 清空现有物件
        self.hit_objects = []
        
        # 获取按优先级排序的轨道
        sorted_sources = sorted(
            [(source_id, source_info) for source_id, source_info in self.analysis_data_map.items()],
            key=lambda x: x[1].get("priority", 0),
            reverse=True
        )
        
        # 合并所有音频事件
        all_events = []
        for source_id, source_info in sorted_sources:
            source_data = source_info["data"]
            priority = source_info.get("priority", 0)
            
            # 从不同特征中提取音符事件
            events = []
            
            # 从节拍时间提取事件
            if "beat_times" in source_data:
                beat_times = source_data["beat_times"]
                for time in beat_times:
                    if random.random() < self.beat_selection_probability * (self.density / 5.0):  # 使用用户自定义概率
                        events.append({
                            "time": time,
                            "priority": priority,
                            "is_beat": True,
                            "source": source_id
                        })
            
            # 从音符起始时间提取事件
            if "onset_times" in source_data:
                onset_times = source_data["onset_times"]
                for time in onset_times:
                    # 使用用户自定义概率
                    if random.random() < self.onset_selection_probability * (self.density / 5.0):
                        is_long = self._is_long_sound(time, source_data)
                        
                        # 如果是长音，查找结束时间
                        end_time = None
                        if is_long and "note_durations" in source_data:
                            for start, duration in source_data["note_durations"]:
                                if abs(start - time) < 0.05:  # 50ms以内认为是同一时间点
                                    end_time = start + duration
                                    break
                        
                        events.append({
                            "time": time,
                            "end_time": end_time,
                            "priority": priority,
                            "is_long": is_long,
                            "is_beat": False,
                            "source": source_id
                        })
            
            # 添加到所有事件列表
            all_events.extend(events)
        
        # 按时间排序所有事件
        all_events.sort(key=lambda x: x["time"])
        
        # 过滤掉太近的事件（根据密度调整）
        filtered_events = []
        min_time_diff = max(0.1, 0.3 - 0.025 * self.density)  # 密度越高，最小时间差越小
        
        for i, event in enumerate(all_events):
            # 第一个事件直接添加
            if i == 0:
                filtered_events.append(event)
                continue
                
            # 检查与上一个事件的时间差
            prev_event = filtered_events[-1]
            if event["time"] - prev_event["time"] >= min_time_diff:
                # 时间差足够大，直接添加
                filtered_events.append(event)
            else:
                # 时间差太小，优先保留优先级高的事件
                if event["priority"] > prev_event["priority"]:
                    # 当前事件优先级更高，替换上一个事件
                    filtered_events[-1] = event
        
        # 生成物件
        prev_position = None
        for event in filtered_events:
            time_ms = int(event["time"] * 1000)  # 转换为毫秒
            
            # 获取物件位置
            if self.use_model and self.model_path:
                position = self._get_model_optimized_position(time_ms, prev_position)
            else:
                position = self._get_random_position()
            
            # 根据事件类型生成不同物件
            if event.get("is_long", False) and event.get("end_time"):
                # 生成滑条
                end_time_ms = int(event["end_time"] * 1000)
                hit_object = self._generate_slider(time_ms, end_time_ms, position, event.get("source"))
            else:
                # 生成圆圈
                hit_object = self._generate_circle(time_ms, position, event.get("source"))
            
            self.hit_objects.append(hit_object)
            prev_position = position
        
        # 生成时间点
        self.generate_timing_points()
        
        return self.hit_objects
    
    def generate_timing_points(self):
        """生成时间点"""
        self.timing_points = []
        
        # 添加主时间点
        main_point = f"{self.offset},{self.beat_length},4,0,0,100,1,0"
        self.timing_points.append(main_point)
        
        return self.timing_points
    
    def generate_osu_file(self, output_path):
        """生成完整的osu文件"""
        # 确保目标目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 生成谱面物件（如果尚未生成）
        if not self.hit_objects:
            self.generate_beatmap()
        
        # 构建osu文件内容
        osu_content = []
        
        # 添加文件头
        osu_content.append("osu file format v14")
        osu_content.append("")
        
        # 添加General部分
        osu_content.append("[General]")
        osu_content.append("AudioFilename: audio.mp3")
        osu_content.append("AudioLeadIn: 0")
        osu_content.append("PreviewTime: -1")
        osu_content.append("Countdown: 0")
        osu_content.append("SampleSet: Normal")
        osu_content.append("StackLeniency: 0.7")
        osu_content.append("Mode: 0")
        osu_content.append("LetterboxInBreaks: 0")
        osu_content.append("WidescreenStoryboard: 0")
        osu_content.append("")
        
        # 添加Editor部分
        osu_content.append("[Editor]")
        osu_content.append("DistanceSpacing: 1")
        osu_content.append("BeatDivisor: 4")
        osu_content.append("GridSize: 32")
        osu_content.append("TimelineZoom: 1")
        osu_content.append("")
        
        # 添加Metadata部分
        osu_content.append("[Metadata]")
        osu_content.append(f"Title:{self.title}")
        osu_content.append(f"TitleUnicode:{self.title}")
        osu_content.append(f"Artist:{self.artist}")
        osu_content.append(f"ArtistUnicode:{self.artist}")
        osu_content.append(f"Creator:{self.creator}")
        osu_content.append(f"Version:{self.version}")
        osu_content.append("Source:")
        osu_content.append("Tags:AI generated")
        osu_content.append(f"BeatmapID:0")
        osu_content.append(f"BeatmapSetID:-1")
        osu_content.append("")
        
        # 添加Difficulty部分
        osu_content.append("[Difficulty]")
        osu_content.append(f"HPDrainRate:{self.hp}")
        osu_content.append(f"CircleSize:{self.cs}")
        osu_content.append(f"OverallDifficulty:{self.od}")
        osu_content.append(f"ApproachRate:{self.ar}")
        osu_content.append(f"SliderMultiplier:{self.slider_multiplier}")
        osu_content.append(f"SliderTickRate:{self.slider_tick_rate}")
        osu_content.append("")
        
        # 添加Events部分
        osu_content.append("[Events]")
        osu_content.append("//Background and Video events")
        osu_content.append("//Break Periods")
        osu_content.append("//Storyboard Layer 0 (Background)")
        osu_content.append("//Storyboard Layer 1 (Fail)")
        osu_content.append("//Storyboard Layer 2 (Pass)")
        osu_content.append("//Storyboard Layer 3 (Foreground)")
        osu_content.append("//Storyboard Layer 4 (Overlay)")
        osu_content.append("//Storyboard Sound Samples")
        osu_content.append("")
        
        # 添加TimingPoints部分
        osu_content.append("[TimingPoints]")
        for point in self.timing_points:
            osu_content.append(point)
        osu_content.append("")
        
        # 添加Colours部分
        osu_content.append("[Colours]")
        osu_content.append("Combo1 : 255,128,192")
        osu_content.append("Combo2 : 128,255,255")
        osu_content.append("Combo3 : 128,192,255")
        osu_content.append("Combo4 : 192,128,255")
        osu_content.append("")
        
        # 添加HitObjects部分
        osu_content.append("[HitObjects]")
        for obj in self.hit_objects:
            osu_content.append(obj)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(osu_content))
        
        return output_path 
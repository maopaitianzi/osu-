#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频分析器 - 用于高级音频特征提取和分析
"""

import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PyQt5 import QtCore

# 添加GPU加速相关导入
import torch

# 检查GPU是否可用
GPU_AVAILABLE = torch.cuda.is_available()


class AudioAnalyzer(QtCore.QObject):
    """
    高级音频分析器类 - 提供各种音频分析功能
    
    特点:
    - 多种BPM检测算法
    - 节拍网格生成
    - 音量/能量变化分析
    - 多种频谱特征提取
    - 音频段落检测
    - 过渡点检测（适合osu谱面关键点）
    - GPU加速支持（需要CUDA环境）
    """
    
    # 定义信号
    analysis_progress = QtCore.pyqtSignal(int)  # 分析进度信号 (0-100)
    analysis_complete = QtCore.pyqtSignal(dict)  # 分析完成信号，发送结果字典
    analysis_error = QtCore.pyqtSignal(str)  # 分析错误信号
    
    def __init__(self, use_gpu=False):
        super().__init__()
        self.audio_path = ""
        self.y = None  # 音频数据
        self.sr = None  # 采样率
        self.features = {}  # 存储提取的特征
        
        # 分析参数
        self.hop_length = 512
        self.onset_envelope = None
        self.tempo = None
        self.beats = None
        
        # BPM设置相关
        self.manual_bpm = None  # 手动设置的BPM
        self.bpm_source = "auto"  # BPM来源：auto, manual, beatmap
        
        # GPU设置
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    def load_audio(self, file_path: str) -> bool:
        """
        加载音频文件
        
        参数:
            file_path: 音频文件路径
            
        返回:
            加载是否成功
        """
        try:
            self.audio_path = file_path
            self.y, self.sr = librosa.load(file_path, sr=None)
            return True
        except Exception as e:
            self.analysis_error.emit(f"加载音频文件失败: {str(e)}")
            return False
    
    def set_use_gpu(self, use_gpu: bool) -> None:
        """
        设置是否使用GPU加速
        
        参数:
            use_gpu: 是否使用GPU
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    def analyze(self) -> Dict:
        """
        执行完整的音频分析，包括所有特征
        
        返回:
            包含所有分析结果的字典
        """
        if self.y is None or self.sr is None:
            self.analysis_error.emit("请先加载音频文件")
            return {}
        
        try:
            # 重置特征字典
            self.features = {}
            
            # 发送进度信号
            self.analysis_progress.emit(5)
            
            # 提取基本信息
            self.features["duration"] = librosa.get_duration(y=self.y, sr=self.sr)
            self.features["sample_rate"] = self.sr
            
            # 检测BPM和节拍
            self.analysis_progress.emit(10)
            self._detect_tempo_and_beats()
            
            # 提取节拍强度
            self.analysis_progress.emit(30)
            self._extract_beat_strength()
            
            # 提取频谱特征
            self.analysis_progress.emit(50)
            self._extract_spectral_features()
            
            # 提取音量包络
            self.analysis_progress.emit(70)
            self._extract_volume_envelope()
            
            # 检测段落和过渡点
            self.analysis_progress.emit(85)
            self._detect_sections_and_transitions()
            
            # 创建节拍网格（适合osu谱面）
            self.analysis_progress.emit(95)
            self._create_beat_grid()
            
            # 完成分析
            self.analysis_progress.emit(100)
            self.analysis_complete.emit(self.features)
            
            return self.features
            
        except Exception as e:
            self.analysis_error.emit(f"音频分析过程中出错: {str(e)}")
            return {}
    
    def _to_gpu(self, data):
        """将数据转移到GPU上（如果启用了GPU加速）"""
        if self.use_gpu:
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).to(self.device)
            elif isinstance(data, torch.Tensor):
                return data.to(self.device)
        return data
    
    def _to_cpu(self, data):
        """将数据从GPU转回CPU"""
        if self.use_gpu and isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, torch.Tensor):
            return data.numpy()
        return data
    
    def _detect_tempo_and_beats(self) -> None:
        """检测BPM和节拍位置"""
        # 如果有手动设置的BPM，优先使用
        if self.manual_bpm is not None and self.bpm_source != "auto":
            self.tempo = self.manual_bpm
            
            # 计算onset强度包络
            self.onset_envelope = librosa.onset.onset_strength(
                y=self.y, sr=self.sr, hop_length=self.hop_length
            )
            
            # 使用手动BPM进行节拍检测
            ac = librosa.autocorrelate(self.onset_envelope, max_size=2 * self.sr // self.hop_length)
            tempo = float(self.manual_bpm)
            
            # 计算每个节拍的帧数
            frames_per_beat = int(60.0 * self.sr / (tempo * self.hop_length))
            
            # 使用动态规划寻找最佳的节拍位置
            beats = librosa.util.peak_pick(self.onset_envelope, 
                                          pre_max=frames_per_beat // 2, 
                                          post_max=frames_per_beat // 2, 
                                          pre_avg=frames_per_beat, 
                                          post_avg=frames_per_beat, 
                                          delta=0.2, 
                                          wait=frames_per_beat // 2)
            
            # 如果没有检测到足够的节拍，使用等间隔的节拍
            if len(beats) < 4:
                # 估计第一个节拍的位置
                start_frame = np.argmax(self.onset_envelope[:frames_per_beat*2])
                
                # 生成均匀的节拍序列
                num_beats = int(len(self.onset_envelope) / frames_per_beat)
                beats = np.arange(start_frame, start_frame + num_beats * frames_per_beat, frames_per_beat)
                
                # 限制在有效范围内
                beats = beats[beats < len(self.onset_envelope)]
        else:
            # 计算onset强度包络
            self.onset_envelope = librosa.onset.onset_strength(
                y=self.y, sr=self.sr, hop_length=self.hop_length
            )
            
            # 如果启用GPU，使用GPU加速
            if self.use_gpu:
                # 将onset_envelope转移到GPU
                onset_env_gpu = self._to_gpu(self.onset_envelope)
                
                # 使用GPU进行FFT计算
                # 注意：实际实现需要替换为torch的实现
                # 这里简化为CPU计算后再转回GPU
                # TODO: 使用torch替换librosa的实现
                
                # 暂时回到CPU执行，因为librosa不直接支持GPU
                onset_env_cpu = self._to_cpu(onset_env_gpu)
                tempo, beats = librosa.beat.beat_track(
                    onset_envelope=onset_env_cpu, 
                    sr=self.sr,
                    hop_length=self.hop_length,
                    trim=False
                )
                
                # 将节拍回到GPU
                beats_gpu = self._to_gpu(beats)
                # 然后再转回CPU存储结果
                beats = self._to_cpu(beats_gpu)
            else:
                # 使用CPU执行原始算法
                tempo, beats = librosa.beat.beat_track(
                    onset_envelope=self.onset_envelope, 
                    sr=self.sr,
                    hop_length=self.hop_length,
                    trim=False
                )
            
            # 算法2: 使用有调谐范围的节拍跟踪
            tempo_range = librosa.beat.tempo(
                onset_envelope=self.onset_envelope,
                sr=self.sr,
                hop_length=self.hop_length,
                aggregate=None
            )
            
            # 如果检测到多个tempo候选，取最明显的
            if len(tempo_range) > 0:
                # 将tempo四舍五入到整数
                tempo_candidates = [int(round(t)) for t in tempo_range]
                
                # 尝试找出更可能的BPM（通常在60-180范围内）
                filtered_tempi = [t for t in tempo_candidates if 60 <= t <= 240]
                
                if filtered_tempi:
                    # 取最常见的值作为可能的BPM
                    from collections import Counter
                    tempo_counts = Counter(filtered_tempi)
                    self.tempo = tempo_counts.most_common(1)[0][0]
                else:
                    self.tempo = int(round(tempo))
            else:
                self.tempo = int(round(tempo))
            
            # 更新BPM来源
            self.bpm_source = "auto"
        
        # 将节拍时间转换为秒
        self.beats = librosa.frames_to_time(beats, sr=self.sr, hop_length=self.hop_length)
        
        # 存储结果
        self.features["bpm"] = self.tempo
        self.features["beat_times"] = self.beats.tolist()
        
        # 如果是手动设置的BPM，也保存原始的自动检测结果
        if self.bpm_source != "auto" and "original" not in self.features:
            # 存储原始自动检测的信息
            self.features["original"] = {
                "bpm_source": "auto",
                "beat_times": self.beats.tolist()
            }
            
        # 保存BPM来源
        self.features["bpm_source"] = self.bpm_source
        
        # 高级分析：检测节奏规律性
        if len(self.beats) > 1:
            beat_intervals = np.diff(self.beats)
            regularity = 1.0 - np.std(beat_intervals) / np.mean(beat_intervals)
            self.features["beat_regularity"] = float(max(0, regularity))  # 转换为float以便JSON序列化
    
    def _extract_beat_strength(self) -> None:
        """提取每个节拍的强度"""
        if self.onset_envelope is None or self.beats is None:
            return
        
        # 将节拍时间转换回帧
        beat_frames = librosa.time_to_frames(self.beats, sr=self.sr, hop_length=self.hop_length)
        
        # 确保所有帧都在有效范围内
        valid_frames = [f for f in beat_frames if f < len(self.onset_envelope)]
        
        # 获取每个节拍点的强度
        beat_strengths = self.onset_envelope[valid_frames]
        
        # 归一化强度值到0-1范围
        if len(beat_strengths) > 0:
            max_strength = np.max(beat_strengths)
            if max_strength > 0:
                normalized_strengths = beat_strengths / max_strength
            else:
                normalized_strengths = beat_strengths
            
            # 存储结果
            self.features["beat_strengths"] = normalized_strengths.tolist()
            
            # 检测强拍位置（强度大于均值的节拍）
            mean_strength = np.mean(normalized_strengths)
            strong_beats = [
                i for i, strength in enumerate(normalized_strengths) 
                if strength > mean_strength * 1.2
            ]
            self.features["strong_beats"] = strong_beats
    
    def _extract_spectral_features(self) -> None:
        """提取频谱特征，使用GPU加速（如可用）"""
        if self.use_gpu:
            # 将音频数据转移到GPU
            y_gpu = self._to_gpu(self.y)
            
            # 使用GPU进行FFT（需要在torch中实现，这里展示概念）
            # 由于librosa不直接支持GPU，这里需要使用torch的函数
            # 注意：以下是概念展示，实际实现会有所不同
            
            # 1. 使用torch实现STFT
            # 转换为浮点数以避免精度问题
            y_float = y_gpu.float() if isinstance(y_gpu, torch.Tensor) else torch.tensor(self.y, dtype=torch.float32, device=self.device)
            
            # 配置STFT参数 (与librosa兼容)
            n_fft = 2048
            win_length = n_fft
            window = torch.hann_window(win_length, device=self.device)
            
            # 计算STFT
            D_gpu = torch.stft(
                y_float, 
                n_fft=n_fft, 
                hop_length=self.hop_length, 
                win_length=win_length, 
                window=window, 
                return_complex=True
            )
            
            # 计算频谱幅度
            magnitude_gpu = torch.abs(D_gpu)
            
            # 转回CPU进行后续处理（因为librosa的高级特征提取仍需CPU）
            magnitude = self._to_cpu(magnitude_gpu)
            
            # 由于后续处理仍使用librosa，需要暂时回到CPU
            # 未来可以完全用torch替代librosa实现GPU端完整处理
            mel_spec = librosa.feature.melspectrogram(
                y=self.y, sr=self.sr, hop_length=self.hop_length, n_mels=128
            )
        else:
            # 原始CPU实现
            # 计算短时傅里叶变换 (STFT)
            D = librosa.stft(self.y, hop_length=self.hop_length)
            
            # 计算频谱幅度
            magnitude = np.abs(D)
            
            # 计算梅尔频谱
            mel_spec = librosa.feature.melspectrogram(
                y=self.y, sr=self.sr, hop_length=self.hop_length, n_mels=128
            )
        
        # 转换为分贝单位
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 计算色度图 (适合音符/和弦检测)
        chroma = librosa.feature.chroma_stft(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )
        
        # 提取梅尔频谱对应的时间轴
        times = librosa.times_like(mel_spec, sr=self.sr, hop_length=self.hop_length)
        
        # 计算频谱质心 (表示声音的"亮度")
        spectral_centroids = librosa.feature.spectral_centroid(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )[0]
        
        # 计算频谱对比度 (高频与低频能量比)
        spectral_contrast = librosa.feature.spectral_contrast(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )
        
        # 计算色度能量归一化 (更好地表示和弦)
        chroma_cens = librosa.feature.chroma_cens(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )
        
        # 存储关键频谱特征
        self.features["spectral"] = {
            "times": times.tolist(),
            "centroid": spectral_centroids.tolist(),
            "contrast_mean": np.mean(spectral_contrast, axis=1).tolist(),
            "chroma_mean": np.mean(chroma, axis=1).tolist(),
        }
        
        # 存储用于可视化的完整频谱数据
        # 注意: 这些数据量较大，仅用于可视化
        self.features["visualization"] = {
            "mel_spec_db": mel_spec_db.tolist(),
            "chroma": chroma.tolist()
        }
    
    def _extract_volume_envelope(self) -> None:
        """提取音量包络"""
        # 计算RMS能量
        rms = librosa.feature.rms(y=self.y, hop_length=self.hop_length)[0]
        
        # 获取对应的时间轴
        times = librosa.times_like(rms, sr=self.sr, hop_length=self.hop_length)
        
        # 平滑RMS曲线 (使用移动平均)
        window_size = 5
        if len(rms) > window_size:
            smoothed_rms = np.convolve(rms, np.ones(window_size)/window_size, mode='valid')
            # 确保时间和RMS长度匹配
            smoothed_times = times[window_size-1:]
            if len(smoothed_times) > len(smoothed_rms):
                smoothed_times = smoothed_times[:len(smoothed_rms)]
            elif len(smoothed_times) < len(smoothed_rms):
                smoothed_rms = smoothed_rms[:len(smoothed_times)]
        else:
            smoothed_rms = rms
            smoothed_times = times
        
        # 存储音量包络数据
        self.features["volume"] = {
            "times": smoothed_times.tolist(),
            "rms": smoothed_rms.tolist()
        }
        
        # 检测音量突变点（可能的过渡点）
        if len(smoothed_rms) > 1:
            # 计算RMS的一阶差分
            rms_diff = np.diff(smoothed_rms)
            
            # 找出超过阈值的变化点
            threshold = np.std(rms_diff) * 2
            change_points = np.where(np.abs(rms_diff) > threshold)[0]
            
            # 转换为时间点
            if len(change_points) > 0:
                change_times = smoothed_times[change_points].tolist()
                self.features["volume_changes"] = change_times
    
    def _detect_sections_and_transitions(self) -> None:
        """检测音频段落和过渡点"""
        if "spectral" not in self.features:
            return
        
        try:
            # 使用谱平面图进行结构分段
            # 首先获得一个自相似矩阵
            mfcc = librosa.feature.mfcc(
                y=self.y, sr=self.sr, hop_length=self.hop_length, n_mfcc=13
            )
            
            # 标准化MFCC特征
            mfcc = librosa.util.normalize(mfcc, axis=1)
            
            # 计算自相似矩阵
            similarity = librosa.segment.recurrence_matrix(
                mfcc, mode='affinity', sym=True
            )
            
            # 使用光谱聚类检测段落边界
            boundaries = librosa.segment.agglomerative(similarity, 10)
            boundary_times = librosa.frames_to_time(boundaries, sr=self.sr, hop_length=self.hop_length)
            
            # 进一步细化边界
            refined_boundaries = []
            min_section_length = 5.0  # 最小段落长度（秒）
            prev_time = 0
            
            for time in boundary_times:
                if time - prev_time >= min_section_length:
                    refined_boundaries.append(float(time))
                    prev_time = time
            
            # 存储段落边界
            self.features["sections"] = refined_boundaries
            
            # 检测过渡点（节拍+音量+频谱变化的组合）
            transitions = []
            
            # 如果有检测到节拍和音量变化
            if "beat_times" in self.features and "volume_changes" in self.features:
                beat_times = self.features["beat_times"]
                volume_changes = self.features["volume_changes"]
                
                # 找出在音量变化附近的节拍点
                for beat in beat_times:
                    # 查找最接近的音量变化点
                    closest_changes = [
                        change for change in volume_changes 
                        if abs(beat - change) < 0.1  # 100ms阈值
                    ]
                    
                    if closest_changes:
                        transitions.append(float(beat))
            
            # 去除过于密集的过渡点
            if transitions:
                filtered_transitions = [transitions[0]]
                min_gap = 1.0  # 最小间隔1秒
                
                for t in transitions[1:]:
                    if t - filtered_transitions[-1] >= min_gap:
                        filtered_transitions.append(t)
                
                self.features["transitions"] = filtered_transitions
        
        except Exception as e:
            # 如果段落检测失败，记录错误但继续执行其他分析
            self.features["section_detection_error"] = str(e)
    
    def _create_beat_grid(self) -> None:
        """创建用于osu谱面的节拍网格"""
        if self.tempo is None or self.beats is None or len(self.beats) == 0:
            return
        
        try:
            # 计算节拍间隔（秒）
            beat_interval = 60.0 / self.tempo
            
            # 估计开始偏移
            offset = self.beats[0]
            
            # 创建一个理想的节拍网格
            duration = self.features["duration"]
            num_beats = int(duration / beat_interval) + 1
            ideal_grid = np.arange(num_beats) * beat_interval + offset
            
            # 限制在音频长度内
            ideal_grid = ideal_grid[ideal_grid < duration]
            
            # 存储节拍网格
            self.features["beat_grid"] = ideal_grid.tolist()
            
            # 对每个实际节拍，找到最接近的网格节拍
            grid_mapped_beats = []
            for beat in self.beats:
                # 找到最接近的网格节拍
                closest_grid_idx = np.argmin(np.abs(ideal_grid - beat))
                grid_mapped_beats.append(float(ideal_grid[closest_grid_idx]))
            
            # 存储映射后的节拍
            self.features["grid_mapped_beats"] = grid_mapped_beats
            
            # 存储节拍网格参数 (用于osu谱面)
            self.features["osu"] = {
                "bpm": self.tempo,
                "offset": float(offset * 1000),  # osu用毫秒
                "beat_divisor": 4,  # 默认4分音符
                "time_signature": [4, 4]  # 默认4/4拍
            }
        
        except Exception as e:
            # 如果节拍网格创建失败，记录错误但继续执行其他分析
            self.features["beat_grid_error"] = str(e)
    
    def get_beat_times(self) -> List[float]:
        """获取节拍时间点列表"""
        if "beat_times" in self.features:
            return self.features["beat_times"]
        return []
    
    def get_bpm(self) -> float:
        """获取检测到的BPM"""
        if "bpm" in self.features:
            return self.features["bpm"]
        return 0.0
    
    def get_bpm_source(self) -> str:
        """获取BPM的来源"""
        return self.bpm_source
    
    def set_manual_bpm(self, bpm: float) -> None:
        """
        手动设置BPM值
        
        参数:
            bpm: 手动设置的BPM值
        """
        try:
            bpm_value = float(bpm)
            if bpm_value <= 0:
                self.analysis_error.emit("BPM必须大于0")
                return
                
            self.manual_bpm = bpm_value
            self.bpm_source = "manual"
            
            # 更新features字典中的BPM
            if self.features:
                self.features["bpm"] = bpm_value
                self.features["bpm_source"] = "manual"
                
            return True
        except (ValueError, TypeError):
            self.analysis_error.emit(f"无效的BPM值: {bpm}")
            return False
            
    def import_beatmap_bpm(self, beatmap_path: str) -> bool:
        """
        从osu谱面文件导入BPM
        
        参数:
            beatmap_path: osu谱面文件的路径
            
        返回:
            导入是否成功
        """
        try:
            if not os.path.exists(beatmap_path):
                self.analysis_error.emit(f"谱面文件不存在: {beatmap_path}")
                return False
                
            # 读取谱面文件
            with open(beatmap_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 查找TimingPoints部分
            timing_section = False
            timing_points = []
            
            for line in lines:
                line = line.strip()
                
                if line == "[TimingPoints]":
                    timing_section = True
                    continue
                    
                if timing_section and line.startswith("["):
                    timing_section = False
                    break
                    
                if timing_section and line and not line.startswith("//"):
                    timing_points.append(line.split(","))
            
            # 查找主要BPM点
            main_bpm = 0
            for point in timing_points:
                if len(point) >= 2:
                    ms_per_beat = float(point[1])
                    if ms_per_beat > 0:  # 正值表示实际BPM，而不是继承的BPM
                        main_bpm = 60000 / ms_per_beat
                        break
            
            if main_bpm <= 0:
                self.analysis_error.emit("无法从谱面中提取有效的BPM")
                return False
                
            # 设置BPM
            self.manual_bpm = main_bpm
            self.bpm_source = "beatmap"
            
            # 更新features字典中的BPM
            if self.features:
                self.features["bpm"] = main_bpm
                self.features["bpm_source"] = "beatmap"
                
            return True
        except Exception as e:
            self.analysis_error.emit(f"从谱面导入BPM失败: {str(e)}")
            return False
    
    def get_osu_timing_points(self) -> List[Tuple[float, float]]:
        """
        获取osu格式的timing points
        
        返回:
            timing points列表，每项包含 (时间点(ms), 毫秒每节拍)
        """
        if "osu" not in self.features:
            return []
        
        osu_data = self.features["osu"]
        offset_ms = osu_data["offset"]
        bpm = osu_data["bpm"]
        
        # 计算毫秒每节拍 (osu!格式)
        ms_per_beat = 60000.0 / bpm
        
        # 创建基本timing point
        timing_points = [(offset_ms, ms_per_beat)]
        
        # 如果有段落，为每个段落添加timing point
        if "sections" in self.features:
            for section_time in self.features["sections"]:
                # 添加一个继承前一个timing point的新timing point (以负值表示)
                timing_points.append((section_time * 1000, -100.0))
        
        return timing_points
    
    def get_energy_points(self, threshold: float = 0.75) -> List[float]:
        """
        获取能量高于阈值的时间点，适合放置osu谱面中的物件
        
        参数:
            threshold: 能量阈值 (0-1)
            
        返回:
            高能量时间点列表
        """
        if "volume" not in self.features:
            return []
        
        volume_data = self.features["volume"]
        times = volume_data["times"]
        rms = volume_data["rms"]
        
        # 归一化RMS
        max_rms = max(rms)
        if max_rms > 0:
            normalized_rms = [r / max_rms for r in rms]
        else:
            normalized_rms = rms
        
        # 找出高于阈值的时间点
        energy_points = [
            times[i] for i in range(len(times))
            if normalized_rms[i] > threshold
        ]
        
        return energy_points
    
    def get_density_suggestion(self) -> Dict:
        """
        根据音频特征，建议osu谱面的密度设置
        
        返回:
            包含密度建议的字典
        """
        result = {
            "stream_density": 0.5,  # 默认值
            "jump_intensity": 0.5,  # 默认值
            "slider_velocity": 1.0   # 默认值
        }
        
        if not self.features:
            return result
        
        # 根据BPM调整流串密度
        if "bpm" in self.features:
            bpm = self.features["bpm"]
            # BPM较高时增加流串密度
            if bpm > 180:
                result["stream_density"] = min(0.8, 0.5 + (bpm - 180) / 100)
            elif bpm < 120:
                result["stream_density"] = max(0.3, 0.5 - (120 - bpm) / 100)
        
        # 根据频谱质心调整跳跃强度
        if "spectral" in self.features and "centroid" in self.features["spectral"]:
            centroids = self.features["spectral"]["centroid"]
            mean_centroid = np.mean(centroids)
            # 标准化到0-1范围 (假设大多数音频的质心在500-8000Hz之间)
            normalized_centroid = min(1.0, max(0.0, (mean_centroid - 500) / 7500))
            result["jump_intensity"] = normalized_centroid
        
        # 根据节拍规律性调整滑条速度
        if "beat_regularity" in self.features:
            regularity = self.features["beat_regularity"]
            # 规律性高的曲目可以有更高的滑条速度
            result["slider_velocity"] = 1.0 + regularity
        
        return result
    
    def generate_preview_image(self, feature_type: str = "waveform") -> Optional[Figure]:
        """
        生成指定类型的音频特征预览图像
        
        参数:
            feature_type: 预览类型，可选 "waveform", "mel_spectrogram", "chroma"
            
        返回:
            matplotlib Figure对象，或None（如果生成失败）
        """
        if self.y is None or self.sr is None:
            return None
        
        try:
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(10, 4))
            
            if feature_type == "waveform":
                # 绘制波形图
                plt.plot(librosa.times_like(self.y, sr=self.sr), self.y, color='#FF66AA')
                plt.title("音频波形", fontsize=14)
                plt.xlabel("时间 (秒)", fontsize=12)
                plt.ylabel("振幅", fontsize=12)
                
                # 如果有节拍点，在波形上标出
                if "beat_times" in self.features:
                    for beat in self.features["beat_times"]:
                        plt.axvline(x=beat, color='w', alpha=0.2)
            
            elif feature_type == "mel_spectrogram":
                if "visualization" in self.features and "mel_spec_db" in self.features["visualization"]:
                    # 绘制梅尔频谱图
                    mel_spec_db = np.array(self.features["visualization"]["mel_spec_db"])
                    plt.imshow(mel_spec_db, aspect='auto', origin='lower', interpolation='nearest', cmap='magma')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title("梅尔频谱图", fontsize=14)
                    plt.xlabel("时间 (帧)", fontsize=12)
                    plt.ylabel("梅尔频率", fontsize=12)
                else:
                    # 实时计算梅尔频谱
                    mel_spec = librosa.feature.melspectrogram(y=self.y, sr=self.sr, hop_length=self.hop_length)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    plt.imshow(mel_spec_db, aspect='auto', origin='lower', interpolation='nearest', cmap='magma')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title("梅尔频谱图", fontsize=14)
                    plt.xlabel("时间 (帧)", fontsize=12)
                    plt.ylabel("梅尔频率", fontsize=12)
            
            elif feature_type == "chroma":
                if "visualization" in self.features and "chroma" in self.features["visualization"]:
                    # 绘制色度图
                    chroma = np.array(self.features["visualization"]["chroma"])
                    plt.imshow(chroma, aspect='auto', origin='lower', interpolation='nearest', cmap='plasma')
                    plt.colorbar()
                    plt.title("色度图", fontsize=14)
                    plt.xlabel("时间 (帧)", fontsize=12)
                    plt.ylabel("音高", fontsize=12)
                    plt.yticks(np.arange(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
                else:
                    # 实时计算色度图
                    chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr, hop_length=self.hop_length)
                    plt.imshow(chroma, aspect='auto', origin='lower', interpolation='nearest', cmap='plasma')
                    plt.colorbar()
                    plt.title("色度图", fontsize=14)
                    plt.xlabel("时间 (帧)", fontsize=12)
                    plt.ylabel("音高", fontsize=12)
                    plt.yticks(np.arange(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"生成预览图像时出错: {str(e)}")
            return None
    
    def export_analysis_to_json(self, output_path: Optional[str] = None) -> str:
        """
        将分析结果导出为JSON文件
        
        参数:
            output_path: 输出文件路径，如果为None，则使用音频文件路径+.analysis.json
            
        返回:
            保存的文件路径
        """
        import json
        
        if not self.features:
            self.analysis_error.emit("没有可导出的分析结果")
            return ""
        
        if output_path is None:
            if self.audio_path:
                output_path = os.path.splitext(self.audio_path)[0] + ".analysis.json"
            else:
                output_path = "audio_analysis.json"
        
        try:
            # 移除大型可视化数据以减小文件大小
            export_data = self.features.copy()
            if "visualization" in export_data:
                del export_data["visualization"]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return output_path
        
        except Exception as e:
            self.analysis_error.emit(f"导出分析结果时出错: {str(e)}")
            return "" 
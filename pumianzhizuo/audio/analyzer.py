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
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PyQt5 import QtCore
import json
import datetime
import copy

# 导入scipy模块
import scipy.signal
from scipy.signal import windows

# 为了兼容性，将windows.hann函数添加到scipy.signal命名空间
scipy.signal.hann = windows.hann

# 添加GPU加速相关导入
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入人声分离相关库
from demucs.pretrained import get_model
from demucs.apply import apply_model
import tqdm

# 检查GPU是否可用
GPU_AVAILABLE = torch.cuda.is_available()

# 导入高级音频分离模型 (MelBand RoFormer)
try:
    import melband_roformer
    from melband_roformer.models import MelBandRoFormer
    from melband_roformer.processor import AudioProcessor
    MELBAND_AVAILABLE = True
except ImportError:
    MELBAND_AVAILABLE = False

# 导入SCNet XL （如果可用）
try:
    import scnetxl
    from scnetxl.models import SCNetXL
    SCNETXL_AVAILABLE = True
except ImportError:
    SCNETXL_AVAILABLE = False

# 自定义JSON编码器，处理numpy数组和其他不可序列化的对象
class NumpyEncoder(json.JSONEncoder):
    """处理Numpy数组的JSON编码器"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用于检测循环引用
        self.seen = set()
        
    def default(self, obj):
        # 检测循环引用
        obj_id = id(obj)
        if obj_id in self.seen:
            return "<circular reference detected>"
        self.seen.add(obj_id)
        
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, (np.complex, np.complex64, np.complex128)):
                return {"real": obj.real, "imag": obj.imag}
            return super().default(obj)
        finally:
            # 完成对当前对象的处理后，从seen集合中移除
            self.seen.remove(obj_id)

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
    - 人声分离功能（支持多种模型：Demucs v4, MelBand RoFormer, SCNet XL）
    """
    
    # 定义信号
    analysis_progress = QtCore.pyqtSignal(int)  # 分析进度信号 (0-100)
    analysis_complete = QtCore.pyqtSignal(dict)  # 分析完成信号，发送结果字典
    analysis_error = QtCore.pyqtSignal(str)  # 分析错误信号
    
    # 定义音频源优先级选项 - 这是各个音频源的标准名称
    AUDIO_SOURCES = ["vocals", "drums", "bass", "other"]
    
    # Demucs模型实际输出顺序
    DEMUCS_SOURCE_ORDER = ["drums", "bass", "other", "vocals"]
    
    # 默认优先级
    DEFAULT_PRIORITY = ["vocals", "drums", "bass", "other"]
    
    # 定义可用的人声分离模型
    SEPARATION_MODELS = {
        "demucs_v4": "Demucs v4 (标准)",
        "htdemucs": "Demucs HT (混合变体)",
        "htdemucs_ft": "Demucs HT 微调版",
    }
    
    # 如果高级模型可用，添加到选项中
    if MELBAND_AVAILABLE:
        SEPARATION_MODELS["melband_roformer"] = "MelBand RoFormer (高性能)"
    
    if SCNETXL_AVAILABLE:
        SEPARATION_MODELS["scnetxl"] = "SCNet XL (高质量)"
    
    def __init__(self, use_gpu=False):
        """
        初始化音频分析器
        
        参数:
            use_gpu: 是否使用GPU加速（如果可用）
        """
        super().__init__()
        
        # 音频数据
        self.y = None  # 原始音频数据
        self.sr = None  # 采样率
        self.file_path = None  # 音频文件路径
        self.hop_length = 512  # 默认帧移动长度
        
        # 分析结果
        self.bpm = None  # 检测到的BPM
        self.manual_bpm = None  # 手动设置的BPM
        self.tempo = None  # 临时存储的tempo值
        self.beat_times = None  # 节拍时间点
        self.beat_strength = None  # 节拍强度
        self.strong_beats = None  # 强拍位置
        self.beat_confidence = None  # 节拍检测置信度
        self.beat_source = "default"  # BPM来源
        self.bpm_source = "auto"  # 兼容旧代码
        
        # 频谱特征
        self.spectral_features = {}  # 频谱特征
        
        # 音量和段落
        self.onset_envelope = None  # 音量起始包络
        self.volume_envelope = {}  # 音量包络
        self.volume_changes = []  # 音量变化点
        self.sections = []  # 段落边界
        self.transitions = []  # 过渡点
        
        # 节拍网格
        self.beat_grid = []  # 理想节拍网格
        self.grid_mapped_beats = []  # 映射到网格的节拍
        self.osu_params = {}  # osu谱面参数
        
        # GPU支持
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # 人声分离
        self.use_source_separation = False  # 是否使用源分离
        self.demucs_model = None  # Demucs模型
        self.melband_model = None  # MelBand RoFormer模型
        self.scnetxl_model = None  # SCNet XL模型
        self.current_model = "htdemucs"  # 当前使用的模型名称
        self.active_source = "original"  # 当前活跃的音频源
        self.source_priority = self.DEFAULT_PRIORITY.copy()  # 源优先级
        self.separated_sources = {}  # 分离后的音频源
        
        # 线程控制
        self._is_running = False  # 分析是否正在运行
        
        # 如果启用GPU，初始化GPU相关设置
        if self.use_gpu:
            self._init_gpu()
    
    def _init_gpu(self):
        """初始化GPU相关资源"""
        try:
            self.device = torch.device("cuda")
            # 设置一些CUDA优化参数
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            # 打印GPU信息，便于调试
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"初始化GPU: {gpu_name}, 内存: {gpu_mem:.2f}GB")
        except Exception as e:
            print(f"GPU初始化失败: {str(e)}")
            self.use_gpu = False
            self.device = torch.device("cpu")
    
    def load_audio(self, file_path: str) -> bool:
        """
        加载音频文件
        
        参数:
            file_path: 音频文件路径
            
        返回:
            bool: 是否成功加载
        """
        try:
            self.file_path = file_path
            self.y, self.sr = librosa.load(file_path, sr=None)
            self.active_source = "original"  # 设置当前活跃源为原始音频
            
            # 重置之前的分析结果
            self.bpm = None
            self.beat_times = None
            self.spectral_features = {}
            self.separated_sources = {}
            
            # 预先加载Demucs模型（如果启用了人声分离）
            if self.use_source_separation and self.demucs_model is None:
                try:
                    self._load_demucs_model()
                except Exception as e:
                    self.analysis_error.emit(f"无法加载人声分离模型: {str(e)}")
            
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
            self._init_gpu()
        else:
            self.device = torch.device("cpu")
    
    def analyze(self) -> Dict:
        """
        进行全面音频分析
        
        返回:
            dict: 分析结果字典
        """
        # 检查是否有加载音频
        if self.y is None or self.sr is None:
            self.analysis_error.emit("没有加载音频数据，无法进行分析")
            return {}
        
        # 如果启用了音频源分离，先进行分离
        if self.use_source_separation and not self.separated_sources:
            try:
                self._separate_audio_sources()
                # 根据优先级选择活跃源
                self._select_active_source()
            except Exception as e:
                self.analysis_error.emit(f"音频源分离失败: {str(e)}")
                # 回退到使用原始音频
                self.active_source = "original"
        
        # 分析所有音频源（包括原始音频和分离的源）
        all_results = {}
        sources_to_analyze = []
        
        # 添加所有需要分析的源
        if self.separated_sources:
            sources_to_analyze = list(self.separated_sources.keys())
        sources_to_analyze.append("original")
        
        # 记住原始活跃源
        original_active_source = self.active_source
        
        # 发出起始进度信号
        self.analysis_progress.emit(0)
        
        # 依次分析每个源
        total_sources = len(sources_to_analyze)
        for i, source_name in enumerate(sources_to_analyze):
            # 更新状态
            progress_base = int((i / total_sources) * 100)
            self.analysis_progress.emit(progress_base)
            
            # 设置当前活跃源
            self.set_active_source(source_name)
            
            # 获取当前源的音频数据
            analysis_data = self._get_active_audio_data()
            
            try:
                # 初始化结果字典
                result = {
                    "file_path": self.file_path,
                    "sample_rate": self.sr,
                    "duration": len(analysis_data) / self.sr,
                    "active_source": source_name
                }
                
                # 检测BPM和节拍
                self._detect_tempo_and_beats()
                self.analysis_progress.emit(progress_base + int((1/total_sources) * 20))
                
                # 提取节拍强度
                self._extract_beat_strength()
                self.analysis_progress.emit(progress_base + int((1/total_sources) * 30))
                
                # 提取频谱特征
                self._extract_spectral_features()
                self.analysis_progress.emit(progress_base + int((1/total_sources) * 50))
                
                # 提取音量包络
                self._extract_volume_envelope()
                self.analysis_progress.emit(progress_base + int((1/total_sources) * 60))
                
                # 检测段落和过渡点
                self._detect_sections_and_transitions()
                self.analysis_progress.emit(progress_base + int((1/total_sources) * 80))
                
                # 创建节拍网格
                self._create_beat_grid()
                self.analysis_progress.emit(progress_base + int((1/total_sources) * 90))
                
                # 更新结果字典，添加分析结果
                result.update({
                    "bpm": self.bpm,
                    "beat_source": self.beat_source,
                    "beat_times": self.beat_times.tolist() if self.beat_times is not None else None,
                    "beat_strength": self.beat_strength.tolist() if self.beat_strength is not None else None,
                    "beat_confidence": self.beat_confidence,
                    "spectral_features": self.spectral_features,
                    "volume_envelope": self.volume_envelope if hasattr(self, 'volume_envelope') else None,
                    "volume_changes": self.volume_changes if hasattr(self, 'volume_changes') else None,
                    "sections": self.sections if hasattr(self, 'sections') else None,
                    "transitions": self.transitions if hasattr(self, 'transitions') else None,
                    "beat_grid": self.beat_grid if hasattr(self, 'beat_grid') else None,
                    "grid_mapped_beats": self.grid_mapped_beats if hasattr(self, 'grid_mapped_beats') else None,
                    "osu_params": self.osu_params if hasattr(self, 'osu_params') else None,
                })
                
                # 保存到所有结果中
                all_results[source_name] = result
                
            except Exception as e:
                self.analysis_error.emit(f"分析音频源 {source_name} 过程中发生错误: {str(e)}")
                # 继续分析下一个源
        
        # 恢复原始活跃源
        self.set_active_source(original_active_source)
        
        # 创建最终结果 - 使用当前活跃源的分析结果作为基础
        active_source_result = all_results.get(original_active_source, {}).copy()
        
        # 添加所有源的分析和可用源信息（避免循环引用）
        active_source_result["all_sources_analysis"] = all_results
        active_source_result["available_sources"] = list(self.separated_sources.keys()) + ["original"] if self.separated_sources else ["original"]
            
        # 发出完成信号
        self.analysis_progress.emit(100)
        self.analysis_complete.emit(active_source_result)
        
        return active_source_result
    
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
        # 获取当前活跃的音频数据
        y = self._get_active_audio_data()
        
        # 如果有手动设置的BPM，优先使用
        if hasattr(self, 'manual_bpm') and self.manual_bpm is not None and hasattr(self, 'bpm_source') and self.bpm_source != "auto":
            self.tempo = self.manual_bpm
            
            # 计算onset强度包络
            self.onset_envelope = librosa.onset.onset_strength(
                y=y, sr=self.sr, hop_length=self.hop_length
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
                y=y, sr=self.sr, hop_length=self.hop_length
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
        self.bpm = self.tempo
        self.beat_times = self.beats
        self.beat_source = self.bpm_source if hasattr(self, 'bpm_source') else "default"
        
        # 计算节拍检测置信度
        if len(self.beat_times) > 1:
            beat_intervals = np.diff(self.beat_times)
            regularity = 1.0 - np.std(beat_intervals) / np.mean(beat_intervals)
            self.beat_confidence = float(max(0, regularity))
        else:
            self.beat_confidence = 0.0
    
    def _extract_beat_strength(self) -> None:
        """提取每个节拍的强度"""
        if self.onset_envelope is None or self.beat_times is None:
            return
        
        # 将节拍时间转换回帧
        beat_frames = librosa.time_to_frames(self.beat_times, sr=self.sr, hop_length=self.hop_length)
        
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
            self.beat_strength = normalized_strengths
            
            # 检测强拍位置（强度大于均值的节拍）
            mean_strength = np.mean(normalized_strengths)
            strong_beats = [
                i for i, strength in enumerate(normalized_strengths) 
                if strength > mean_strength * 1.2
            ]
            self.strong_beats = strong_beats
    
    def _extract_spectral_features(self) -> None:
        """提取频谱特征，使用GPU加速（如可用）"""
        # 获取当前活跃的音频数据
        y = self._get_active_audio_data()
        
        if self.use_gpu:
            # 将音频数据转移到GPU
            y_gpu = self._to_gpu(y)
            
            # 使用GPU进行FFT（需要在torch中实现，这里展示概念）
            # 由于librosa不直接支持GPU，这里需要使用torch的函数
            # 注意：以下是概念展示，实际实现会有所不同
            
            # 1. 使用torch实现STFT
            # 转换为浮点数以避免精度问题
            y_float = y_gpu.float() if isinstance(y_gpu, torch.Tensor) else torch.tensor(y, dtype=torch.float32, device=self.device)
            
            # 配置STFT参数 (与librosa兼容)
            n_fft = 2048
            hop_length = 512  # 使用通用hop_length
            win_length = n_fft
            window = torch.hann_window(win_length, device=self.device)
            
            # 计算STFT
            D_gpu = torch.stft(
                y_float, 
                n_fft=n_fft, 
                hop_length=hop_length, 
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
                y=y, sr=self.sr, hop_length=hop_length, n_mels=128
            )
        else:
            # 原始CPU实现
            # 计算短时傅里叶变换 (STFT)
            hop_length = 512  # 使用通用hop_length
            D = librosa.stft(y, hop_length=hop_length)
            
            # 计算频谱幅度
            magnitude = np.abs(D)
            
            # 计算梅尔频谱
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=self.sr, hop_length=hop_length, n_mels=128
            )
        
        # 转换为分贝单位
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 计算色谱图 (适合音符/和弦检测)
        chroma = librosa.feature.chroma_stft(
            y=y, sr=self.sr, hop_length=hop_length
        )
        
        # 提取梅尔频谱对应的时间轴
        times = librosa.times_like(mel_spec, sr=self.sr, hop_length=hop_length)
        
        # 计算频谱质心 (表示声音的"亮度")
        spectral_centroids = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, hop_length=hop_length
        )[0]
        
        # 计算频谱对比度 (高频与低频能量比)
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=self.sr, hop_length=hop_length
        )
        
        # 计算色度能量归一化 (更好地表示和弦)
        chroma_cens = librosa.feature.chroma_cens(
            y=y, sr=self.sr, hop_length=hop_length
        )
        
        # 存储关键频谱特征
        self.spectral_features = {
            "times": times.tolist(),
            "centroid": spectral_centroids.tolist(),
            "contrast_mean": np.mean(spectral_contrast, axis=1).tolist(),
            "chroma_mean": np.mean(chroma, axis=1).tolist(),
            "mel_spec_db": mel_spec_db.tolist(),
            "chroma": chroma.tolist()
        }
    
    def _extract_volume_envelope(self) -> None:
        """提取音量包络"""
        # 获取当前活跃的音频数据
        y = self._get_active_audio_data()
        
        # 计算RMS能量
        hop_length = 512  # 使用通用hop_length
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # 获取对应的时间轴
        times = librosa.times_like(rms, sr=self.sr, hop_length=hop_length)
        
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
        self.volume_envelope = {
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
                self.volume_changes = change_times
    
    def _detect_sections_and_transitions(self) -> None:
        """检测音频段落和过渡点"""
        # 获取当前活跃的音频数据
        y = self._get_active_audio_data()
        
        try:
            # 使用谱平面图进行结构分段
            # 首先获得一个自相似矩阵
            hop_length = 512  # 使用通用hop_length
            mfcc = librosa.feature.mfcc(
                y=y, sr=self.sr, hop_length=hop_length, n_mfcc=13
            )
            
            # 标准化MFCC特征
            mfcc = librosa.util.normalize(mfcc, axis=1)
            
            # 计算自相似矩阵
            similarity = librosa.segment.recurrence_matrix(
                mfcc, mode='affinity', sym=True
            )
            
            # 使用光谱聚类检测段落边界
            boundaries = librosa.segment.agglomerative(similarity, 10)
            boundary_times = librosa.frames_to_time(boundaries, sr=self.sr, hop_length=hop_length)
            
            # 进一步细化边界
            refined_boundaries = []
            min_section_length = 5.0  # 最小段落长度（秒）
            prev_time = 0
            
            for time in boundary_times:
                if time - prev_time >= min_section_length:
                    refined_boundaries.append(float(time))
                    prev_time = time
            
            # 存储段落边界
            self.sections = refined_boundaries
            
            # 检测过渡点（节拍+音量+频谱变化的组合）
            transitions = []
            
            # 如果有检测到节拍和音量变化
            if hasattr(self, 'beat_times') and self.beat_times is not None and hasattr(self, 'volume_changes') and self.volume_changes is not None:
                beat_times = self.beat_times
                volume_changes = self.volume_changes
                
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
                
                self.transitions = filtered_transitions
        
        except Exception as e:
            # 如果段落检测失败，记录错误但继续执行其他分析
            print(f"段落检测失败: {str(e)}")
            self.sections = []
            self.transitions = []
    
    def _create_beat_grid(self) -> None:
        """创建用于osu谱面的节拍网格"""
        if not hasattr(self, 'tempo') or self.tempo is None or not hasattr(self, 'beat_times') or self.beat_times is None or len(self.beat_times) == 0:
            return
        
        try:
            # 计算节拍间隔（秒）
            beat_interval = 60.0 / self.tempo
            
            # 估计开始偏移
            offset = self.beat_times[0]
            
            # 创建一个理想的节拍网格
            duration = len(self._get_active_audio_data()) / self.sr
            num_beats = int(duration / beat_interval) + 1
            ideal_grid = np.arange(num_beats) * beat_interval + offset
            
            # 限制在音频长度内
            ideal_grid = ideal_grid[ideal_grid < duration]
            
            # 存储节拍网格
            self.beat_grid = ideal_grid.tolist()
            
            # 对每个实际节拍，找到最接近的网格节拍
            grid_mapped_beats = []
            for beat in self.beat_times:
                # 找到最接近的网格节拍
                closest_grid_idx = np.argmin(np.abs(ideal_grid - beat))
                grid_mapped_beats.append(float(ideal_grid[closest_grid_idx]))
            
            # 存储映射后的节拍
            self.grid_mapped_beats = grid_mapped_beats
            
            # 存储节拍网格参数 (用于osu谱面)
            self.osu_params = {
                "bpm": self.tempo,
                "offset": float(offset * 1000),  # osu用毫秒
                "beat_divisor": 4,  # 默认4分音符
                "time_signature": [4, 4]  # 默认4/4拍
            }
        
        except Exception as e:
            # 如果节拍网格创建失败，记录错误但继续执行其他分析
            print(f"节拍网格创建失败: {str(e)}")
            self.beat_grid = []
            self.grid_mapped_beats = []
    
    def get_beat_times(self) -> List[float]:
        """获取节拍时间点列表"""
        if self.beat_times is not None:
            return self.beat_times.tolist()
        return []
    
    def get_bpm(self) -> float:
        """获取检测到的BPM"""
        if self.bpm is not None:
            return self.bpm
        return 0.0
    
    def get_bpm_source(self) -> str:
        """获取BPM的来源"""
        return self.beat_source if hasattr(self, 'beat_source') else self.bpm_source
    
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
                return False
                
            self.manual_bpm = bpm_value
            self.bpm = bpm_value
            self.bpm_source = "manual"
            self.beat_source = "manual"
            
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
            self.bpm = main_bpm
            self.bpm_source = "beatmap"
            self.beat_source = "beatmap"
                
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
        if not hasattr(self, 'osu_params') or not self.osu_params:
            return []
        
        osu_data = self.osu_params
        offset_ms = osu_data["offset"]
        bpm = osu_data["bpm"]
        
        # 计算毫秒每节拍 (osu!格式)
        ms_per_beat = 60000.0 / bpm
        
        # 创建基本timing point
        timing_points = [(offset_ms, ms_per_beat)]
        
        # 如果有段落，为每个段落添加timing point
        if hasattr(self, 'sections') and self.sections:
            for section_time in self.sections:
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
        if not hasattr(self, 'volume_envelope') or not self.volume_envelope:
            return []
        
        volume_data = self.volume_envelope
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
        
        # 根据BPM调整流串密度
        if self.bpm is not None:
            # BPM较高时增加流串密度
            if self.bpm > 180:
                result["stream_density"] = min(0.8, 0.5 + (self.bpm - 180) / 100)
            elif self.bpm < 120:
                result["stream_density"] = max(0.3, 0.5 - (120 - self.bpm) / 100)
        
        # 根据频谱质心调整跳跃强度
        if self.spectral_features and "centroid" in self.spectral_features:
            centroids = self.spectral_features["centroid"]
            mean_centroid = np.mean(centroids)
            # 标准化到0-1范围 (假设大多数音频的质心在500-8000Hz之间)
            normalized_centroid = min(1.0, max(0.0, (mean_centroid - 500) / 7500))
            result["jump_intensity"] = normalized_centroid
        
        # 根据节拍规律性调整滑条速度
        if hasattr(self, 'beat_confidence') and self.beat_confidence is not None:
            # 规律性高的曲目可以有更高的滑条速度
            result["slider_velocity"] = 1.0 + self.beat_confidence
        
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
            
            # 获取当前活跃音频源数据
            y = self._get_active_audio_data()
            
            if feature_type == "waveform":
                # 绘制波形图
                plt.plot(librosa.times_like(y, sr=self.sr), y, color='#FF66AA')
                plt.title("音频波形", fontsize=14)
                plt.xlabel("时间 (秒)", fontsize=12)
                plt.ylabel("振幅", fontsize=12)
                
                # 如果有节拍点，在波形上标出
                if hasattr(self, 'beat_times') and self.beat_times is not None:
                    for beat in self.beat_times:
                        plt.axvline(x=beat, color='w', alpha=0.2)
            
            elif feature_type == "mel_spectrogram":
                if self.spectral_features and "mel_spec_db" in self.spectral_features:
                    # 使用预计算的梅尔频谱图
                    mel_spec_db = np.array(self.spectral_features["mel_spec_db"])
                    plt.imshow(mel_spec_db, aspect='auto', origin='lower', interpolation='nearest', cmap='magma')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title("梅尔频谱图", fontsize=14)
                    plt.xlabel("时间 (帧)", fontsize=12)
                    plt.ylabel("梅尔频率", fontsize=12)
                else:
                    # 实时计算梅尔频谱
                    hop_length = 512
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, hop_length=hop_length)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    plt.imshow(mel_spec_db, aspect='auto', origin='lower', interpolation='nearest', cmap='magma')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title("梅尔频谱图", fontsize=14)
                    plt.xlabel("时间 (帧)", fontsize=12)
                    plt.ylabel("梅尔频率", fontsize=12)
            
            elif feature_type == "chroma":
                if self.spectral_features and "chroma" in self.spectral_features:
                    # 使用预计算的色度图
                    chroma = np.array(self.spectral_features["chroma"])
                    plt.imshow(chroma, aspect='auto', origin='lower', interpolation='nearest', cmap='plasma')
                    plt.colorbar()
                    plt.title("色度图", fontsize=14)
                    plt.xlabel("时间 (帧)", fontsize=12)
                    plt.ylabel("音高", fontsize=12)
                    plt.yticks(np.arange(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
                else:
                    # 实时计算色度图
                    hop_length = 512
                    chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, hop_length=hop_length)
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
            保存的文件路径，如果导出多个文件，则返回主文件路径
        """
        import json
        
        # 检查是否有all_sources_analysis
        if hasattr(self, 'features') and 'all_sources_analysis' in self.features:
            # 导出所有音频源的分析结果
            return self._export_all_sources_analysis(output_path)
        
        # 构建分析结果字典
        export_data = {
            "file_path": self.file_path,
            "sample_rate": self.sr,
            "duration": len(self.y) / self.sr if self.y is not None else 0,
            "active_source": self.active_source,
            "bpm": self.bpm,
            "beat_source": self.beat_source,
            "beat_confidence": self.beat_confidence,
            "available_sources": list(self.separated_sources.keys()) + ["original"] if self.separated_sources else ["original"]
        }
        
        # 添加其他分析数据
        if self.beat_times is not None:
            export_data["beat_times"] = self.beat_times.tolist()
            
        if self.beat_strength is not None:
            export_data["beat_strength"] = self.beat_strength.tolist()
            
        if hasattr(self, 'strong_beats') and self.strong_beats:
            export_data["strong_beats"] = self.strong_beats
            
        if self.spectral_features:
            export_data["spectral"] = self.spectral_features
            
        if hasattr(self, 'volume_envelope') and self.volume_envelope:
            export_data["volume"] = self.volume_envelope
            
        if hasattr(self, 'volume_changes') and self.volume_changes:
            export_data["volume_changes"] = self.volume_changes
            
        if hasattr(self, 'sections') and self.sections:
            export_data["sections"] = self.sections
            
        if hasattr(self, 'transitions') and self.transitions:
            export_data["transitions"] = self.transitions
            
        if hasattr(self, 'beat_grid') and self.beat_grid:
            export_data["beat_grid"] = self.beat_grid
            
        if hasattr(self, 'grid_mapped_beats') and self.grid_mapped_beats:
            export_data["grid_mapped_beats"] = self.grid_mapped_beats
            
        if hasattr(self, 'osu_params') and self.osu_params:
            export_data["osu"] = self.osu_params
        
        if output_path is None:
            if self.file_path:
                output_path = os.path.splitext(self.file_path)[0] + ".analysis.json"
            else:
                output_path = "audio_analysis.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return output_path
        
        except Exception as e:
            self.analysis_error.emit(f"导出分析结果时出错: {str(e)}")
            return ""

    def _export_all_sources_analysis(self, output_path: Optional[str] = None) -> str:
        """
        导出所有音频源的分析结果
        
        参数:
            output_path: 输出文件路径根目录，如果为None，则使用音频文件路径目录
            
        返回:
            主文件路径（包含所有结果的索引文件）
        """
        import json
        
        # 确定输出目录和基本文件名
        if output_path is None:
            if self.file_path:
                base_dir = os.path.dirname(self.file_path)
                base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            else:
                base_dir = "."
                base_name = "audio_analysis"
        else:
            base_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # 创建专门用于存放分析结果的文件夹
        analysis_folder = os.path.join(base_dir, f"{base_name}_analysis")
        os.makedirs(analysis_folder, exist_ok=True)
        
        # 音频源类型的人类可读名称
        source_display_names = {
            "vocals": "人声(vocals)",
            "drums": "鼓声(drums)",
            "bass": "贝斯(bass)",
            "other": "其他乐器(other)",
            "original": "原始音频(original)"
        }
        
        # 构建导出文件路径映射
        source_files = {}
        all_results = self.features.get('all_sources_analysis', {})
        
        for source_name, source_data in all_results.items():
            # 获取更具描述性的名称
            display_name = source_display_names.get(source_name, source_name)
            
            # 构建源分析文件路径
            source_filename = f"{base_name}_{display_name}_analysis.json"
            source_path = os.path.join(analysis_folder, source_filename)
            
            # 创建干净的副本，移除可能导致循环引用的字段
            clean_data = copy.deepcopy(source_data)
            
            # 移除可能导致循环引用的字段
            if "all_sources_analysis" in clean_data:
                del clean_data["all_sources_analysis"]
            
            # 保存源分析数据
            try:
                with open(source_path, 'w', encoding='utf-8') as f:
                    json.dump(clean_data, f, indent=2, ensure_ascii=False)
                
                # 记录文件路径
                source_files[source_name] = {
                    "display_name": display_name,
                    "file_path": os.path.relpath(source_path, base_dir)
                }
                
                print(f"已导出 {source_name} 的分析结果到 {source_filename}")
                
            except Exception as e:
                self.analysis_error.emit(f"导出 {source_name} 分析结果时出错: {str(e)}")
        
        # 创建包含所有源索引的主文件
        main_data = {
            "file_path": self.file_path,
            "analysis_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active_source": self.active_source,
            "available_sources": list(self.separated_sources.keys()) + ["original"] if self.separated_sources else ["original"],
            "source_files": source_files
        }
        
        # 主索引文件路径
        main_file_path = os.path.join(analysis_folder, f"{base_name}_all_sources_analysis.json")
        
        try:
            with open(main_file_path, 'w', encoding='utf-8') as f:
                json.dump(main_data, f, indent=2, ensure_ascii=False)
            
            print(f"已导出所有源的分析索引到 {main_file_path}")
            return main_file_path
            
        except Exception as e:
            self.analysis_error.emit(f"导出分析索引文件时出错: {str(e)}")
            return ""
    
    def _get_active_audio_data(self) -> np.ndarray:
        """
        获取当前活跃音频源的数据
        
        返回:
            np.ndarray: 活跃音频源的数据
        """
        if self.active_source == "original" or self.active_source not in self.separated_sources:
            return self.y
        return self.separated_sources[self.active_source]
        
    def _load_demucs_model(self) -> None:
        """加载Demucs模型用于音频源分离"""
        self.analysis_progress.emit(5)
        # 使用选择的Demucs变体
        self.demucs_model = get_model(self.current_model)
        if self.use_gpu and GPU_AVAILABLE:
            self.demucs_model.to(torch.device("cuda"))
        else:
            self.demucs_model.to(torch.device("cpu"))
    
    def _load_melband_model(self) -> None:
        """加载MelBand RoFormer模型用于高质量人声分离"""
        if not MELBAND_AVAILABLE:
            raise ImportError("MelBand RoFormer模型未安装，请使用pip install melband-roformer安装")
        
        self.analysis_progress.emit(5)
        # 加载预训练的MelBand RoFormer模型
        self.melband_model = MelBandRoFormer.from_pretrained("melband/melband-roformer-base")
        if self.use_gpu and GPU_AVAILABLE:
            self.melband_model.to(torch.device("cuda"))
        else:
            self.melband_model.to(torch.device("cpu"))
        
        # 初始化音频处理器
        self.melband_processor = AudioProcessor()
    
    def _load_scnetxl_model(self) -> None:
        """加载SCNet XL模型用于更高质量的音频分离"""
        if not SCNETXL_AVAILABLE:
            raise ImportError("SCNet XL模型未安装，请使用pip install scnetxl安装")
        
        self.analysis_progress.emit(5)
        # 加载预训练的SCNet XL模型
        self.scnetxl_model = SCNetXL.from_pretrained("scnet/scnetxl-base")
        if self.use_gpu and GPU_AVAILABLE:
            self.scnetxl_model.to(torch.device("cuda"))
        else:
            self.scnetxl_model.to(torch.device("cpu"))
    
    def _separate_audio_sources(self) -> None:
        """使用所选模型分离音频源"""
        if self.current_model in ["demucs_v4", "htdemucs", "htdemucs_ft"]:
            # 使用Demucs模型
            if self.demucs_model is None:
                self._load_demucs_model()
            self._separate_with_demucs()
        elif self.current_model == "melband_roformer":
            # 使用MelBand RoFormer模型
            if self.melband_model is None:
                self._load_melband_model()
            self._separate_with_melband()
        elif self.current_model == "scnetxl":
            # 使用SCNet XL模型
            if self.scnetxl_model is None:
                self._load_scnetxl_model()
            self._separate_with_scnetxl()
        else:
            # 回退到默认Demucs模型
            if self.demucs_model is None:
                self.current_model = "htdemucs"
                self._load_demucs_model()
            self._separate_with_demucs()
        
        # 打印所有可用源用于调试
        print(f"音频分离完成，使用模型: {self.current_model}")
        print(f"可用音频源: {list(self.separated_sources.keys())}")
        
        self.analysis_progress.emit(15)
    
    def _separate_with_demucs(self) -> None:
        """使用Demucs模型分离音频源"""
        self.analysis_progress.emit(8)
        
        # 转换为torch张量并重塑为Demucs期望的格式
        audio_tensor = torch.tensor(self.y).float()
        
        # 检查形状并转换为立体声 [2, samples]
        if audio_tensor.dim() == 1:  # 单声道 [samples]
            # 直接转换为立体声 [2, samples]
            audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)  # [2, samples]
        elif audio_tensor.dim() == 2:
            if audio_tensor.shape[0] != 2:  # 如果第一维不是2
                # 确保通道维度是2
                audio_tensor = audio_tensor.transpose(0, 1) if audio_tensor.shape[1] == 2 else audio_tensor.repeat(2, 1)
        
        # 添加批次维度 [1, 2, samples]
        audio_tensor = audio_tensor.unsqueeze(0)
        
        # 移动到正确的设备
        device = torch.device("cuda" if self.use_gpu and GPU_AVAILABLE else "cpu")
        audio_tensor = audio_tensor.to(device)
        
        # 打印形状用于调试
        print(f"音频张量形状: {audio_tensor.shape}")
        
        # 应用模型 - 使用 apply_model 而不是直接调用模型以获得更好的效率
        with torch.no_grad():
            sources = apply_model(self.demucs_model, audio_tensor, device=device)[0]
        
        # 从神经网络输出中提取各个源
        self.separated_sources = {}
        
        # 打印调试信息
        print(f"Demucs模型输出源的顺序: {self.DEMUCS_SOURCE_ORDER}")
        
        # 将Demucs输出的源直接映射到对应的标准名称
        # Demucs输出顺序为: [drums, bass, other, vocals]
        for i, source_name in enumerate(self.DEMUCS_SOURCE_ORDER):
            # 转换为numpy数组并取平均值如果是立体声
            source_np = sources[i].mean(dim=0).cpu().numpy()
            self.separated_sources[source_name] = source_np
            print(f"处理源 {i}: {source_name}")
        
        # 打印全部可用的源
        print(f"分离后的源: {list(self.separated_sources.keys())}")
    
    def _separate_with_melband(self) -> None:
        """使用MelBand RoFormer模型分离音频源"""
        self.analysis_progress.emit(8)
        
        # 确保音频采样率为16kHz (MelBand RoFormer的标准)
        if self.sr != 16000:
            y_resampled = librosa.resample(self.y, orig_sr=self.sr, target_sr=16000)
        else:
            y_resampled = self.y
        
        # 转换为torch张量
        audio_tensor = torch.tensor(y_resampled).float()
        if audio_tensor.dim() == 2:
            # 如果是立体声，转换为单声道
            audio_tensor = audio_tensor.mean(dim=0)
        
        # 移动到正确的设备
        device = torch.device("cuda" if self.use_gpu and GPU_AVAILABLE else "cpu")
        audio_tensor = audio_tensor.to(device)
        
        # 处理音频
        with torch.no_grad():
            # 准备输入
            inputs = self.melband_processor(audio_tensor, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 模型推理
            outputs = self.melband_model(**inputs)
            
            # 获取分离结果
            vocals = outputs.vocals.cpu().numpy().squeeze()
            instruments = outputs.instruments.cpu().numpy().squeeze()
        
        # 如果原始采样率不是16kHz，重新采样回原始采样率
        if self.sr != 16000:
            vocals = librosa.resample(vocals, orig_sr=16000, target_sr=self.sr)
            instruments = librosa.resample(instruments, orig_sr=16000, target_sr=self.sr)
        
        # 保存分离后的源
        self.separated_sources = {}
        
        # MelBand模型输出对应关系
        self.separated_sources["vocals"] = vocals        # 人声
        self.separated_sources["other"] = instruments    # 其他乐器
        
        # 尝试进一步分离乐器（如果模型支持）
        try:
            drums = outputs.drums.cpu().numpy().squeeze()
            bass = outputs.bass.cpu().numpy().squeeze()
            
            if self.sr != 16000:
                drums = librosa.resample(drums, orig_sr=16000, target_sr=self.sr)
                bass = librosa.resample(bass, orig_sr=16000, target_sr=self.sr)
            
            self.separated_sources["drums"] = drums    # 鼓声
            self.separated_sources["bass"] = bass      # 贝斯
        except (AttributeError, KeyError):
            # 如果模型不支持进一步分离，使用近似方法
            self._approximate_drums_bass(instruments)
        
        # 打印全部可用的源
        print(f"MelBand分离后的源: {list(self.separated_sources.keys())}")
    
    def _approximate_drums_bass(self, instruments: np.ndarray) -> None:
        """使用简单频率分离方法近似分离鼓声和贝斯"""
        # 鼓声主要在中高频段
        drums = librosa.effects.percussive(instruments, margin=3.0)
        
        # 贝斯主要在低频段
        bass = librosa.effects.harmonic(instruments)
        # 应用低通滤波器来获取贝斯
        bass = librosa.decompose.nn_filter(
            bass,
            aggregate=np.median,
            metric='cosine',
            width=int(self.sr/30)  # 约33ms窗口
        )
        
        # 添加近似分离的鼓声和贝斯
        self.separated_sources["drums"] = drums    # 鼓声
        self.separated_sources["bass"] = bass      # 贝斯
    
    def _separate_with_scnetxl(self) -> None:
        """使用SCNet XL模型分离音频源"""
        self.analysis_progress.emit(8)
        
        # SCNet XL使用44.1kHz采样率
        if self.sr != 44100:
            y_resampled = librosa.resample(self.y, orig_sr=self.sr, target_sr=44100)
        else:
            y_resampled = self.y
        
        # 确保音频是单声道
        if len(y_resampled.shape) == 2:
            y_mono = np.mean(y_resampled, axis=0)
        else:
            y_mono = y_resampled
        
        # 转换为torch张量
        audio_tensor = torch.tensor(y_mono).float().unsqueeze(0)
        
        # 移动到正确的设备
        device = torch.device("cuda" if self.use_gpu and GPU_AVAILABLE else "cpu")
        audio_tensor = audio_tensor.to(device)
        
        # 应用模型
        with torch.no_grad():
            outputs = self.scnetxl_model(audio_tensor)
        
        # 从输出中提取各个源
        self.separated_sources = {}
        
        # SCNet模型输出的标准名称
        source_names = ["vocals", "drums", "bass", "other"]
        
        # 直接从模型输出映射到标准源名称
        for source_name in source_names:
            if source_name in outputs:
                source_audio = outputs[source_name].cpu().numpy().squeeze()
                
                # 如果需要，重新采样回原始采样率
                if self.sr != 44100:
                    source_audio = librosa.resample(source_audio, orig_sr=44100, target_sr=self.sr)
                
                self.separated_sources[source_name] = source_audio
        
        # 打印全部可用的源
        print(f"SCNet分离后的源: {list(self.separated_sources.keys())}")
    
    def set_separation_model(self, model_name: str) -> None:
        """
        设置使用的人声分离模型
        
        参数:
            model_name: 模型名称，应该是SEPARATION_MODELS字典中的一个键
        """
        if model_name in self.SEPARATION_MODELS:
            self.current_model = model_name
            # 清除现有模型缓存
            self.demucs_model = None
            self.melband_model = None
            self.scnetxl_model = None
            # 清除已分离的源
            self.separated_sources = {}
            self.active_source = "original"
        else:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型有: {list(self.SEPARATION_MODELS.keys())}")
    
    def get_available_models(self) -> Dict[str, str]:
        """
        获取所有可用的人声分离模型
        
        返回:
            Dict[str, str]: 模型ID到模型名称的映射
        """
        return self.SEPARATION_MODELS.copy()
    
    def _select_active_source(self) -> None:
        """根据优先级选择活跃音频源"""
        # 检查优先级列表中的源是否可用
        for source in self.source_priority:
            if source in self.separated_sources:
                self.active_source = source
                return
        # 如果没有找到匹配的源，使用原始音频
        self.active_source = "original"
    
    def set_use_source_separation(self, enabled: bool) -> None:
        """
        设置是否使用音频源分离
        
        参数:
            enabled: 是否启用音频源分离
        """
        self.use_source_separation = enabled
        
    def set_source_priority(self, priority_list: List[str]) -> None:
        """
        设置音频源优先级
        
        参数:
            priority_list: 音频源优先级列表
        """
        # 验证优先级列表
        valid_sources = set(self.AUDIO_SOURCES + ["original"])
        for source in priority_list:
            if source not in valid_sources:
                self.analysis_error.emit(f"无效的音频源: {source}")
                return
        
        self.source_priority = priority_list
        
        # 如果已经有分离的源，重新选择活跃源
        if self.separated_sources:
            self._select_active_source()
            
    def set_active_source(self, source: str) -> None:
        """
        设置当前活跃的音频源
        
        参数:
            source: 音频源名称
        """
        if source == "original" or source in self.separated_sources:
            self.active_source = source
        else:
            self.analysis_error.emit(f"无效的音频源: {source}")
    
    def get_available_sources(self) -> List[str]:
        """
        获取可用的音频源列表
        
        返回:
            List[str]: 可用音频源列表
        """
        return list(self.separated_sources.keys()) + ["original"]
        
    def export_separated_audio(self, output_dir: str) -> Dict[str, str]:
        """
        导出分离后的音频到指定目录，并为每个音频源生成分析数据
        
        参数:
            output_dir: 输出目录
            
        返回:
            Dict[str, str]: 源名称到输出文件路径的映射
        """
        if not self.separated_sources:
            self.analysis_error.emit("没有分离的音频源可以导出")
            return {}
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取原始文件名（不带扩展名）
        basename = os.path.splitext(os.path.basename(self.file_path))[0]
        
        # 音频源类型的人类可读名称
        source_display_names = {
            "vocals": "人声(vocals)",
            "drums": "鼓声(drums)",
            "bass": "贝斯(bass)",
            "other": "其他乐器(other)"
        }
        
        # 打印调试信息
        print("导出分离音频文件和分析数据:")
        for source_name in self.separated_sources.keys():
            print(f"  - {source_name}: {source_display_names.get(source_name, source_name)}")
        
        # 导出每个源，使用更清晰的文件名
        result = {}
        
        # 建立源类型与真实内容的映射 - 这是临时映射，仅用于调试目的
        actual_content_map = {
            "vocals": "人声内容",
            "drums": "鼓声内容",
            "bass": "贝斯内容", 
            "other": "其他乐器内容"
        }
        
        # 保存当前活跃音频源
        original_active_source = self.active_source
        original_features = {}
        # 安全地复制features，避免循环引用
        if hasattr(self, 'features'):
            # 如果features包含all_sources_analysis，移除它以避免循环引用
            if isinstance(self.features, dict) and 'all_sources_analysis' in self.features:
                original_features = {k: v for k, v in self.features.items() if k != 'all_sources_analysis'}
            else:
                original_features = self.features.copy()
        
        # 导出每个源并在结果字典中使用正确的显示名称作为键
        for source_name, source_data in self.separated_sources.items():
            # 获取更具描述性的名称
            display_name = source_display_names.get(source_name, source_name)
            
            # 创建包含源类型的文件名
            output_filename = f"{basename}_{display_name}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # 写入音频文件
            sf.write(output_path, source_data, self.sr)
            
            # 对每个分离的音频源进行分析并导出分析数据
            self.set_active_source(source_name)
            
            # 为当前源执行分析（复用分析功能但不触发信号）
            temp_y = self._get_active_audio_data()
            
            # 创建一个临时的音频数据，避免干扰原始分析结果
            temp_features = {}
            
            # 提取基本特征
            self._detect_tempo_and_beats()
            self._extract_beat_strength()
            self._extract_spectral_features()
            self._extract_volume_envelope()
            self._detect_sections_and_transitions()
            self._create_beat_grid()
            
            # 复制必要的分析结果到临时字典
            for key in ["bpm", "beat_times", "beat_frames", "beat_strengths", 
                        "spectral_contrast", "spectral_bandwidth", "spectral_rolloff",
                        "mfcc", "volume_envelope", "sections", "transitions", 
                        "beat_grid", "energy_points", "active_source"]:
                if hasattr(self, key):
                    temp_features[key] = getattr(self, key)
            
            # 创建分析数据输出文件名
            analysis_filename = f"{basename}_{display_name}_analysis.json"
            analysis_path = os.path.join(output_dir, analysis_filename)
            
            # 将分析数据保存为JSON文件
            try:
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    # 添加元数据
                    temp_features["source_type"] = source_name
                    temp_features["display_name"] = display_name
                    temp_features["analysis_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    json.dump(temp_features, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
                
                # 记录文件路径
                result[source_name] = {
                    "audio": output_path,
                    "analysis": analysis_path
                }
                
                # 打印详细的导出信息用于调试
                print(f"导出 {source_name} 到 {output_filename} 和 {analysis_filename}")
            except Exception as e:
                self.analysis_error.emit(f"导出 {source_name} 分析数据时出错: {str(e)}")
        
        # 恢复原始活跃源和特征
        self.set_active_source(original_active_source)
        if original_features:
            # 恢复时避免将features直接设置为可能有循环引用的对象
            if hasattr(self, 'features'):
                self.features = original_features
        
        return result 
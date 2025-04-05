import os
import pygame
import numpy as np
from moviepy.editor import VideoClip, AudioFileClip, CompositeVideoClip
from pygame.locals import *

class OsuParser:
    def __init__(self, osu_file):
        self.osu_file = osu_file
        self.audio_filename = ""
        self.timing_points = []
        self.hit_objects = []
        self.circle_size = 4  # 默认值
        self.slider_multiplier = 1.0  # 默认值
        self.parse()

    def parse(self):
        with open(self.osu_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        section = None
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
                continue
            
            if section == 'General':
                if line.startswith('AudioFilename:'):
                    self.audio_filename = line.split(':', 1)[1].strip()
            
            elif section == 'Difficulty':
                if line.startswith('CircleSize:'):
                    self.circle_size = float(line.split(':', 1)[1].strip())
                elif line.startswith('SliderMultiplier:'):
                    self.slider_multiplier = float(line.split(':', 1)[1].strip())
            
            elif section == 'TimingPoints':
                parts = line.split(',')
                if len(parts) >= 2:
                    time = int(float(parts[0]))  # 使用float()先转换为浮点数，再转为整数
                    beat_length = float(parts[1])
                    self.timing_points.append((time, beat_length))
            
            elif section == 'HitObjects':
                parts = line.split(',')
                if len(parts) >= 5:
                    x = int(parts[0])
                    y = int(parts[1])
                    time = int(float(parts[2]))  # 使用float()先转换为浮点数，再转为整数
                    type_bit = int(parts[3])
                    hit_sound = int(parts[4])
                    
                    # 判断是否为圆圈(1)或滑条(2)
                    is_circle = (type_bit & 1) > 0
                    is_slider = (type_bit & 2) > 0
                    
                    if is_circle:
                        self.hit_objects.append({"type": "circle", "time": time, "x": x, "y": y})
                    elif is_slider:
                        # 简单处理滑条，获取滑条长度
                        curve_points = []
                        if len(parts) >= 6 and '|' in parts[5]:
                            curve_type = parts[5][0]
                            curve_points_str = parts[5][2:].split('|')
                            for point_str in curve_points_str:
                                if ':' in point_str:
                                    px, py = map(int, point_str.split(':'))
                                    curve_points.append((px, py))
                        
                        # 获取滑条长度
                        slides = 1
                        if len(parts) >= 7:
                            slides = int(parts[6])
                        
                        # 估算滑条持续时间
                        duration = 0
                        for tp_time, beat_length in reversed(self.timing_points):
                            if time >= tp_time:
                                if beat_length > 0:
                                    # 正常时间点
                                    duration = beat_length * slides
                                else:
                                    # 继承时间点，使用负值表示速度乘数
                                    slider_velocity = 100.0 / abs(beat_length)
                                    # 使用第一个时间点的beat_length（假设它是正的）
                                    first_beat_length = next((bl for t, bl in self.timing_points if bl > 0), 1000)
                                    duration = first_beat_length * slides / slider_velocity
                                break
                        
                        # 估算滑条长度（像素）
                        slider_length = 100  # 默认值，实际应该根据滑条点计算
                        
                        self.hit_objects.append({
                            "type": "slider",
                            "time": time,
                            "x": x,
                            "y": y,
                            "duration": duration,
                            "end_time": time + int(duration)
                        })
    
    def get_hits_for_1k(self):
        """获取适合1k下落式音游的按键序列"""
        hits = []
        for obj in self.hit_objects:
            if obj["type"] == "circle":
                hits.append({"type": "tap", "time": obj["time"]})
            elif obj["type"] == "slider":
                hits.append({"type": "hold", "time": obj["time"], "end_time": obj["end_time"]})
        
        # 按时间排序
        hits.sort(key=lambda x: x["time"])
        return hits

class VSRGRenderer:
    def __init__(self, width=800, height=600, fps=60, lane_width=200):
        # 设置宽度等于轨道宽度
        self.lane_width = lane_width
        self.width = lane_width
        self.height = height
        self.fps = fps
        self.receptors_y = self.height - 100  # 接收点的Y坐标
        self.bg_color = (50, 50, 50)  # 灰色背景
        
        # 初始化Pygame
        pygame.init()
        self.surface = pygame.Surface((self.width, self.height))
        
        # 单键模式的X坐标 - 居中
        self.lane_x = self.width // 2
    
    def render_frame(self, t, hits, scroll_speed=1000):
        """渲染某一时刻的帧"""
        # 清空背景
        self.surface.fill(self.bg_color)
        
        # 绘制接收点
        receptor_width = self.lane_width * 0.8
        pygame.draw.rect(self.surface, (200, 200, 200), 
                         (self.lane_x - receptor_width/2, self.receptors_y, receptor_width, 10))
        
        # 绘制分隔线 - 轨道边界
        pygame.draw.line(self.surface, (100, 100, 100), 
                         (0, 0), 
                         (0, self.height), 2)
        pygame.draw.line(self.surface, (100, 100, 100), 
                         (self.width-1, 0), 
                         (self.width-1, self.height), 2)
        
        # 计算当前时间毫秒
        t_ms = t * 1000
        
        # 绘制音符
        for hit in hits:
            hit_time = hit["time"]
            time_diff = hit_time - t_ms
            
            # 如果音符还没出现在屏幕上或者已经过去了，跳过
            if time_diff < -1000 or time_diff > scroll_speed:
                continue
            
            # 计算音符Y坐标
            note_y = self.receptors_y - (time_diff / scroll_speed) * self.height
            
            if hit["type"] == "tap":
                # 绘制单点音符（方块）
                note_width = self.lane_width * 0.8
                note_height = 20
                pygame.draw.rect(self.surface, (255, 255, 255), 
                                (self.lane_x - note_width/2, note_y - note_height, note_width, note_height))
            elif hit["type"] == "hold":
                # 绘制长条音符（长方块）
                end_time = hit["end_time"]
                end_time_diff = end_time - t_ms
                
                # 如果长条尾部还没出现，跳过
                if end_time_diff < -1000:
                    continue
                
                # 计算长条尾部Y坐标
                end_note_y = self.receptors_y - (end_time_diff / scroll_speed) * self.height
                
                # 绘制长条
                note_width = self.lane_width * 0.8
                pygame.draw.rect(self.surface, (240, 240, 240), 
                                (self.lane_x - note_width/2, end_note_y - 10, note_width, note_y - end_note_y + 10))
                
                # 绘制长条头部和尾部
                pygame.draw.rect(self.surface, (250, 250, 250), 
                                (self.lane_x - note_width/2, note_y - 20, note_width, 20))
                pygame.draw.rect(self.surface, (250, 250, 250), 
                                (self.lane_x - note_width/2, end_note_y - 10, note_width, 10))
        
        # 转换Surface为numpy数组
        frame = pygame.surfarray.array3d(self.surface)
        # 转置以符合MoviePy格式要求
        return frame.transpose(1, 0, 2)

def create_vsrg_video(osu_file, output_file, fps=60, scroll_speed=1000, lane_width=200):
    """创建VSRG视频"""
    # 解析osu文件
    parser = OsuParser(osu_file)
    hits = parser.get_hits_for_1k()
    
    # 获取音频文件路径
    audio_path = os.path.join(os.path.dirname(osu_file), parser.audio_filename)
    
    # 创建渲染器
    renderer = VSRGRenderer(fps=fps, lane_width=lane_width)
    
    # 计算视频时长（毫秒）
    duration_ms = max([hit["end_time"] if "end_time" in hit else hit["time"] for hit in hits]) + 3000
    duration = duration_ms / 1000.0  # 转换为秒
    
    # 创建视频剪辑
    def make_frame(t):
        return renderer.render_frame(t, hits, scroll_speed)
    
    video = VideoClip(make_frame, duration=duration)
    
    # 添加音频
    try:
        audio = AudioFileClip(audio_path)
        video = video.set_audio(audio)
    except Exception as e:
        print(f"无法加载音频文件: {e}")
    
    # 写入视频文件
    video.write_videofile(output_file, fps=fps, codec='libx264', audio_codec='aac')
    
    # 清理
    pygame.quit()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python osu_to_vsrg.py [osu文件路径] [输出视频文件路径(可选)]")
        osu_files = [f for f in os.listdir() if f.endswith('.osu')]
        if osu_files:
            print(f"检测到当前目录下的osu文件：{osu_files[0]}，将使用此文件")
            osu_file = osu_files[0]
            output_file = os.path.splitext(osu_file)[0] + "_1k.mp4"
        else:
            sys.exit(1)
    else:
        osu_file = sys.argv[1]
        if len(sys.argv) >= 3:
            output_file = sys.argv[2]
        else:
            output_file = os.path.splitext(osu_file)[0] + "_1k.mp4"
    
    print(f"正在将 {osu_file} 转换为1k下落式音游视频...")
    create_vsrg_video(osu_file, output_file)
    print(f"转换完成，输出文件：{output_file}") 
import re
import os

def extract_chinese_text(line):
    # 匹配<b>标签中的中文文本
    pattern = r'<b>\s*​?\s*​?(.*?)​?\s*​?\s*</b>'
    match = re.search(pattern, line)
    if match:
        return match.group(1).strip()
    return None

def parse_time(time_str):
    # 解析时间格式为秒数，便于比较
    hours, minutes, rest = time_str.split(':', 2)
    seconds, milliseconds = rest.split(',')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def format_time(seconds):
    # 将秒数转换回SRT时间格式
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_part = seconds % 60
    seconds_int = int(seconds_part)
    milliseconds = int((seconds_part - seconds_int) * 1000)
    return f"{hours}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"

def process_srt_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    subtitle_dict = {}  # 用于存储唯一文本的字幕
    current_index = None
    current_timestamp = None
    current_start_time = None
    current_end_time = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 字幕索引
        if line.isdigit():
            current_index = int(line)
            i += 1
            continue
        
        # 时间戳
        if '-->' in line:
            current_timestamp = line
            time_parts = line.split(' --> ')
            current_start_time = parse_time(time_parts[0])
            current_end_time = parse_time(time_parts[1])
            i += 1
            continue
        
        # 字幕文本
        if '<font' in line and current_start_time is not None:
            chinese_text = extract_chinese_text(line)
            if chinese_text:
                # 使用文本作为键，确保相同文本只处理一次
                if chinese_text not in subtitle_dict:
                    subtitle_dict[chinese_text] = {
                        'start_times': [current_start_time],
                        'end_times': [current_end_time]
                    }
                else:
                    # 检查是否与上一个时间段连续或重叠
                    last_end = subtitle_dict[chinese_text]['end_times'][-1]
                    if current_start_time <= last_end or abs(current_start_time - last_end) < 0.5:  # 允许0.5秒的误差
                        # 更新结束时间为较大的值
                        subtitle_dict[chinese_text]['end_times'][-1] = max(current_end_time, last_end)
                    else:
                        # 添加新的时间段
                        subtitle_dict[chinese_text]['start_times'].append(current_start_time)
                        subtitle_dict[chinese_text]['end_times'].append(current_end_time)
        
        i += 1
    
    # 将处理后的字幕写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        index = 1
        for text, times in subtitle_dict.items():
            for i in range(len(times['start_times'])):
                start_formatted = format_time(times['start_times'][i])
                end_formatted = format_time(times['end_times'][i])
                
                f.write(f"{index}\n")
                f.write(f"{start_formatted} --> {end_formatted}\n")
                f.write(f"{text}\n\n")
                index += 1
    
    print(f"处理完成，共处理了{len(subtitle_dict)}个不同的字幕文本")
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    input_file = "MIMI - 天使の涙 (feat.初音ミク).srt"
    output_file = "MIMI - 天使の涙 (feat.初音ミク)_中文.srt"
    process_srt_file(input_file, output_file)

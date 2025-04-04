import sys

# 打开文件并读取内容
file_path = 'MIMI - te n shi no na mi da (Fake Emperor) [Expert].osu'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 处理文件
new_lines = []
for line in lines:
    # 如果找到时间点行(TimingPoints部分)
    if ',' in line and not line.startswith('//') and any(c.isdigit() for c in line):
        parts = line.split(',')
        if len(parts) >= 2 and parts[0].strip().replace('.', '', 1).replace('-', '', 1).isdigit():
            try:
                time = float(parts[0])
                new_time = max(0, time - 20)  # 减少20ms,但确保不会是负数
                line = str(new_time) + ',' + ','.join(parts[1:])
            except ValueError:
                pass  # 如果转换失败,保持原样
    
    # 如果是HitObjects部分的行
    if len(line.split(',')) >= 3 and not line.startswith('//'):
        parts = line.split(',')
        if len(parts) >= 3 and parts[0].replace('-', '', 1).isdigit() and parts[1].replace('-', '', 1).isdigit() and parts[2].replace('.', '', 1).replace('-', '', 1).isdigit():
            try:
                time = float(parts[2])
                new_time = max(0, time - 20)  # 减少20ms
                parts[2] = str(int(new_time))
                line = ','.join(parts)
            except ValueError:
                pass  # 如果转换失败,保持原样
    
    new_lines.append(line)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('文件处理完成,所有时间轴已减少20ms') 
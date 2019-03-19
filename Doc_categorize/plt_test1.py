import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

jiaodu = ['0', '15', '30', '15', '60', '75', '90', '105', '120']
x = range(len(jiaodu))

y = [85.6801, 7.64586, 86.0956, 159.229, 179.534, 163.238, 96.4436, 10.1619, 90.9262, ]

# plt.figure(figsize=(10, 6))

plt.plot(x, y, 'b-', label="1", marker='*', markersize=7, linewidth=3)  # b代表blue颜色  -代表直线

plt.title('各个区域亮度变化')
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8), ('0', '15', '30', '15', '60', '75', '90', '105', '120'))
plt.xlabel('角度')
plt.ylabel('亮度')
# plt.grid(x1)
plt.show()

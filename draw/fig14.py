import matplotlib.pyplot as plt
import numpy as np

font1 = {'family': 'Arial',
         'weight': 'normal',
         'size': 15,
         }
font2 = {'family': 'Arial',
         'weight': 'normal',
         'size': 17,
         }

# 数据
punica = [
    [[0.029, 0.03, 0.05, 0.14, 0.24],[0.0125, 0.0545, 0.0948, 0.1904, 0.2896]], # qwen

]


dlora = [
    [[0.08, 0.1, 0.138, 0.236, 0.32],[0.016, 0.04, 0.07, 0.19, 0.3]],           # qwen

]


slora = [
    [[0.0210, 0.0237, 0.03, 0.09, 0.183], [0.0032, 0.0342, 0.0581, 0.1263, 0.2031]],    # qwen

]


ours = [
    [[0.017, 0.022, 0.025, 0.073, 0.15], [0.0001, 0.0002, 0.02, 0.03, 0.07]],     # qwen

]

skewness = [
    [0.1369, 0.1263, 0.128, 0.1317],  # slora
    [0.1883, 0.1904, 0.1901, 0.1645],  # punica
    [0.27, 0.19, 0.15, 0.13],  # dlora
    [0.05, 0.03, 0.028, 0.027],  # ours
]

# 创建2行3列的子图布局
fig, axs = plt.subplots(2, 3, figsize=(15, 6.5))

# 模型和任务标识
models = ['QwenVL', 'LLaVA-1.5-7B', 'LLaVA-1.5-13B']
tasks = ['VQA', 'VAT']
x_labels = [2, 4, 6, 10, 16]

# 循环绘制每个子图
for i in range(2):  # 任务
    for j in range(1):  # 模型
        ax = axs[i, j]
        
        # 绘制四个模型的曲线
        ax.plot(range(5), dlora[j][i], color='#2D4059', linewidth=1.5, marker='s', label='dLoRA', linestyle='-', markersize=5, zorder=10)
        ax.plot(range(5), slora[j][i], color='#EA5455', linewidth=1.5, marker='o', label='S-LoRA', linestyle='-', markersize=5, zorder=10)
        ax.plot(range(5), punica[j][i], color='#9DBDFF', linewidth=1.5, marker='D', label='Punica', linestyle='-', markersize=5, zorder=10)
        ax.plot(range(5), ours[j][i], color='#FFD460', linewidth=1.5, marker='^', label='Ours', linestyle='-', markersize=5, zorder=10)
        # 设置每个子图的y轴范围 (可以根据需要自定义每个子图的y轴)
        if j == 0:  # QwenVL
            ax.set_ylim(0, 0.3)  # 设置QwenVL的y轴范围
        elif j == 1:  # LLaVA-1.5-7B
            ax.set_ylim(0, 0.3)  # 设置LLaVA-1.5-7B的y轴范围
        elif j == 2:  # LLaVA-1.5-13B
            ax.set_ylim(0, 0.3)  # 设置LLaVA-1.5-13B的y轴范围
        # 设置坐标轴范围、刻度等
        # ax.set_ylim(0., 0.3)
        ax.set_xticks(range(5))
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        
        # 设置每个子图的x label和y label
        ax.set_xlabel('Arrival Rate (reqs/s)\n', fontdict=font2)
        ax.set_ylabel('Latency (ms)', fontdict=font2)

        # 添加子图的caption
        caption = f"({chr(97 + i * 3 + j)}) {models[j]}, {tasks[i]}"
        ax.text(0.5, -0.38, caption, transform=ax.transAxes, fontdict=font2, va='center', ha='center')

# 统一的图例
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=15)

plt.rcParams['hatch.linewidth'] 

# 调整子图布局
plt.tight_layout(rect=[0, 0, 1, 0.94])
# plt.tight_layout(rect=[0, 0.05, 1, 0.95])


# 保存图像
plt.savefig("compare_vqa_vat.pdf")

# 展示图像
plt.show()
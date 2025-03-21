slora=[0.01648426055908203, 0.016264915466308594, 0.015158653259277344, 0.016832351684570312, 0.01923084259033203, 0.02602100372314453, 0.0901937484741211, 0.12331485748291016, 0.15936851501464844, 0.1782989501953125, 0.22422313690185547, 0.3705596923828125]
dlora=[0.10906219482421875, 0.08595466613769531, 0.07985115051269531, 0.0786447525024414, 0.0987386703491211, 0.09883403778076172, 0.1078939437866211, 0.08381366729736328, 0.10532379150390625, 0.13916015625, 0.20157337188720703, 0.30683040618896484]
punica=[0.04736521327868104, 0.046624832320958376, 0.0457071615383029, 0.0456430995836854, 0.06357237184420228, 0.0634633214212954, 0.06306445226073265, 0.06584329064935446, 0.07156308647245169, 0.09744480345398188, 0.1503181280568242, 0.2522342912852764]
ours=[0.02231597900390625, 0.02071857452392578, 0.020132064819335938, 0.02009868621826172, 0.020189285278320312, 0.020546913146972656, 0.02170562744140625, 0.02270221710205078, 0.027923583984375, 0.036797523498535156, 0.07511138916015625, 0.1253223419189453]
import matplotlib.pyplot as plt
import numpy as np
font1 = {'family': 'Arial',
         'weight': 'normal',
         'size': 15,
         }
font2 = {'family': 'Arial',
         'weight': 'normal',
         'size': 15,
         }
fig, ax = plt.subplots(figsize=(3, 2.5))

plt.rcParams['hatch.linewidth'] = 10



width = 0.15  # 每个子柱子的宽度
model=0 # 0-qwen 1-llava7b 2-llava13b
task=0 # 0-vqa 1-vat

x = [1,2,4,6,8,16,24,28,32,36,40,44]
# ['#EA5455','#2D4059','#FFD460','#9DBDFF']

plt.plot(x, ours, color='#EA5455',linewidth=1.5,marker='s',label='Ours',linestyle='-',markersize=5,zorder=10)
plt.plot(x, punica, color='#2D4059',linewidth=1.5,marker='o',label='Punica',linestyle='-',markersize=5,zorder=10)
plt.plot(x, slora, color='#FFD460',linewidth=1.5,marker='D',label='S-LoRA',linestyle='-',markersize=5,zorder=10)
plt.plot(x, dlora, color='#9DBDFF',linewidth=1.5,marker='^',label='dLoRA',linestyle='-',markersize=5,zorder=10)

plt.text(1.5, 0.21, 'Decode\n stage',size=11)
plt.text(20, 0.21, 'Prefill\nstage',size=11)
plt.ylim(0.,0.5)
plt.yticks([0, 0.1, 0.2, 0.3,0.4])
plt.xticks([1,8,16,24,32,44], [1,16,32,256,1024,8192])
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)
plt.rcParams['hatch.linewidth'] = 10
legend=plt.legend(ncol=2,fontsize=10, handletextpad=0.1,handlelength=1.5,columnspacing=.5, loc='upper left' )
plt.grid(axis='y', zorder=-1)
plt.ylabel('Latency (ms)',fontdict=font2)
plt.xlabel('Batch size',fontdict=font2)
plt.axvline(x=16, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig("opcost.pdf")
plt.show()
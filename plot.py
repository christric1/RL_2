import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


yolov7    = [0.698, 0.592, 0.635]
yolov7_RL = [0.667, 0.538, 0.579]
index     = ['precision', 'recall', 'mAP0.5']

sns.set_style("darkgrid")
df = pd.DataFrame({'yolov7': yolov7, 'yolov7_RL': yolov7_RL}, index=index) \
                    .plot(rot=0, kind='bar', color=["#5975a4", "#cc8963"], width=0.3)
plt.title("Mince Pie Consumption in Seaborn style")
plt.ylim(0, 1)
plt.show()
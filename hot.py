import itertools
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
trans_mat = np.array([[8, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                      [0, 9, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0,10, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0,10, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 9, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 3, 0, 6, 0, 0],
                      [0, 1, 0, 1, 0, 0, 0, 0, 0, 8, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
                      ], dtype=int)
   
trans_prob_mat = (trans_mat.T).T

zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')

if True:
    label = ["     {}".format(i) for i in range(1, trans_mat.shape[0]+1)]
    df = pd.DataFrame(trans_prob_mat, index=label, columns=label)

    
    # Plot
    plt.figure(figsize=(7.5, 6.3))
    ax = sns.heatmap(df, xticklabels=df.corr().columns, 
                     yticklabels=df.corr().columns, cmap=plt.cm.gray_r,
                     linewidths=6, annot=True)
    plt.xlabel('预测身份', fontproperties=zhfont1, fontsize=18)
    plt.ylabel('真实身份', fontproperties=zhfont1, fontsize=18)
    # Decorations
    plt.xticks(fontsize=16,family='Times New Roman')
    plt.yticks(fontsize=16,family='Times New Roman')
    
    plt.tight_layout()
    plt.savefig('method_3.png', transparent=True, dpi=800)   
    plt.show()
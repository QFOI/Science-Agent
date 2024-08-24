from two_prompts import main
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
res = []
result = []
for _ in range(10):
    a,b = main()
    res.append(a)
    result.append(b)
def draw_func(f, x_label, arm_num):
        nums = 20
        x = np.linspace(1, nums, nums,endpoint=True, retstep=False, dtype=int, axis=0)
        y = []
        y_min = []
        y_max = []
        for __ in range(nums):
            sum = 0
            _min = 100
            _max = 0
            for _ in range(arm_num):
                sum += f[_][__]
                _min = min(_min, f[_][__])
                _max = max(_max, f[_][__])
            sum = sum/arm_num
            sum = sum/100
            _min = _min/100
            _max = _max/100
            y.append(sum)
            y_min.append(_min)
            y_max.append(_max)
        plt.xticks(np.arange(1, nums+1, 1))
        plt.plot(x, y, label='')
        plt.fill_between(x, y_min, y_max, color='gray', alpha=0.2, label='Range')
        plt.title(x_label)
        plt.xlabel('Times')
        plt.ylabel('Fields')
        plt.legend()
        plt.savefig(f'{x_label}.png', dpi=300, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.show()
draw_func(result, "Single Agent two prompts", 10)
draw_func(res, "Single Agent two prompts Yield", 10)
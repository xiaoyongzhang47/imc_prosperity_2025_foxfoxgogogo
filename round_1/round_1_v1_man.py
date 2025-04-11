import numpy as np

currency_names = {0: "Snowballs", 1: "Pizza's", 2: "Silicon Nuggets", 3: "Sea Shells"}


weights = np.array([[1,    1.45, 0.52, 0.72],
                    [0.7,  1,    0.31, 0.48],
                    [1.95, 3.1,  1,    1.49],
                    [1.34, 1.98, 0.64, 1   ]], dtype=float)



loop_verteces = np.zeros(4)

v0 = 3
profit = {}
for v1 in range(4):
    for v2 in range(4):
        for v3 in range(4):
            for v4 in range(4):
                
                profit[(v1, v2, v3, v4)] = weights[v0, v1] * weights[v1, v2] * weights[v2, v3] * weights[v3, v4] * weights[v4, v0]



best_loop = max(profit, key=profit.get)
best_loop_profit = profit.get(best_loop)

print(f"Best loop: {best_loop}, Profit: {best_loop_profit}")


# print(best_loop_profit)








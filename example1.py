import numpy as np

eps = np.zeros((2,2), dtype=int)
eps[0,1] = 1
eps[1,0] = -1

S = np.random.randn(2,2)
detS = np.linalg.det(S)

# compute eps_{ij} S^i_0 S^j_1  (indexes: i,j over rows; columns 0 and 1)
val = 0.0
for i in range(2):
    for j in range(2):
        val += eps[i,j] * S[i,0] * S[j,1]

print('detS =', detS)
print('eps contraction =', val)
# should be equal (within numerical precision)
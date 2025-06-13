import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

NMIN = 1000
NMAX = 1000
NSTEP = 100
DMIN = 1
DMAX = 50
DSTEP = 1
SMIN = DMAX
SMAX = 500
SSTEP = 10
TDIFF_MIN = 1
TDIFF_MAX = 3
TMIN = SMAX - TDIFF_MAX
TMAX = SMAX - TDIFF_MIN


N_LIST = np.arange(NMIN, NMAX+NSTEP, NSTEP)
S_LIST = np.arange(SMIN, SMAX+SSTEP, SSTEP)
D_LIST = np.arange(DMIN, DMAX+DSTEP, DSTEP)

ALL_P_LIST = []

ALL_MAX_LIST = []

for i in range(len(N_LIST)):
    N = N_LIST[i]
    predicted_N = int(N/np.e)
    SSTEP = 1
    SRANGE = 0.2*N*SSTEP

    # S_LIST = np.arange(predicted_N-SRANGE, predicted_N+SRANGE, SSTEP)
    # # S_LIST = [20, 30, 50]

 
    for j in range(len(S_LIST)):
        P_LIST = np.zeros(len(D_LIST), dtype=float)        
        MAX_P_LIST = []
        P_TRACKER = []

        S = S_LIST[j]
        for k in range(len(D_LIST)):
            D = D_LIST[k]            
            T = S-D
            B = N//S
            R = N%S
            # print(T, D, B, R, S/D)
            if R == 0:
                # print(B,S)
                P = np.float32((S/D)*np.log(B))
                # print(P)
            else: 
                P = np.float32(np.log((B+R))+(S/D-1)*np.log(B))
            P_LIST[k] = P
        # MAX_P_LIST.append(max(P_LIST))
        ALL_P_LIST.append(P_LIST)
        ALL_MAX_LIST.append(MAX_P_LIST)

ALL_P_ARRAY = np.array(ALL_P_LIST)
MAX_P_ARRAY = np.max(ALL_P_ARRAY, axis=0)

# plt.figure()
# for j in range(len(S_LIST)):
#     plt.plot(N_LIST, ALL_P_ARRAY[:, j])
# plt.xlabel('N')
# plt.ylabel('log P')
# plt.legend(S_LIST)
# plt.show()

plt.rcdefaults()
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['font.size'] = 24

print('sizes', D_LIST.size, MAX_P_ARRAY.size)

plt.figure(figsize=(10,8), dpi=300)
for i in range(len(S_LIST)):
    plt.plot(D_LIST.T, MAX_P_ARRAY, linewidth=3)
plt.xlabel('D', labelpad=10)
plt.ylabel('log C', labelpad=10)
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.title('N = {}'.format(N))
# plt.legend(S_LIST, loc='upper right')
plt.savefig("MaxCapacity{}.svg".format(NMAX))

plt.figure(figsize=(10,8), dpi=300)
for i in range(len(D_LIST)):
    plt.plot(S_LIST, ALL_P_ARRAY[:, i], linewidth=3)
plt.xlabel('S', labelpad=10)
plt.ylabel('log C', labelpad=10)
# plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.title('N = {}'.format(N))
# plt.legend(D_LIST, loc='upper left')
plt.savefig("MaxCapacity{}.svg".format(NMAX))

plt.show()
max_index = np.argmax(ALL_P_ARRAY[:, i])
print('Maximum Capacity at S =', S_LIST[max_index])

x_list = [2, 3, 4]
y1_list = [86.07, 61.51, 46.036]
y2_list = [100, 100, 100]

ax = plt.figure(figsize=(12,8), dpi=300).gca()

plt.plot(x_list, y1_list, label='Random', linewidth=4)
plt.plot(x_list, y2_list, label='Subsampling', linewidth=4)
plt.xlabel('S', labelpad=10)
plt.ylabel('Percent of Theoretical Max Capacity', labelpad=10)
# plt.xlim(xmin=0)
plt.ylim(ymin=0)
ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))

# plt.title('N = {}; MAX Threshold'.format(N))
plt.legend(loc='lower right')
plt.savefig("PctPerformance{}.svg".format(NMAX))

XMIN = 0.1
XMAX = 0.5
XSTEP = 0.1
X_LIST = np.arange(XMIN, XMAX+XSTEP, XSTEP)

NMIN = 2
NMAX = 8
NBASE = NMAX-NMIN
N_LIST = np.logspace(NMIN, NMAX, NBASE, endpoint=False)

ALL_Y_LIST = []
for i in range(len(N_LIST)):
    N = N_LIST[i]
    Y_LIST = []
    for j in range(len(X_LIST)):
        X = X_LIST[j]           # S/N
        Y_LIST.append(np.float32(np.log((X*N)*np.log(1/X))))
    ALL_Y_LIST.append(Y_LIST)

ALL_Y_ARRAY = np.array(ALL_Y_LIST)
N_LOG_LIST = np.arange(NMIN, NMAX, 1)
# N_LOG_LIST = [_**2 for _ in N_LOG_LIST]

plt.figure(figsize=(10,8), dpi=300)
for i in range(len(N_LIST)):
    plt.plot(X_LIST, ALL_Y_ARRAY[i, :], linewidth=3)
plt.xlabel('S/N', labelpad=10)
plt.ylabel('log(log(C))', labelpad=10)
plt.xlim(xmin=0)
plt.ylim(ymin=0)
# plt.title('N = {}'.format(N))
# plt.legend(X_LIST, loc='upper right')
plt.savefig("MaxCapacityVaryingNoverS.svg")

plt.figure(figsize=(10,8), dpi=300)
for i in range(len(X_LIST)):
    if i == 1:
        plt.plot(N_LOG_LIST, ALL_Y_ARRAY[:, i], linewidth=3)
plt.xlabel('log(N)', labelpad=10)
plt.ylabel('log(log(C))', labelpad=10)
# plt.xlim(xmin=0)
plt.ylim(ymin=0)
# plt.title('N = {}'.format(N))
# plt.legend(X_LIST, loc='upper right')
plt.savefig("MaxCapacityVaryingN.svg")
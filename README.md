# 需要修改的数据

## 1. random_write_disk

这个代表 1~MAX_TAG 的每个 tag 应该分到哪 REP_NUM 块硬盘上，以及 query_time 也是在这里获取的。
获取途径：

```py
# 第一个 py 文件
import numpy as np

MAX_TAG = 16

cnt = [0] * MAX_TAG
# tag.txt 中存储的是所有 Query 中 tag 的存储情况
with open("tag.txt", "r") as f:
    for i, line in enumerate(f):
        s = list(map(int, line.split(' ')))
        for j in range(len(s)):
            cnt[s[j] - 1] += 1

sorted_indices = np.argsort(cnt)[::-1]

print(sorted_indices + 1)
for i in range(MAX_TAG):
    print(cnt[sorted_indices], end=', ')
for i in range(MAX_TAG):
    print(cnt[i], end=', ') # 这个是 query_time


    
# 第二个 py 文件
import numpy as np

# 输入数组
ii = [14,  9,  3, 16,  5,  1, 12,  8, 15, 10,  4, 11,  2,  6, 13,  7]
arr = [714684, 678137, 675307, 672907, 656267, 655440, 623525, 498815,
       464401, 454897, 453136, 450530, 446193, 437673, 361323, 358938]

# 初始化负载数组
loads = np.zeros(10)  # 10个编号
assignments = [[] for _ in range(17)]  # 每个编号对应的元素列表

# 逐个元素进行分配
for i, elem in enumerate(arr):
    # 选择当前负载最小的三个编号
    sorted_indices = np.argsort(loads)[:3]  # 选择负载最小的三个编号
    # 将当前元素分配给这三个编号
    for idx in sorted_indices:
        assignments[ii[i]].append(idx)
        loads[idx] += elem  # 更新负载

# 输出每个编号的元素和总负载
for i in range(17):
    print('{', end='')
    for x in assignments[i]:
        print(x + 1, end=', ')
    print('},', end='\n')

# 输出每个编号的总负载
print(f"\n每个编号的总负载: {loads}")

```





## 2. total_init 函数

这里需要获取不同 size 的物体总存储的比例，决定对每个 size 要分配多少个位置。
获取途径：

`````py
import matplotlib.pyplot as plt
import numpy as np

MAX_SIZE = 5

adj = np.zeros((MAX_SIZE, 90000))
with open("size.txt", "r") as f:
    for i, line in enumerate(f):
        cnt = [0] * MAX_SIZE
        s = list(map(int, line.split(' ')))
        for j in range(len(s)):
            cnt[s[j] - 1] += 1
        for j in range(MAX_SIZE):
            adj[j][i] = cnt[j]


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

fig, axes = plt.subplots(4, 4, figsize=(20, 20))  # 4行4列，总共16张图
fig.suptitle('5 Variables Over Time', fontsize=16)


for i in range(MAX_SIZE):
    row = i // 4  # 子图的行索引
    col = i % 4   # 子图的列索引
    ax = axes[row, col]  # 获取当前子图
    
    # 获取当前行的数据
    data = adj[i]
    
    # 创建掩码，过滤掉值为 -1 的点
    mask = data != -1
    x_values = np.arange(len(data))[mask]  # x 轴为时间步长
    y_values = data[mask]  # y 轴为有效数据
    
    # # 对数据进行滑动平均（窗口大小为100）
    window_size = 100
    if len(y_values) > window_size:  # 确保数据足够长
        y_values = moving_average(y_values, window_size)
        x_values = x_values[:len(y_values)]  # 调整 x 轴长度
    else:
        y_values = y_values
        x_values = x_values
    
    # 绘制平滑后的曲线
    ax.plot(x_values, y_values, label=f'Variable {i}')
    ax.set_title(f'Variable {i}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()


# 调整子图间距
# plt.tight_layout()
# plt.show()

cnt = [0] * MAX_SIZE
for i in range(MAX_SIZE):
    cnt[i] = sum(adj[i])


tot = 0
for x in cnt:
    tot += x

mx = 1e9
best = []
for j in range(1, 1000):
    cur = [0] * MAX_SIZE
    for i in range(MAX_SIZE):
        cur[i] = j * cnt[i] / tot
    cost = 0
    for k in range(MAX_SIZE):
        cost += abs(round(cur[k]) - cur[k])
    if cost < mx:
        mx = cost
        best = cur

for i in range(MAX_SIZE):
    print(best[i], end=', ')
print()
for i in range(MAX_SIZE):
    print(round(best[i]), end=', ')
print()


`````



这里得到的解不一定精确，需要通过下面的方法求一晚上得到最优解，windows 版本的命令运行与这里不太一样，参见大电脑。

```py
import re
import time
import numpy as np
import subprocess

variables = 5
command = ['python', '../run.py', '../interactor/linux/interactor', '../data/sample_practice.in', './main']

def run():
    # 使用 subprocess.run 来执行命令，并捕获输出
    result = subprocess.run(
        command,
        capture_output=True, text=True
    )
    # 检查是否执行成功（返回码为0表示成功）
    if result.returncode == 0:
        pattern = r'"score":"(\d+\.\d+)"'
        match = re.search(pattern, result.stderr)
        if match:
            return float(match.group(1))
        else:
            return 0
    else:
        print("执行过程中出现错误:")
        print(result.stderr)
        exit(0)


def save(w, path):
    with open(path, 'w') as f:
        f.write(" ".join(map(str, w)))
    time.sleep(0.2)


def send_weight(w):
    with open('./weight.txt', 'w') as f:
        f.write(" ".join(map(str, w)))
    time.sleep(0.2)


def game(w):
    send_weight(w)
    score = run()
    return score


def progress_bar(current, total, msg=None):
    progress = current / total
    bar_length = 20  # Length of progress bar to display
    filled_length = int(round(bar_length * progress))
    bar = '#' * filled_length + '-' * (bar_length - filled_length)

    if msg:
        print(f'\r[{bar}] {progress * 100:.1f}% {msg}', end='')
    else:
        print(f'\r[{bar}] {progress * 100:.1f}%', end='')

    if current == total - 1:
        print()
    

if __name__ == '__main__':

    w = np.array([87, 58, 52, 27, 11])

    limit = np.array([
        [80, 95],
        [50, 65],
        [45, 55],
        [20, 30],
        [9, 14],
    ])
    print('Start optimizing')
    best_score = game(w)
    print(f'Initial score is {best_score}')
    epochs = 1000
    set = {}
    for epoch in range(epochs):
        random_w = np.array([np.random.randint(low, high) for low, high in limit])
        if random_w in set:
            continue
        set.add(random_w)
        score = game(random_w)
        if score > best_score:
            save(random_w, f'./checkpoints/_best_weight_{best_score}_.txt')
            best_score = score
        
        progress_bar(epoch+1, epochs, f'score: {score}, best_score: {best_score}')
    
```









## 3. 数据文件的获取

记得不要先输出 n，因为不好处理

```C++
#include <bits/stdc++.h>
using namespace std;

#define MAX_TAG (16)
#define MAX_DISK_NUM (10)
#define MAX_DISK_SIZE (16384)
#define MAX_REQUEST_NUM (30000000)
#define MAX_OBJECT_NUM (100000)
#define REP_NUM (3)
#define FRE_PER_SLICING (1800)
#define EXTRA_TIME (105)
#define MAX_EPOCH (50)

int T, M, N, V, G;
int fre_del[MAX_TAG + 1][MAX_EPOCH];
int fre_write[MAX_TAG + 1][MAX_EPOCH];
int fre_read[MAX_TAG + 1][MAX_EPOCH];

struct Disk {
	pair<int, int> d[MAX_DISK_SIZE];
	int head = 0;
	int siz = 0;
	Disk() {
		for (int i = 0; i < MAX_DISK_SIZE; i++) {
			d[i] = {-1, -1};
		}
	}
	int size() {
		return siz;
	}
	int last() {
		return V - siz;
	}
	vector<int> Write(int obj_id, int obj_size, int obj_tag) {
		vector<int> write_idx;
		for (int i = 0; i < V; i++) {
			if (write_idx.size() == obj_size) break;
			if (d[i].first != -1) continue;
			d[i] = {obj_id, write_idx.size()};
			write_idx.emplace_back(i);
			siz++;
		}
		assert(write_idx.size() == obj_size);
		return write_idx;
	}
	void erase(int erase_idx) {
		assert(d[erase_idx].first != -1);
		d[erase_idx] = {-1, -1};
		siz--;
	}
};
Disk disk[MAX_DISK_NUM]; // 硬盘

struct Request {
	int obj_id = -1;
	vector<bool> ned;
	bool is_done = false;
	void update() {
		is_done = accumulate(ned.begin(), ned.end(), 0) == ned.size();
	}
};
Request requests[MAX_REQUEST_NUM + 1];

struct Object {
	vector<int> bel;
	vector<int> unit[REP_NUM];
	int size = 0;
	int tag = -1;
	bool is_delete = false;
};
Object objects[MAX_OBJECT_NUM + 1];
vector<int> query[MAX_OBJECT_NUM + 1]; // 每个对象的查询








// ------------------------------------ 全局预处理 ----------------------------------------

void Pre_Process() {
	
}




// ------------------------------------ 删除 ----------------------------------------

void Delete_Action() {
	int n_delete;
	cin >> n_delete;
	static int delete_id[MAX_OBJECT_NUM];
	for (int i = 0; i < n_delete; i++) {
		cin >> delete_id[i];
	}
    // cout << n_delete;
    // for (int i = 0; i < n_delete; i++) {
    //     cout << " " << delete_id[i];
    // }
    // cout << "\n";
}




// ------------------------------------ 写入 ----------------------------------------

void Write_Action() {
	int n_write;
	cin >> n_write;
	vector<int> write_id(n_write), write_obj_size(n_write), write_obj_tag(n_write);
	for (int i = 0; i < n_write; i++) {
		cin >> write_id[i] >> write_obj_size[i] >> write_obj_tag[i];
	}
    for (int i = 0; i < n_write; i++) {
		int obj_id = write_id[i];
		int obj_size = write_obj_size[i];
		int obj_tag = write_obj_tag[i];

		objects[obj_id].tag = obj_tag;
		objects[obj_id].size = obj_size;
	}
    for (int i = 0; i < n_write; i++) {
        cout << " " << write_id[i];
    }
    cout << "\n";
}



// ------------------------------------ 读取 ----------------------------------------

void Read_Action() {
	int n_read;
	cin >> n_read;
	vector<int> request_id(n_read), read_id(n_read);
	for (int i = 0; i < n_read; i++) {
		cin >> request_id[i] >> read_id[i];
	}
    // cout << n_read;
    // for (int i = 0; i < n_read; i++) {
    //     cout << " " << read_id[i];
    // }
    // cout << "\n";
}



// ------------------------------------ 磁头移动 ----------------------------------------

void Solve() {
	Delete_Action();
	Write_Action();
	Read_Action();
}



 

void TimeStamp() {
	string pattern;
	int timeStamp;
	cin >> pattern >> timeStamp;
}

int main() {
    freopen("sample_practice.in", "r", stdin);
    freopen("query_id.txt", "w", stdout);
	cin >> T >> M >> N >> V >> G;
	int batch_num = (T + FRE_PER_SLICING - 1) / FRE_PER_SLICING;
	for (int i = 1; i <= M; i++) {
		for (int j = 1; j <= batch_num; j++) {
			cin >> fre_del[i][j];
		}
	}
	for (int i = 1; i <= M; i++) {
		for (int j = 1; j <= batch_num; j++) {
			cin >> fre_write[i][j];
		}
	}
	for (int i = 1; i <= M; i++) {
		for (int j = 1; j <= batch_num; j++) {
			cin >> fre_read[i][j];
		}
	}
	Pre_Process();
	for (int i = 1; i <= T + 105; i++) {
		TimeStamp();
		Solve();
	}
	return 0;
}
```
































































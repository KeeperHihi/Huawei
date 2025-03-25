// 不再严格划分 size，有连续空位置就存放
// sublime 里那段代码要设计一种算法更加合理地分配空间，不能让 16 号出现在所有盘里
// 存储的时候先存 size 大的，再存 size 小的，石头 -> 沙子 -> 水




#include <bits/stdc++.h>
using namespace std;

// #define DEBUG

#ifdef DEBUG
#define UPDATE_DISK_SCORE_FREQUENCY (10)
#define MAX_DISK_SIZE (5792)
#define BLOCK_NUM (32)
const int BLOCK_SIZE = MAX_DISK_SIZE / BLOCK_NUM;
#else
#define UPDATE_DISK_SCORE_FREQUENCY (2)
#define MAX_DISK_SIZE (16384)
#define BLOCK_NUM (32)
const int BLOCK_SIZE = MAX_DISK_SIZE / BLOCK_NUM;
#endif

#define MAX_TAG (16)
#define MAX_SIZE (5)
#define MAX_DISK_NUM (10)

const int DISK_SPLIT_BLOCK = MAX_DISK_SIZE / 35.7;
const int DISK_SPLIT_1 = DISK_SPLIT_BLOCK * 6;
const int DISK_SPLIT_2 = DISK_SPLIT_BLOCK * 14;
const int DISK_SPLIT_3 = DISK_SPLIT_BLOCK * 24.5;
const int DISK_SPLIT_4 = DISK_SPLIT_BLOCK * 31.7;
const int DISK_SPLIT_5 = DISK_SPLIT_BLOCK * 35.7;
// 60 : 40 : 35 : 18 : 8     sum = 161

#define JUMP_FREQUENCY (5)

#define MAX_REQUEST_NUM (30000000)
#define MAX_OBJECT_NUM (100000)
#define REP_NUM (3)
#define FRE_PER_SLICING (1800)
#define EXTRA_TIME (105)
#define MAX_EPOCH (50)
#define MAX_WRITE_LEN (100005)
#define INF (1000000000)
#define PREDICT (2)

#define DROP_SCORE (0)
#define MAX_DROP_NUM (1000000)

#define SEED (11111111)

vector<int> query_times = {0, 2136, 1048, 2276, 1996, 1208, 799, 2048, 1695, 782, 1679, 923, 1876, 764, 1421, 824, 2349};


int T, M, N, V, G;
int fre_del[MAX_TAG + 1][MAX_EPOCH];
int fre_write[MAX_TAG + 1][MAX_EPOCH];
int fre_read[MAX_TAG + 1][MAX_EPOCH];

vector<vector<tuple<int, int, int>>> disk_manage; // 前提是必须有 10 块！
vector<vector<int>> disk_select;
int pre_cost[MAX_DISK_NUM];
char pre_move[MAX_DISK_NUM];

int timestamp = 0; // 全局时间戳



// ------------------------------------ 全局函数声明 ----------------------------------------

float Get_Pos_Score(int disk_id, int pos, int time);   // 获取一个硬盘上一个位置 pos 的得分
void Pre_Process(); 					     		   // 对总和输入数据的预处理
void total_init();						     		   // 预处理乱七八糟的东西，比如 disk 的余量集合
bool Random_Appear(int p);				     		   // 判断概率 p% 是否发生
pair<float, int> Simulate(int disk_id, int idx, int time, char &pre_mov, int &pre_cos);



// ------------------------------------ 结构体定义 ----------------------------------------

struct Disk {
	pair<int, int> d[MAX_DISK_SIZE];
	set<int> space[MAX_SIZE + 1];
	vector<int> has_tag;
	int color_tag[MAX_DISK_SIZE];
	float score[BLOCK_NUM];
	int head = 0;
	int siz = 0;
	int disk_id = 0;
	Disk() {
		for (int i = 0; i < MAX_DISK_SIZE; i++) {
			d[i] = {-1, -1};
		}
	}
	int size() {
		return siz;
	}
	bool capacity(int obj_tag, int obj_size, bool is_limit) {
		// is_limit 代表的是是否要强制这个 obj 放在自己 tag 的位置里
		if (!is_limit) return space[obj_size].size() > 0;
		bool cap = 0;
		for (auto pos : space[obj_size]) {
			if (color_tag[pos] == obj_tag) {
				cap = 1;
				break;
			}
		}
		return cap;
	}
	
	void Cal_Block_Score() {
		for (int block = 0; block < BLOCK_NUM; block++) {
			int i = block * BLOCK_SIZE;
			float sc = 0;
			char pre_mov = 'j';
			int pre_cos = 0;
			for (int t = timestamp + 1; t < timestamp + PREDICT; t++) {
				// auto [score, idx] = Simulate(disk_id, i, t, pre_mov, pre_cos);
				// i = idx;
				// sc += score;
				for (int cnt = BLOCK_SIZE; cnt--; i = (i + 1) % V) {
					sc += Get_Pos_Score(disk_id, i, t);
				}
			}
			score[block] = sc;

			// int l = block * BLOCK_SIZE;
			// int r = l + BLOCK_SIZE - 1;
			// float sc = 0;
			// for (int i = l; i <= r; i++) {
			// 	sc += Get_Pos_Score(disk_id, i, timestamp);
			// }
			// score[block] = sc;
		}
	}
	int max_score_pos = 0;  // 价值最大块位置
	double max_score = -1;  // 价值最大块价值
	double cur_score = 0;   // 当前位置向前走 PREDICT 个块的价值
	void Cal_Score() {	    // 计算下一个时刻走 PREDICT - 1 个块的价值
		Cal_Block_Score();
		max_score = -1;
		int block = -1;
		for (int i = 0; i < BLOCK_NUM; i++) {
			if (score[i] > max_score) {
				max_score = score[i];
				block = i;
			}
		}
		cur_score = 0;
		int i = head;
		char pre_mov = pre_move[disk_id];
		int pre_cos = pre_cost[disk_id];
		for (int t = timestamp; t < timestamp + PREDICT; t++) {
			// auto [score, idx] = Simulate(disk_id, i, t, pre_mov, pre_cos);
			// i = idx;
			// cur_score += score;
			for (int cnt = BLOCK_SIZE; cnt--; i = (i + 1) % V) {
				cur_score += Get_Pos_Score(disk_id, i, t);
			}
		}
		assert(block != -1);
		max_score_pos = block * BLOCK_SIZE;
	}
	int Write(int obj_id, int obj_size, int obj_tag, bool is_limit) {
		int write_idx;
		siz += obj_size;

		// vector<int> er;
		// for (auto it = space.begin(); write_idx.size() < obj_size; it++) {
			// 	assert(d[*it].first == -1);
			// 	d[*it] = {obj_id, write_idx.size()};
		// 	write_idx.emplace_back(*it);
		// 	er.emplace_back(*it);
		// }	
		// for (auto t : er) {
			// 	space.erase(t);
		// }
		assert(space[obj_size].size());
		auto it = space[obj_size].begin();
		while (is_limit && it != space[obj_size].end() && color_tag[*it] != obj_tag) {
			it++;
		}
		// if (it == space[obj_size].end()) {
		// 	cerr << timestamp << endl;
		// 	cerr << obj_id << ' ' << obj_size << ' ' << obj_tag << endl;
		// 	cerr << disk_id << endl;
		// 	cerr << space[obj_size].size() << endl;
		// 	for (auto t : space[obj_size]) {
		// 		cerr << t << ' ' << color_tag[t] << endl;
		// 	}
		// }
		assert(it != space[obj_size].end());
		// for (int i = 0; i < obj_size; i++) {
		// 	if (d[(*it) + i].first != -1) {
		// 		for (int i = 0; i < obj_size; i++) cerr << d[(*it) + i].first << ' '; cerr << endl;
		// 	}
		// 	assert(d[(*it) + i].first == -1);
		// }
		for (int i = *it, size = 0; size < obj_size; i++, size++) {
			d[i] = {obj_id, size};
		}
		write_idx = *it;
		space[obj_size].erase(it);
		
		// for (auto it = space[obj_size].begin(); write_idx.size() < obj_size && it != space[obj_size].end(); ) {
		// 	auto p = it;
		// 	it++;
		// 	assert(d[*p].first == -1);
		// 	d[*p] = {obj_id, write_idx.size()};
		// 	write_idx.emplace_back(*p);
		// 	space[obj_size].erase(p);
		// }	
		
		// for (int i = 0; i < V; i++) {
		// 	if (write_idx.size() == obj_size) break;
		// 	if (d[i].first != -1) continue;
		// 	d[i] = {obj_id, write_idx.size()};
		// 	write_idx.emplace_back(i);
		// }
		
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
	int query_time = 0;
	int obj_id = -1;
	int ned = 0;
	int mask = -1;
	bool is_done() {
		return ned == mask;
	}
	bool has_part(int part) {
		return ned >> part & 1;
	}
	void set(int part) {
		ned |= 1 << part;
	}
	void erase(int part) {
		ned &= ~(1 << part);
	}
};
vector<Request> requests(MAX_REQUEST_NUM + 1); // 使用 vector 是因为静态空间不够了


struct Object {
	vector<int> bel;
	int unit[REP_NUM];
	int size = 0;
	int tag = -1;
	bool is_delete = false;
};
Object objects[MAX_OBJECT_NUM + 1];
unordered_set<int> query[MAX_OBJECT_NUM + 1]; // 每个对象的查询






// ------------------------------------ 全局预处理 ----------------------------------------

void Pre_Process() {
	// int batch_num = (T + FRE_PER_SLICING - 1) / FRE_PER_SLICING;
	// vector<int> tag(MAX_TAG + 1);
	// for (int i = 1; i <= MAX_TAG; i++) {
	// 	for (int j = 1; j <= batch_num; j++) {
	// 		cerr << fre_write[i][j] << ' ';
	// 		tag[i] += fre_write[i][j];
	// 	}
	// 	cerr << endl;
	// }
	// for (int i = 1; i <= MAX_TAG; i++) {
	// 	cerr << tag[i] << ", ";
	// }
	cout << "OK\n";
	cout.flush();
}

// 60 : 40 : 35 : 18 : 8     sum = 161
void total_init() {
	vector<int> best_w = {212, 149, 132, 68, 28};
	vector<int> count(17);
	for (int i = 0; i < MAX_DISK_NUM; i++) {
		disk[i].disk_id = i;
		vector<int> count(6);
		// int j = 0;
		// while (j < V) {
		// 	for (int c = best_w[0]; j < V && c--; j++) {
		// 		disk[i].space[1].insert(j);
		// 		count[1]++;
		// 	}
		// 	for (int c = best_w[1]; j + 1 < V && c--; j += 2) {
		// 		disk[i].space[2].insert(j);
		// 		count[2]++;
		// 	}
		// 	for (int c = best_w[2]; j + 2 < V && c--; j += 3) {
		// 		disk[i].space[3].insert(j);
		// 		count[3]++;
		// 	}
		// 	for (int c = best_w[3]; j + 3 < V && c--; j += 4) {
		// 		disk[i].space[4].insert(j);
		// 		count[4]++;
		// 	}
		// 	for (int c = best_w[4]; j + 4 < V && c--; j += 5) {
		// 		disk[i].space[5].insert(j);
		// 		count[5]++;
		// 	}
		// }

		int divide = 0;
		for (int j = 0; j < MAX_SIZE; j++) {
			divide += (j + 1) * best_w[j];
		}

		// 染色，对应位置只能存储对应 tag
		int tot = 0;
		for (auto tag : disk[i].has_tag) {
			tot += query_times[tag];
		}
		int idx = 0;
		// cerr << "tot = " << tot << endl;
		for (auto tag : disk[i].has_tag) {
			int len = 1. * query_times[tag] / tot * V;
			vector<int> cnt(MAX_SIZE);
			for (int j = 0; j < MAX_SIZE; j++) {
				cnt[j] = 1. * best_w[j] / divide * len;
			}
			int pre_idx = idx; 
			// if (i == 0) {
			// 	cerr << "i = 0, " << best_w[0] << ' ' << divide << ' ' << len << endl;
			// 	for (int j = 0; j < MAX_SIZE; j++) {
			// 		cerr << cnt[j] << ' ';
			// 	}
			// 	cerr << endl;
			// }

			// 把 size 大的放在前面，能提升一点点
			for (int c = cnt[4]; idx + 4 < min(V, pre_idx + len) && c--; idx += 5) {
				disk[i].space[5].insert(idx);
				count[5]++;
			}
			for (int c = cnt[3]; idx + 3 < min(V, pre_idx + len) && c--; idx += 4) {
				disk[i].space[4].insert(idx);
				count[4]++;
			}
			for (int c = cnt[2]; idx + 2 < min(V, pre_idx + len) && c--; idx += 3) {
				disk[i].space[3].insert(idx);
				count[3]++;
			}
			for (int c = cnt[1]; idx + 1 < min(V, pre_idx + len) && c--; idx += 2) {
				disk[i].space[2].insert(idx);
				count[2]++;
			}
			for (int c = cnt[0]; idx < min(V, pre_idx + len) && c--; idx++) {
				disk[i].space[1].insert(idx);
				count[1]++;
			}
			

			idx = pre_idx;
			for (int cnt = len; idx < V && cnt--; idx++) {
				disk[i].color_tag[idx] = tag;
			}
		}
		
		// 理论比例：972 648 567 291 129
		// for (int j = 0; j < DISK_SPLIT_1; j++) {
		// 	disk[i].space[1].insert(j);
		// 	count[1]++;
		// }
		// for (int j = DISK_SPLIT_1; j < DISK_SPLIT_2 - 1; j += 2) {
		// 	disk[i].space[2].insert(j);
		// 	count[2]++;
		// }
		// for (int j = DISK_SPLIT_2; j < DISK_SPLIT_3 - 2; j += 3) {
		// 	disk[i].space[3].insert(j);
		// 	count[3]++;
		// }
		// for (int j = DISK_SPLIT_3; j < DISK_SPLIT_4 - 3; j += 4) {
		// 	disk[i].space[4].insert(j);
		// 	count[4]++;
		// }
		// for (int j = DISK_SPLIT_4; j < DISK_SPLIT_5 - 4; j += 5) {
		// 	disk[i].space[5].insert(j);
		// 	count[5]++;
		// }
		// for (int i = 1; i <= 5; i++) {
		// 	cerr << count[i] << " \n"[i == 5];
		// }
	}
}


// mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
mt19937_64 rng(SEED);
bool Random_Appear(int p) {
	return rng() % 100 + 1 <= p;
}

// 分数应该与 obj_size 正相关
// 分数应该与 wait_time 正相关 (因为目标是优先提高命中率？)
// 分数应该与 predict_delete_time 负相关
int Predict_Delete_Time(int obj_id) {
	int obj_size = objects[obj_id].size;
	int obj_tag = objects[obj_id].tag;
	int predict_delete_time = INF;
	//
	//
	//
	return predict_delete_time;
}

float Get_Pos_Score(int disk_id, int pos, int time) {
	int obj_id = disk[disk_id].d[pos].first;	
	int obj_size = objects[obj_id].size;
	float score = 0;
	for (auto qry : query[obj_id]) {
		int x = time - requests[qry].query_time;
		float g = (obj_size + 1) * 0.5;
		float f;
		if (x <= 10) {
			f = -0.005 * x + 1;
		} else if (x <= 105) {
			f = -0.01 * x + 1.05;
		} else {
			f = 0.;
		}
		score += f * g;
	}
	return score;
}


// 注意这个函数绝对不能放到多线程里用！！！！！！！！！
pair<float, int> Simulate(int disk_id, int idx, int time, char &pre_mov, int &pre_cos) {
	int step = G;

	float score = 0;
	vector<pair<int, int>> change;
	
	while (step) {
		auto [obj_id, obj_part] = disk[disk_id].d[idx];
		int cost = INF;
		if (pre_mov == 'r') {
			cost = max(16, (int)ceil(pre_cos * 0.8));
		} else {
			cost = 64;
		}
		if (step < cost) {
			break;
		}
		bool is_hit = false;
		for (auto it = query[obj_id].begin(); it != query[obj_id].end(); it++) {
			int qry = *it;
			if (requests[qry].is_done()) continue;
			bool has = requests[qry].has_part(obj_part);
			if (has) continue;
			is_hit = true;
			change.push_back({qry, obj_part});
			requests[qry].set(obj_part);
			if (requests[qry].is_done()) {
				int x = time - requests[qry].query_time;
				double g = (objects[obj_id].size + 1) * 0.5;
				double f;
				if (x <= 10) {
					f = -0.005 * x + 1;
				} else if (x <= 105) {
					f = -0.01 * x + 1.05;
				} else {
					f = 0.;
				}
				score += f * g;
			}
		}
		if (is_hit) {
			pre_mov = 'r';
			step -= cost;
		} else {
			pre_mov = 'p';
			step--;
		}
		pre_cos = cost;
		idx = (idx + 1) % V;
	}
	for (auto [qry, obj_part] : change) {
		requests[qry].erase(obj_part);
	}
	return {score, idx};
}






// ------------------------------------ 删除 ----------------------------------------

void Delete_Action() {
	int n_delete;
	cin >> n_delete;
	static int delete_id[MAX_OBJECT_NUM];
	for (int i = 0; i < n_delete; i++) {
		cin >> delete_id[i];
	}
	int n_abort = 0;
	vector<int> aborts;
	for (int i = 0; i < n_delete; i++) {
		int obj_id = delete_id[i];
		for (auto query_idx : query[obj_id]) {
			if (requests[query_idx].is_done() == false) {
				n_abort++;
				aborts.emplace_back(query_idx);
			}
		}
		query[obj_id].clear();
	}
	cout << n_abort << "\n";
	for (auto fail : aborts) {
		cout << fail << "\n";
	}
	cout.flush();


	for (int i = 0; i < n_delete; i++) {
		int obj_id = delete_id[i];
		for (int j = 0; j < REP_NUM; j++) {
			int obj_size = objects[obj_id].size;
			int disk_id = objects[obj_id].bel[j];
			int block = objects[obj_id].unit[j];
			disk[disk_id].space[obj_size].insert(block);
			for (int size = 0; size < obj_size; size++) {
				disk[disk_id].erase(block + size);
			}
		}
		objects[obj_id].is_delete = true;
	}
}




// ------------------------------------ 写入 ----------------------------------------

vector<vector<int>> random_write_disk;
vector<vector<int>> second_choice;

void hash_init() {
	int idx = 0;
	random_write_disk = {
		{},
		{0, 1, 5, },
		{3, 8, 2, },
		{7, 4, 9, },
		{8, 7, 9, },
		{5, 8, 2, },
		{7, 9, 3, },
		{2, 4, 6, },
		{0, 1, 3, },
		{3, 6, 5, },
		{2, 6, 5, },
		{4, 0, 1, },
		{7, 4, 9, },
		{6, 5, 8, },
		{0, 1, 2, },
		{6, 5, 8, },
		{8, 3, 6, },
	};
	second_choice.resize(MAX_TAG + 1);
	for (int i = 1; i <= MAX_TAG; i++) {
		set<int> oth;
		while (oth.size() < REP_NUM) {
			int x = rng() % 10;
			if (oth.count(x) || find(random_write_disk[i].begin(), random_write_disk[i].end(), x) != random_write_disk[i].end()) {
				continue;
			}
			oth.insert(x);
		}
		for (auto hash_id : oth) {
			second_choice[i].push_back(hash_id);
		}
	}
	for (int i = 1; i <= MAX_TAG; i++) {
		for (auto disk_id : random_write_disk[i]) {
			disk[disk_id].has_tag.emplace_back(i);
		}
	}
	// for (int i = 1; i <= MAX_TAG; i++) {
	// 	for (int j = 0; j < REP_NUM; j++) {
	// 		random_write_disk[i].emplace_back(idx);
	// 		idx = (idx + 1) % N;
	// 	}
	// 	// set<int> set;
	// 	// while (set.size() < REP_NUM) {
	// 	// 	set.insert(rng() % N);
	// 	// }
	// 	// for (auto disk_id : set) {
	// 	// 	random_write_disk[i].emplace_back(disk_id);
	// 	// }
	// }

	// for (int i = 1; i <= 16; i++) {
	// 	for (auto t : random_write_disk[i]) {
	// 		cerr << t << ' ';
	// 	}
	// 	cerr << endl;
	// }
	// for (int i = 0; i < N; i++) {
	// 	for (auto t : random_write_disk[i]) {
	// 		cout << t << ' ';
	// 	}
	// 	cout << endl;
	// }
}

vector<pair<int, int>> Decide_Write_disk(int obj_id, int obj_size, int obj_tag) {
	static int idx = 0;
	// select 用位表示选过的 disk
	int select = 0;
	vector<pair<int, int>> write_disk;

	// 随机分布 -----------------------------------------------------
	// for (; write_disk.size() < REP_NUM; idx = (idx + 1) % N) {
	// 	if (find(write_disk.begin(), write_disk.end(), idx) != write_disk.end()) continue;
	// 	if (!disk[idx].capacity(obj_size)) continue;
	// 	write_disk.emplace_back(idx);
	// }
	// assert(write_disk.size() == REP_NUM);
	// return write_disk;
	// ---------------------------------------------------------------

	// size 法，把 size 相同的放到一起，但效果并不好
	// for (auto disk_id : disk_select[obj_size]) {
	// 	if (disk[disk_id].capacity(obj_size)) continue;
	// 	write_disk.emplace_back(disk_id);
	// }

	// hash 法，把 hash 值相同的放到一起
	for (auto hash_id : random_write_disk[obj_tag]) {
		if (disk[hash_id].capacity(obj_tag, obj_size, 1)) {
			write_disk.push_back({hash_id, 1});
			select |= 1 << hash_id;
		}
	}

	for (auto hash_id : second_choice[obj_tag]) {
		if (write_disk.size() < REP_NUM && disk[hash_id].capacity(obj_tag, obj_size, 0)) {
			write_disk.push_back({hash_id, 0});
			select |= 1 << hash_id;
		}
	}

	// 查缺补漏，如果不够 REP_NUM 个再顺序选几个
	for (int cnt = 0; write_disk.size() < REP_NUM; cnt++, idx = (idx + 1) % N) {
		assert(cnt <= N);
		if (select >> idx & 1) continue;
		if (!disk[idx].capacity(obj_tag, obj_size, 0)) continue;
		write_disk.push_back({idx, 0});
	}
	assert(write_disk.size() == REP_NUM);
	return write_disk;
}










void Write_Action() {
	int n_write;
	cin >> n_write;
	static int write_id[MAX_WRITE_LEN], write_obj_size[MAX_WRITE_LEN], write_obj_tag[MAX_WRITE_LEN];
	vector<vector<pair<int, int>>> tag_vec(MAX_TAG + 1);
	for (int i = 0; i < n_write; i++) {
		cin >> write_id[i] >> write_obj_size[i] >> write_obj_tag[i];
		tag_vec[write_obj_tag[i]].emplace_back(write_id[i], write_obj_size[i]);
	}

	// 简单把所有相同 tag 的先放到一起存储，后续具体放在哪里由 Decide_Write_Disk 来考虑
	for (int obj_tag = 1; obj_tag <= MAX_TAG; obj_tag++) {
		auto &vec = tag_vec[obj_tag];
		for (auto [obj_id, obj_size] : vec) {
			auto write_disk = Decide_Write_disk(obj_id, obj_size, obj_tag);
			// assert(write_disk.size() == REP_NUM);
			// for (auto t : write_disk) {
			// 	assert(disk[t].last() >= obj_size);
			// }
			
			for (auto [disk_id, is_limit] : write_disk) {
				objects[obj_id].bel.push_back(disk_id);
			}
			objects[obj_id].size = obj_size;
			objects[obj_id].tag = obj_tag;
			objects[obj_id].is_delete = false;
			
			for (int j = 0; j < REP_NUM; j++) {
				auto [disk_idx, is_limit] = write_disk[j];
				int write_idx = disk[disk_idx].Write(obj_id, obj_size, obj_tag, is_limit);
				// cout << "obj: " << obj_id << ' ' << obj_size << " WriteDisk: " << disk_idx << " WriteIdx: ";
				// for (auto t : write_idx) cout << t << " "; cout << endl;
				objects[obj_id].unit[j] = write_idx;
			}

			
			cout << obj_id << "\n";
			for (int j = 0; j < REP_NUM; j++) {
				cout << objects[obj_id].bel[j] + 1;
				for (int k = 0; k < objects[obj_id].size; k++) {
					cout << " " << objects[obj_id].unit[j] + k + 1;
				}
				cout << "\n";
			}
		}
	}

	// for (int i = 0; i < n_write; i++) {
	// 	int obj_id = write_id[i];
	// 	int obj_size = write_obj_size[i];
	// 	int obj_tag = write_obj_tag[i];

	// 	auto write_disk = Decide_Write_disk(obj_id, obj_size, obj_tag);
	// 	assert(write_disk.size() == REP_NUM);

	// 	objects[obj_id].bel = write_disk;
	// 	objects[obj_id].size = obj_size;
	// 	objects[obj_id].tag = obj_tag;
	// 	objects[obj_id].is_delete = false;
	// 	for (int j = 0; j < REP_NUM; j++) {
	// 		objects[obj_id].unit[j].clear();
	// 	}

	// 	for (int j = 0; j < REP_NUM; j++) {
	// 		int disk_idx = write_disk[j];
	// 		auto write_idx = disk[disk_idx].Write(obj_id, obj_size, obj_tag);
	// 		// cout << "obj: " << obj_id << ' ' << obj_size << " WriteDisk: " << disk_idx << " WriteIdx: ";
	// 		// for (auto t : write_idx) cout << t << " "; cout << endl;
	// 		objects[obj_id].unit[j] = write_idx;
	// 	}

	// 	cout << obj_id << "\n";
	// 	for (int j = 0; j < REP_NUM; j++) {
	// 		cout << objects[obj_id].bel[j] + 1;
	// 		assert(objects[obj_id].unit[j].size() == obj_size);
	// 		for (int k = 0; k < objects[obj_id].size; k++) {
	// 			cout << " " << objects[obj_id].unit[j][k] + 1;
	// 		}
	// 		cout << "\n";
	// 	}
	// }
	cout.flush();
}



// ------------------------------------ 读取 ----------------------------------------

void Read_Action() {
	int n_read;
	cin >> n_read;
	vector<int> request_id(n_read), read_id(n_read);
	for (int i = 0; i < n_read; i++) {
		cin >> request_id[i] >> read_id[i];
	}

	for (int i = 0; i < n_read; i++) {
		int qry_id = request_id[i];
		int obj_id = read_id[i];
		requests[qry_id] = {
			timestamp,
			obj_id,
			0,
			(1 << objects[obj_id].size) - 1
		};
		query[obj_id].insert(qry_id);
	}
}



// ------------------------------------ 磁头移动 ----------------------------------------

int Decide_Jump_Pos(int disk_id) {
	return disk[disk_id].max_score_pos;
}

void Process(int i) {
	disk[i].Cal_Score();
}

int cnt = 0;
int drop = 0;
void Move() {
	// if (timestamp % 1800 == 1) {
	// 	// cerr << "time = " << timestamp << endl;
	// 	// for (int i = 0; i < N; i++) {
	// 	// 	for (int j = 0; j < BLOCK_NUM; j++) {
	// 		// 		cerr << disk[i].score[j] << " ";
	// 	// 	}
	// 	// 	cerr << endl;
	// 	// }
	// 	cerr << "jump fre = " << 100. * cnt / N / 1800 << "%" << endl;
	// 	// for (int i = 0; i < N; i++) {
	// 	// 	cerr << i << ' ' << disk[i].cur_score << ' ' << disk[i].max_score << endl;
	// 	// }
	// 	cnt = 0;
	// }
	// if (timestamp % 1800 == 1) {
	// 	// cerr << "time = " << timestamp << endl;
	// 	// for (int i = 0; i < N; i++) {
	// 	// 	for (int j = 0; j < BLOCK_NUM; j++) {
	// 		// 		cerr << disk[i].score[j] << " ";
	// 	// 	}
	// 	// 	cerr << endl;
	// 	// }
	// 	// cerr << "drop = " << 100. * drop / N / 1800 << "%" << endl;
	// 	cerr << "drop = " << drop / 1800. / N << endl;
	// 	// for (int i = 0; i < N; i++) {
	// 	// 	cerr << i << ' ' << disk[i].cur_score << ' ' << disk[i].max_score << endl;
	// 	// }
	// 	cnt = 0;
	// }
	
	
	vector<int> finish_qid;
	if (timestamp % UPDATE_DISK_SCORE_FREQUENCY == 0) {
		// for (int i = 0; i < N; i++) {
		// 	Process(i);
		// }
		vector<thread> threads;
		for (int i = 0; i < N; i++) {
			threads.emplace_back(Process, i);
		}
		for (auto &thread : threads) {
			thread.join();
		}
	}
	vector<int> drop_num(N);
	vector<int> pos(N);
	iota(pos.begin(), pos.end(), 0);
	shuffle(pos.begin(), pos.end(), rng);
	vector<string> moves(N);
	// for (int i = 0; i < N; i++) {
	for (auto i : pos) {
		string move;
		int step = G;

		// 方案零：每次更新完考虑跳跃
		// if (timestamp % UPDATE_DISK_SCORE_FREQUENCY == 0 && disk[i].cur_score < disk[i].max_score && Random_Appear(JUMP_FREQUENCY)) {
		// 	cnt++;
		// 	int jump_to = disk[i].max_score_pos;
		// 	disk[i].head = jump_to;
		// 	cout << "j " << jump_to + 1 << "\n";
		// 	pre_move[i] = 'j';
		// 	pre_cost[i] = 0;
		// 	continue;
		// }

		// 方案一：比较往前走两个块根跳转在走一个块的价值决定是否 jump
		if (disk[i].cur_score < disk[i].max_score && Random_Appear(JUMP_FREQUENCY)) {
			cnt++;
			int jump_to = disk[i].max_score_pos;
			disk[i].head = jump_to;
			moves[i] = "j " + to_string(jump_to + 1);
			pre_move[i] = 'j';
			pre_cost[i] = 0;
			continue;
		}
		
		while (step) {
			auto [obj_id, obj_part] = disk[i].d[disk[i].head];
			int cost = INF;
			if (pre_move[i] == 'r') {
				cost = max(16, (int)ceil(pre_cost[i] * 0.8));
			} else {
				cost = 64;
			}
			if (step < cost) {
				break;
			}

			if (drop_num[i] < MAX_DROP_NUM) {
				float score = Get_Pos_Score(i, disk[i].head, timestamp);
				if (score <= DROP_SCORE) {
					move += 'p';
					pre_move[i] = 'p';
					pre_cost[i] = 0;
					step--;
					disk[i].head = (disk[i].head + 1) % V;
					
					if (query[obj_id].size()) {
						drop_num[i]++;
						drop++;
					}
					continue;
				}
			}
			
			bool is_hit = false;
			for (auto it = query[obj_id].begin(); it != query[obj_id].end(); ) {
				int qry = *it;
				auto prev = it;
				it++;
				assert(!requests[qry].is_done());
				// if (requests[qry].is_done()) continue; // 798254419 / 818602371 = 97.5% 的 is_done()
				bool has = requests[qry].has_part(obj_part);
				if (has) continue;
				is_hit = true;
				requests[qry].set(obj_part);
				if (requests[qry].is_done()) {
					query[obj_id].erase(prev);
					finish_qid.emplace_back(qry);
				}
			}
			if (is_hit) {
				move += 'r';
				step -= cost;
			} else {
				move += 'p';
				step--;
			}
			pre_cost[i] = cost;
			pre_move[i] = move.back();
			disk[i].head = (disk[i].head + 1) % V;
		}
		move += '#';
		moves[i] = move;
	}
	for (int i = 0; i < N; i++) {
		cout << moves[i] << "\n";
	}
	cout << finish_qid.size() << "\n";
	for (auto finish : finish_qid) {
		cout << finish << "\n";
	}
	cout.flush();
}


void Solve() {
	Delete_Action();
	Write_Action();
	Read_Action();
	Move();
}



 








void TimeStamp() {
	string pattern;
	int timeStamp;
	cin >> pattern >> timeStamp;
	cout << pattern << " " << timeStamp << "\n";
	cout.flush();
}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);

	cin >> T >> M >> N >> V >> G;
	hash_init();
	total_init();
	
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
		timestamp = i;
		TimeStamp();
		Solve();
	}
	return 0;
}
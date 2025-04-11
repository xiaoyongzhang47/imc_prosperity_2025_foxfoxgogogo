#include<iostream>
#include<vector>

using namespace std;

vector<vector<float>> graph = vector<vector<float>>({{1, 1.45, 0.52, 0.72},
                                                {0.7, 1, 0.31, 0.48},
                                                {1.95, 3.1, 1, 1.49},
                                                {1.34, 1.98, 0.64, 1}});

vector<int> max_path;
float max_value = 0;

void dfs(vector<vector<float>> &graph, vector<int> &path, int current_node, int steps_left, float current_value){
    if (steps_left == 1){
        current_value *= graph[current_node][3];
        if (current_value > max_value){
            max_value = current_value;
            max_path = path;
        }
    }
    else{
        for (int i = 0; i < 4; i++){
            path.push_back(i);
            dfs(graph, path, i, steps_left - 1, current_value*graph[current_node][i]);
            path.pop_back();
        }
    }
}

int main() {
    vector<int> path;
    dfs(graph, path, 3, 5, 1.0);
    cout << max_value << endl;
    for (auto i : max_path){
        cout << i;
    }
    return 0;
}
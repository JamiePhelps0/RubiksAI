#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <fstream>
#include <vector>
#include <cppflow/cppflow.h>
#include <chrono>
#include <thread>
#include <filesystem>

using namespace std;

class RubiksCube {
    public:
        map<string, int> edge_map = {{"43", 0}, {"10", 1}, {"30", 2}, {"20", 3}, {"41", 4}, {"15", 5}, {"35", 6}, {"25", 7}, {"32", 8}, {"12", 9}, {"54", 10}, {"04", 11}, {"52", 12}, {"21", 13}, {"34", 14}, {"51", 15}, {"03", 16}, {"40", 17}, {"01", 18}, {"53", 19}, {"02", 20}, {"45", 21}, {"14", 22}, {"23", 23}};
        map<string, int> corner_map =  {{"104", 0}, {"023", 1}, {"325", 2}, {"435", 3}, {"302", 4}, {"041", 5}, {"201", 6}, {"120", 7}, {"253", 8}, {"514", 9}, {"340", 10}, {"145", 11}, {"521", 12}, {"034", 13}, {"410", 14}, {"215", 15}, {"403", 16}, {"230", 17}, {"532", 18}, {"152", 19}, {"451", 20}, {"354", 21}, {"543", 22}, {"012", 23}, {"320", 0}, {"523", 1}, {"102", 2}, {"304", 3}, {"345", 4}, {"014", 5}, {"541", 6}, {"352", 7}, {"453", 8}, {"154", 9}, {"401", 10}, {"043", 11}, {"430", 12}, {"235", 13}, {"512", 14}, {"415", 15}, {"021", 16}, {"032", 17}, {"251", 18}, {"140", 19}, {"534", 20}, {"210", 21}, {"203", 22}, {"125", 23}};

        class Cubelet {
            private:
                int frontBack, leftRight, topBottom;
                class Edge {
                    private:
                        int topBottom, leftRight, frontBack;

                    public:
                        Edge(int tb, int lr, int fb)
                        {
                            topBottom = tb;
                            leftRight = lr;
                            frontBack = fb;
                        }
                };
                class Center {
                    private:
                        int colour;

                    public:
                        Center(int c) {
                            colour = c;
                        }
                };
                class Corner {
                    private:
                        int topBottom;
                        int leftRight;
                        int frontBack;

                    public:
                        Corner(int tb, int lr, int fb) {
                            topBottom = tb;
                            leftRight = lr;
                            frontBack = fb;
                        }
        };

            public:
                string type;
                Cubelet(string t, int tb, int lr, int fb) {
                    type = t;
                    topBottom = tb;
                    leftRight = lr;
                    frontBack = fb;
                    if (type == "corner") {
                        Corner cubelet(tb, lr, fb);
                    }
                    else if (type == "edge") {
                        Edge cubelet(tb, lr, fb);
                    }
                    else if (type == "center") {
                        Center cubelet(tb);
                    }
                }

                void printCubelet() { cout << type << topBottom << leftRight << frontBack << endl; }

                void rotate(int a) {
                    if (type == "center") {
                        return;
                    }
                    int temp;
                    if (a == 2 || a == 4) {
                        temp = leftRight;
                        leftRight = topBottom;
                        topBottom = temp;
                    }
                    else if (a == 1 || a == 3) {
                        temp = topBottom;
                        topBottom = frontBack;
                        frontBack = temp;
                    }
                    else if (a == 0 || a == 5) {
                        temp = leftRight;
                        leftRight = frontBack;
                        frontBack = temp;
                    }
                }

                bool operator == (Cubelet const other) {
                    if (type == other.type && leftRight == other.leftRight && topBottom == other.topBottom && frontBack == other.frontBack) { return true;}
                    return false;
                }

                string getID() {
                    if (type == "center") {
                        return "0";
                    }
                    else if (type == "edge") {
                        string out = "";
                        if (!(leftRight == -1)) {
                            out += to_string(leftRight);
                        }
                        if (!(frontBack == -1)) {
                            out += to_string(frontBack);
                        }
                        if (!(topBottom == -1)) {
                            out += to_string(topBottom);
                        }
                        return out;
                    }
                    return to_string(leftRight) + to_string(frontBack) + to_string(topBottom);
                }

            };

        Cubelet cubeArray[3][9] =
        {
        {Cubelet("corner", 0, 1, 4), Cubelet("edge", 0, -1, 4)   , Cubelet("corner", 0, 3, 4),
         Cubelet("edge", 0, 1, -1) , Cubelet("center", 0, -1, -1), Cubelet("edge", 0, 3, -1) ,
         Cubelet("corner", 0, 1, 2), Cubelet("edge", 0, -1, 2)   , Cubelet("corner", 0, 3, 2)},

        {Cubelet("edge", -1, 1, 4)   , Cubelet("center", 4, -1, -1) , Cubelet("edge", -1, 3, 4)   ,
         Cubelet("center", 1, -1, -1), Cubelet("center", -1, -1, -1), Cubelet("center", 3, -1, -1),
         Cubelet("edge", -1, 1, 2)   , Cubelet("center", 2, -1, -1)    , Cubelet("edge", -1, 3, 2)  },

        {Cubelet("corner", 5, 1, 4), Cubelet("edge", 5, -1, 4)   , Cubelet("corner", 5, 3, 4),
         Cubelet("edge", 5, 1, -1) , Cubelet("center", 5, -1, -1), Cubelet("edge", 5, 3, -1) ,
         Cubelet("corner", 5, 1, 2), Cubelet("edge", 5, -1, 2)   , Cubelet("corner", 5, 3, 2)}
        };
        void printCube() {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 9; j++) {
                    cubeArray[i][j].printCubelet();
                }
            }
         }

        void takeAction(int i_a) {
            bool cwise;
            int a;
            switch(i_a) {
                case 0:
                    a = 2;
                    cwise = true;
                    break;

                case 1:
                    a = 2;
                    cwise = false;
                    break;

                case 2:
                    a = 0;
                    cwise = true;
                    break;

                case 3:
                    a = 0;
                    cwise = false;
                    break;

                case 4:
                    a = 1;
                    cwise = true;
                    break;

                case 5:
                    a = 1;
                    cwise = false;
                    break;

                case 6:
                    a = 3;
                    cwise = true;
                    break;

                case 7:
                    a = 3;
                    cwise = false;
                    break;

                case 8:
                    a = 5;
                    cwise = true;
                    break;

                case 9:
                    a = 5;
                    cwise = false;
                    break;

                case 10:
                    a = 4;
                    cwise = true;
                    break;

                case 11:
                    a = 4;
                    cwise = false;
                    break;
            }

            Cubelet* arr[9] =
                {&cubeArray[0][0], &cubeArray[0][1], &cubeArray[0][2],
                 &cubeArray[0][3], &cubeArray[0][4], &cubeArray[0][5],
                 &cubeArray[0][6], &cubeArray[0][7], &cubeArray[0][8]};
            if (a == 1) {
                arr[0] = &cubeArray[0][0];
                arr[1] = &cubeArray[0][3];
                arr[2] = &cubeArray[0][6];
                arr[3] = &cubeArray[1][0];
                arr[4] = &cubeArray[1][3];
                arr[5] = &cubeArray[1][6];
                arr[6] = &cubeArray[2][0];
                arr[7] = &cubeArray[2][3];
                arr[8] = &cubeArray[2][6];
                }
            else if (a == 2) {
                arr[0] = &cubeArray[0][6];
                arr[1] = &cubeArray[0][7];
                arr[2] = &cubeArray[0][8];
                arr[3] = &cubeArray[1][6];
                arr[4] = &cubeArray[1][7];
                arr[5] = &cubeArray[1][8];
                arr[6] = &cubeArray[2][6];
                arr[7] = &cubeArray[2][7];
                arr[8] = &cubeArray[2][8];
            }
            else if (a == 3) {
                arr[0] = &cubeArray[0][8];
                arr[1] = &cubeArray[0][5];
                arr[2] = &cubeArray[0][2];
                arr[3] = &cubeArray[1][8];
                arr[4] = &cubeArray[1][5];
                arr[5] = &cubeArray[1][2];
                arr[6] = &cubeArray[2][8];
                arr[7] = &cubeArray[2][5];
                arr[8] = &cubeArray[2][2];
            }
            else if (a == 4) {
                arr[0] = &cubeArray[0][2];
                arr[1] = &cubeArray[0][1];
                arr[2] = &cubeArray[0][0];
                arr[3] = &cubeArray[1][2];
                arr[4] = &cubeArray[1][1];
                arr[5] = &cubeArray[1][0];
                arr[6] = &cubeArray[2][2];
                arr[7] = &cubeArray[2][1];
                arr[8] = &cubeArray[2][0];
            }
            else if (a == 5) {
                arr[0] = &cubeArray[2][6];
                arr[1] = &cubeArray[2][7];
                arr[2] = &cubeArray[2][8];
                arr[3] = &cubeArray[2][3];
                arr[4] = &cubeArray[2][4];
                arr[5] = &cubeArray[2][5];
                arr[6] = &cubeArray[2][0];
                arr[7] = &cubeArray[2][1];
                arr[8] = &cubeArray[2][2];
            }



            Cubelet face_copy[9] = {Cubelet("center", -1, -1, -1), Cubelet("center", -1, -1, -1), Cubelet("center", -1, -1, -1), Cubelet("center", -1, -1, -1), Cubelet("center", -1, -1, -1), Cubelet("center", -1, -1, -1), Cubelet("center", -1, -1, -1), Cubelet("center", -1, -1, -1), Cubelet("center", -1, -1, -1)};
            for (int i = 0; i < 9; i++) {
                face_copy[i] = **(arr + i);
            }

            if (cwise) {
                int to_idx;
                for (int i = 0; i < 9; i++) {
                    switch (i) {
                        case 0:
                            to_idx = 6;
                            break;
                        case 1:
                            to_idx = 3;
                            break;
                        case 2:
                            to_idx = 0;
                            break;
                        case 3:
                            to_idx = 7;
                            break;
                        case 4:
                            to_idx = 4;
                            break;
                        case 5:
                            to_idx = 1;
                            break;
                        case 6:
                            to_idx = 8;
                            break;
                        case 7:
                            to_idx = 5;
                            break;
                        case 8:
                            to_idx = 2;
                            break;
                    }

                    face_copy[to_idx].rotate(a);
                    **(arr + i) = face_copy[to_idx];
                }
            }
            else {
                int to_idx;
                for (int i = 0; i < 9; i++) {
                    switch (i) {
                        case 0:
                            to_idx = 2;
                            break;
                        case 1:
                            to_idx = 5;
                            break;
                        case 2:
                            to_idx = 8;
                            break;
                        case 3:
                            to_idx = 1;
                            break;
                        case 4:
                            to_idx = 4;
                            break;
                        case 5:
                            to_idx = 7;
                            break;
                        case 6:
                            to_idx = 0;
                            break;
                        case 7:
                            to_idx = 3;
                            break;
                        case 8:
                            to_idx = 6;
                            break;
                    }

                    face_copy[to_idx].rotate(a);
                    **(arr + i) = face_copy[to_idx];
                }

            }
        }

        bool operator == (RubiksCube const &other) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 9; j++) {
                    if (!(cubeArray[i][j] == other.cubeArray[i][j])) {
                        return false;
                    }
                }
            }
            return true;
        }

        int * getState() {
            static int state[27];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 9; j++) {
                    string id = cubeArray[i][j].getID();
                    if (id.length() == 2) {
                        state[i * 9 + j] = edge_map[id];
                    }
                    else if (id.length() == 3) {
                        state[i * 9 + j] = corner_map[id];
                    }
                    else {
                        state[i * 9 + j] = -1;
                    }

                }
            }
            static int newState[20];
            int idx = 0;
            for (int i = 0; i < 27; i++) {
                if (!(state[i] == -1)) {
                    newState[idx] = state[i];
                    idx++;
                }
            }
            return newState;
        }

        bool isSolved() {
        RubiksCube other;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 9; j++) {
                    if (!(cubeArray[i][j] == other.cubeArray[i][j])) {
                        return false;
                    }
                }
            }
            return true;
        }

        RubiksCube copy() {
            RubiksCube nextGame;

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 9; j++) {
                    nextGame.cubeArray[i][j] = cubeArray[i][j];
                }
            }
            return nextGame;
        }

        void Scramble(int depth) {
            if (depth == 0) {return;}
            srand(time(nullptr));
            int action = 0;

            for (int i = 0; i < depth; i++) {
                action = rand() % 12;
                takeAction(action);
            }
            if (isSolved()) {
                action = rand() % 12;
                takeAction(action);
            }
        }

};

class Node {
    public:
        Node *children[12], *parent;
        int action_counts[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int action, ch_solved = -1;
        float q_values[12];
        RubiksCube game;
        bool start;
        float c;

        Node(Node *p, RubiksCube g, int a, float v[12], float s, bool st) {
            start = st;
            parent = p;
            c = s;
            game = g;
            action = a;
            if (g.isSolved()) {
                (*parent).ch_solved = action;
            }
            for (int i = 0; i < 12; i++) {
                q_values[i] = *(v + i);
            }
        }

        ~Node() {
            for (int i = 0; i < 12; i++) {
                if (action_counts[i] > 0) {
                    delete children[i];
                }
            }
        }

        int select() {
            float best = -999.0;
            int best_a = -1;
            int action_sum = 0;
            for (int i = 0; i < 12; i++) {
                action_sum += action_counts[i];
            }
            float sqrt_action_sum = sqrt(action_sum);
            float ucb;
            for (int i = 0; i < 12; i++) {
                if (i == ch_solved) {
                    continue;
                }
                ucb = q_values[i] + c * sqrt_action_sum / (1.0 + action_counts[i]);
                if (ucb > best) {
                    best = ucb;
                    best_a = i;
                }
            }
            return best_a;
        }

        RubiksCube getNextGame(int a) {
            RubiksCube nextGame = game.copy();
            nextGame.takeAction(a);
            return nextGame;
        }

        void createChild(RubiksCube g, int a, float v[12], float s) {
            Node* newNode = new Node(this, g, a, v, s, false);
            children[a] = newNode;
        }

};

Node* mcts(RubiksCube env, int simulations, cppflow::model net) {
    int* state = env.getState();
    vector<float> input(20 * 24, 0.0);
    int onehot_idx = 0;
    for (int i = 0; i < 20; i++) {
        onehot_idx = *(state + i);
        input[i * 24 + onehot_idx] = 1.0f;
    }
    auto input_t = cppflow::tensor(input, {20, 24});

    auto output = net(input_t);

    vector<float> val_vec = output.get_data<float>();
    float val[12];
    for (int i = 0; i < 12; i++) {
        val[i] = val_vec[i];
    }
    Node* startNode = new Node(NULL, env.copy(), -1, val, 2.5, true);
    int a = 0;
    for (int i = 0; i < simulations; i++) {
        a = (*startNode).select();
        Node* currentNode = startNode;
        while ((*currentNode).action_counts[a] > 0) {
            (*currentNode).action_counts[a] += 1;
            currentNode = (*currentNode).children[a];
            a = (*currentNode).select();
        }
        RubiksCube newGame = (*currentNode).getNextGame(a);

        if (!newGame.isSolved()) {
            state = newGame.getState();
            vector<float> input(20 * 24, 0.0);
            for (int i = 0; i < 20; i++) {
                onehot_idx = *(state + i);
                input[i * 24 + onehot_idx] = 1.0f;
            }
            input_t = cppflow::tensor(input, {20, 24});

            output = net(input_t);
            val_vec = output.get_data<float>();
            for (int i = 0; i < 12; i++) {
                val[i] = val_vec[i];
            }
        }
        else {
            for (int i = 0; i < 12; i++) {
                val[i] = -1.0;
            }
        }

        (*currentNode).createChild(newGame, a, val, 2.5);
        (*currentNode).action_counts[a] += 1;
        currentNode = (*currentNode).children[a];
        while (!((*currentNode).parent == NULL)) {
            float q_max = -999.0;
            for (int i = 0; i < 12; i++) {
                if ((*currentNode).q_values[i] > q_max) {
                    q_max = (*currentNode).q_values[i];
                }
            }
            (*(*currentNode).parent).q_values[(*currentNode).action] = -1 + q_max;
            currentNode = (*currentNode).parent;
        }
    }
    return startNode;
}

inline bool exists (const string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

void generateDataset(int datapoints, int offset, int ScramNum, int fileNum, int fidx) {
    filesystem::create_directory("/home/jamie/IdeaProjects/RubiksNew/DQN/DataFiles/Dataset" + to_string(fidx));
    ofstream dataout;
    string directory = "/home/jamie/IdeaProjects/RubiksNew/DQN/DataFiles/Dataset" + to_string(fidx) + "/data" + to_string(fileNum) + ".txt";
    dataout.open(directory, ios::app);

    RubiksCube cube;
    cube.Scramble(offset);

    srand(time(nullptr));

    int action;
    int* state = cube.getState();
    int* state_old = state;

    for (int i = 1; i <= datapoints; i++) {
        action = rand() % 12;
        state_old = state;
        cube.takeAction(action);
        state = cube.getState();
        if (action % 2 == 0) {action++;}
        else {action--;}
        dataout << ':';
        for (int j = 0; j < 20; j++) {
            dataout << *(state + j) << ';';
        }
        dataout << ':' << action << ':';
        for (int j = 0; j < 20; j++) {
            dataout << *(state_old + j) << ';';
        }
        dataout << '\n';
        if (i % (ScramNum - offset) == 0) {
            cube = RubiksCube();
            cube.Scramble(offset);
        }
    }
    dataout.close();
}


void exhaustDataset(int fileNum, int fidx) {
//    filesystem::create_directory("/home/jamie/IdeaProjects/RubiksNew/DQN/DataFiles/Dataset" + to_string(fidx));
//    ofstream dataout;
//    string directory = "/home/jamie/IdeaProjects/RubiksNew/AlphaZero/DataFiles/Dataset" + to_string(fidx) + "/data" + to_string(fileNum) + ".txt";
//    dataout.open(directory, ios::app);

    RubiksCube cube;

    vector<RubiksCube> old = {cube};
    vector<int> old_a;
    int skip = 0;
    int n_action;

    for (int j = 0; j < 2; j++) {
        vector<RubiksCube> explore(12 * pow(11, j));
        vector<int> explore_a(12 * pow(11, j), -1);
        for (int i = 0; i < old.size(); i++) {
            if (j == 0) {
                for (int k = 0; k < 12; k++) {
                    explore[k] = cube.copy();
                    explore[k].takeAction(k);
                    explore_a[k] = k;
                }
            }
            else {
                n_action = 0;
                skip = 0;
                if (old_a[i] % 2 == 0) {n_action++;}
                else {n_action--;}
                for (int k = 0; k < 12; k++) {
                    if (!(k == n_action)) {
                        explore[i * 11 + k - skip] = cube.copy();
                        explore[i * 11 + k - skip].takeAction(k);
                        explore_a[i * 11 +  k - skip] = k;
                    }
                    else {
                        skip++;
                    }
                }
            }
            cout << explore_a[explore_a.size() - 1];
        }
//        for (int i = 0; i < explore.size(); i++) {
//
//        }
    }

}


void test(int depth, cppflow::model net, int tests, int simulations) {
    int solved = 0;
    RubiksCube cube;
    srand(time(nullptr));
    int action;
    for (int i = 0; i < tests; i++) {
        cout << i << endl;
        cube = RubiksCube();
        for (int j = 0; j < depth; j++) {
            action = rand() % 12;
            cube.takeAction(action);
            }
        if (cube.isSolved()) {
            action = rand() % 12;
            cube.takeAction(action);
        }
        int counter = 0;
        while (counter < 2 * depth) {

            Node* search = mcts(cube, simulations, net);
            int act = -1;
            float best = -999.0;
            for (int j = 0; j < 12; j++) {
                if ((*search).q_values[j] > best) {
                    act = j;
                    best = (*search).q_values[j];
                }
            }
            delete search;
            cube.takeAction(act);
            counter++;
            if (cube.isSolved()) {
                solved++;
                break;
            }
        }
    }
    float percentage = (float)solved / (float)tests;
    cout << "% Success at depth " << depth << " == " << percentage << endl;
}


float* softmax(float array[12]) {
    static float probs[12];
    float expArray[12];
    for (int i = 0; i < 12; i++) {
        expArray[i] = exp(array[i]);
    }
    float expSum = 0;
    for (int i = 0; i < 12; i++) {
        expSum += expArray[i];
    }
    for (int i = 0; i < 12; i++) {
        probs[i] = expArray[i] / expSum;
    }
    return probs;
}


int main() {
//    std::vector<uint8_t> config{0x32,0x9,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xc9,0x3f};
//    TFE_ContextOptions* options = TFE_NewContextOptions();
//    TFE_ContextOptionsSetConfig(options, config.data(), config.size(), cppflow::context::get_status());
//    cppflow::get_global_context() = cppflow::context(options);

    while(1) {

    thread threads[8];

    for (int i = 0; i < 8; i++) {
        threads[i] = thread(generateDataset, 100000, 1, 20, i, 0);
        threads[i].join();
    }
    ofstream file;
    file.open("l.lock");
    file.close();
    cout << "done" << endl;
    while (exists("l.lock")) {this_thread::sleep_for(chrono::milliseconds(100));}

    }
    return 0;
}
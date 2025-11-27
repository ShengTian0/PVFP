import numpy as np
import copy
import queue
import operator

inf = 999999


# 邻接矩阵转为邻接表
def matrix_to_table(E):
    adj = {}
    for i in range(len(E)):
        i_neighbor = []
        for j in range(len(E[i])):
            if E[i][j] != 0 and E[i][j] != 999:
                i_neighbor.append(j)
        if len(i_neighbor) != 0:
            adj[i] = i_neighbor
    return adj


# 带权图邻接矩阵转为邻接表
def wg_matrix_to_table(E):
    adj = {}
    for i in range(len(E)):
        i_neighbors = []
        i_neighbor = {}
        for j in range(len(E)):
            if E[i][j] != 0 and E[i][j] != inf:
                i_neighbor[j] = E[i][j]
        i_neighbors.append(i_neighbor)
        if len(i_neighbors) != 0:
            adj[i] = i_neighbors
    return adj


class MUGraph():
    def __init__(self, V, E):
        self.V = V
        self.E = E
        self.vnum = len(V)
        enum = 0
        for i in range(len(E)):
            for j in range(i, len(E)):
                if E[i][j] != 0 and E[i][j] != 999:
                    enum += 1
        self.enum = enum

    # floyd_warshall算法求各节点之间的最短路径,a为1返回距离矩阵，a为2返回parent矩阵
    # def floyd_warshall(self, a):
    def floyd_warshall(self, a=0):
        n = len(self.E)
        parent = np.zeros((len(self.E), len(self.E)))
        for i in range(len(self.E)):
            for j in range(len(self.E)):
                parent[i][j] = j
        dist = copy.deepcopy(self.E)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        parent[i][j] = parent[i][k]
        if a == 0:
            return dist, parent
        elif a == 1:
            return dist
        elif a == 2:
            return parent

    # 返回邻接结点
    def adj(self, s):
        if s > self.vnum:
            return
        neigh = []
        for i in range(len(self.V)):
            if self.E[s][i] == 1:
                neigh.append(i)
        return neigh

    # 求floyd最短路径
    def find_floyd_route(self, parent, a, b):
        route = []
        route.append(b)
        while a != b:
            b_pre = int(parent[b][a])
            b = b_pre
            route.append(b)
        route.reverse()
        return route

    # 广度优先搜索
    def BFS(self, s):
        route = []
        visit = np.zeros(8)
        q = queue.Queue()
        q.put(s)
        route.append(s)

        while q.empty() == False:
            u = q.get()
            visit[u] = 1
            neigh = self.adj(u)
            for v in neigh:
                if visit[v] == 0:
                    visit[v] = 1
                    route.append(v)
                    q.put(v)
        return route

    # 广度优先搜索最短路径
    def BFSMinroute(self, s):
        route = []
        visit = np.zeros(8)
        q = queue.Queue()
        q.put(s)
        route.append(s)

        while q.empty() == False:
            u = q.get()
            visit[u] = 1
            neigh = self.adj(u)
            for v in neigh:
                if visit[v] == 0:
                    visit[v] = 1
                    route.append(v)
                    q.put(v)
        return route

    # 广度优先生成树
    def BFS_Tree(self, s):
        T = np.zeros((8, 8))
        visit = np.zeros(8)
        q = queue.Queue()
        q.put(s)

        while q.empty() == False:
            u = q.get()
            u_pre = u
            visit[u] = 1
            neigh = self.adj(u)
            for v in neigh:
                if visit[v] == 0:
                    visit[v] = 1
                    T[u_pre][v] = T[v][u_pre] = 1
                    q.put(v)
        return T

    # 深度优先遍历访问结点(使用visit)
    def DFSv_visit(self, v, visit):
        visit[v] = 1
        route.append(v)
        neigh = self.adj(v)
        for u in neigh:
            if visit[u] != 1:
                self.DFS_visit(u, visit)

    # 深度优先遍历，v为起始节点（使用visit)
    def DFSv(self, v):
        global route
        route = []
        visit = np.zeros(8)
        self.DFS_visit(v, visit)
        for v in self.V:
            if visit[v] != 1:
                self.DFS_visit(v, visit)
        return route

    # 深度优先生成树访问结点
    def DFS_T_visit(self, v, visit):
        visit[v] = 1
        pre_v = v
        neigh = self.adj(v)
        for u in neigh:
            if visit[u] != 1:
                T[pre_v][u] = T[u][pre_v] = 1
                self.DFS_T_visit(u, visit)

    # 深度优先生成树,v为起始节点
    def DFS_T(self, v):
        global T
        T = np.zeros((8, 8))
        visit = np.zeros(8)
        self.DFS_T_visit(v, visit)
        for v in self.V:
            if visit[v] != 1:
                self.DFS_T_visit(v, visit)
        return T


# 无向带权图类
class MUWGraph(MUGraph):
    def __init__(self, V, E):
        MUGraph.__init__(self, V, E)

    # 求点的临接顶点和边
    def adj(self, v):
        if v > self.vnum:
            return
        nextneighs = []
        i = 0
        for value in self.E[v]:
            if (value != 0 and value != 999):
                key = str(i)
                nextneigh = {'node': i, 'key': value}
                nextneighs.append(nextneigh)
            i += 1
        return nextneighs

    # kruskal求最小生成树算法
    def kruskal(self):
        T = np.zeros((9, 9))
        T_set = []
        for v in range(len(self.V)):  # 将每个结点初始化为各个子树集合
            a = [v]
            a = set(a)
            T_set.append(a)
        w_sorts = {}
        for i in range(len(self.E)):  # 以权值为关键字、对应边集合为值构造字典
            for j in range(i):
                if self.E[i][j] != 0:
                    w = self.E[i][j]
                    w_e = {i: j}
                    if w in w_sorts:
                        w_sorts[w].append(w_e)
                    else:
                        w_sort = []
                        w_sort.append(w_e)
                        w_sorts[w] = w_sort
        weight = sorted(w_sorts.keys())  # 权值排序
        for i in range(len(weight)):  # 遍历对应权值的边进行最小生成树合并
            while len(w_sorts[weight[i]]) != 0:  # 当权值对应的边集合部位不为0时，不断pop出边进行合并
                e = w_sorts[weight[i]].pop()
                v1 = list(e.keys())[0]
                v2 = list(e.values())[0]
                for Ts in T_set:  # 寻找结点所属的树集合
                    if v1 in Ts:
                        v_set1 = Ts
                    if v2 in Ts:
                        v_set2 = Ts
                if v_set1 != v_set2:  # 若结点不属于同一集合则可以通过该边进行合并
                    T[v1][v2] = T[v2][v1] = weight[i]
                    v_set12 = v_set1 | v_set2
                    T_set.append(v_set12)
                    T_set.remove(v_set1)
                    T_set.remove(v_set2)
        print('最小生成树的权值和为：' + str(sum(sum(T)) / 2))
        return T

    # prim算法
    def prim(self, v):
        nodes = []
        inf = 9999999
        Q_flag = []  # 标记已选中的节点，避免形成回路
        for value in range(len(self.V)):
            Q_flag.append(1)
            if value != v:
                node = {'name': value, 'key': inf, 'parent': None}
                nodes.append(node)
        nodes[0] = {'name': v, 'key': 0, 'parent': None}

        A = []  # 最小生成树的边按顺序放进A内
        Q = nodes
        node = Q[0]
        Q_flag[v] = 0
        Q_flag_result = []  # 确定所有结点都遍历到
        A.append(node)
        T = np.zeros((self.vnum, self.vnum))  # 最小生成树
        for i in range(len(Q_flag)):
            Q_flag_result.append(0)
        while Q_flag != Q_flag_result:  # 遍历生成最小生成树
            min_neigh = {'node': 8, 'key': inf}
            for value in A:  # 在A中选出结点并找到其邻接节点
                v = value
                neighs = self.adj(v['name'])
                neighss = neighs
                for i in range(len(neighss)):  # 将已在A中的邻接点删除，避免形成回路
                    if Q_flag[neighs[i]['node']] == 0:
                        neighs[i] = {'node': 10, 'key': inf}

                for neigh in neighs:  # 在Q中找到与A内结点相邻的边权值最小的结点
                    if neigh['key'] < min_neigh['key']:
                        min_neigh = neigh
                        min_v_parent = value

            T[min_neigh['node']][min_v_parent['name']] = T[min_v_parent['name']][min_neigh['node']] = min_neigh['key']
            Q_flag[min_neigh['node']] = 0
            node = {'name': min_neigh['node'], 'key': min_neigh['key'], 'parent': min_v_parent['name']}
            A.append(node)

        print('最小生成树的权值和为：' + str(sum(sum(T)) / 2))
        return (T)


#
# # 有向带权图类
# class MDWGraph(MUWGraph):
#     # 初始化数据
#     def __init__(self, V, E):
#         MUWGraph.__init__(self, V, E)
#         self.time = 0
#
#     # 返回邻接结点
#     def adj(self, s):
#         if s > self.vnum:
#             return
#         neigh = []
#         for i in range(len(self.V)):
#             if self.E[s][i] != 0 and self.E[s][i] != inf:
#                 neigh.append(i)
#         return neigh
#
#     # Bellman_Ford单源最短路径算法
#     def Bellman_Ford(self, s):
#         # 松弛操作
#         relaxation = []
#         for v in self.V:
#             relaxation.append({'dist': inf, 'pre': None})
#         relaxation[s]['dist'] = 0
#         for k in range(1, len(self.V)):
#             for i in range(len(self.V)):
#                 for j in range(len(self.V)):
#                     if self.E[i][j] != inf:
#                         if relaxation[j]['dist'] > relaxation[i]['dist'] + self.E[i][j]:
#                             relaxation[j]['dist'] = relaxation[i]['dist'] + self.E[i][j]
#                             relaxation[j]['pre'] = i
#         for i in range(len(self.V)):
#             for j in range(len(self.V)):
#                 if self.E[i][j] != inf:
#                     if relaxation[j]['dist'] > relaxation[i]['dist'] + self.E[i][j]:
#                         return False
#         return relaxation
#
#     def dijkstra(self, s):
#         # 松弛操作
#         relaxation = []
#         for v in self.V:
#             relaxation.append({'dist': inf, 'pre': None})
#         relaxation[s]['dist'] = 0
#
#         S = set()
#         Q = set(self.V)
#         while len(Q) != 0:
#             QS = Q - S
#             u = QS.pop()
#             S.append(u)
#             adju = self.adj(u)
#             for v in adju:
#                 if relaxation[v]['dist'] > relaxation[u]['dist'] + self.E[u][v]:
#                     relaxation[v]['dist'] = relaxation[u]['dist'] + self.E[u][v]
#                     relaxation[v]['pre'] = u
#         return relaxation
#
#         # 有向无权图类


class MDWGraph(MUGraph):
    # 初始化数据
    def __init__(self, V, E):
        MUGraph.__init__(self, V, E)
        self.time = 0

    # 求点的临接顶点和边
    def adj(self, v):
        if v > self.vnum:
            return
        nextneighs = []
        i = 0
        for i in range(len(self.E[v])):
            if (i != v and self.E[v][i] != inf):
                key = str(i)
                nextneigh = {'node': i, 'key': self.E[v][i]}
                nextneighs.append(nextneigh)
            i += 1
        return nextneighs

    # 深度优先遍历，v为起始节点（算法导论方案）
    def DFS(self):
        vertex = []
        for i in range(len(self.V)):
            u = {'v': i, 'color': 'WHITE', 'pre': None}
            vertex.append(u)
        self.time = 0
        for v in range(len(vertex)):
            if vertex[v]['color'] == 'WHITE':
                self.DFS_visit(v, vertex)
        return vertex

    # 深度优先遍历访问结点（算法导论方案）
    def DFS_visit(self, u, vertex):
        self.time = self.time + 1
        vertex[u]['d'] = self.time
        vertex[u]['color'] = 'GRAY'
        neigh = self.adj(u)
        for v in neigh:
            if vertex[v]['color'] == 'WHITE':
                vertex[v]['pre'] = u
                self.DFS_visit(v, vertex)
        vertex[u]['color'] = 'BLACK'
        self.time = self.time + 1
        vertex[u]['finish'] = self.time

    # 拓扑排序
    def topologucal_sort(self):
        vertex = self.DFS()
        vertex_sort = sorted(vertex, key=lambda keys: keys['finish'], reverse=True)
        print(vertex_sort)
        list = []
        for v in vertex_sort:
            list.append(v['v'])
        return list

    # 强连通分量专用DFS
    def scc_DFS(self, v_DFS):
        v_DFS_sort = sorted(v_DFS, key=lambda keys: keys['finish'], reverse=True)
        v_DFS_sort_vertex = []
        for v in v_DFS_sort:
            v_DFS_sort_vertex.append(v['v'])
        vertex = []
        for i in range(len(self.V)):
            u = {'v': i, 'color': 'WHITE', 'pre': None}
            vertex.append(u)
        self.time = 0
        for v in v_DFS_sort_vertex:
            if vertex[v]['color'] == 'WHITE':
                self.DFS_visit(v, vertex)
        return vertex

    # 强连通分量
    def strongly_connected_components(self):
        vertex = self.DFS()
        E_T = np.zeros((self.vnum, self.vnum))
        for i in range(len(self.V)):
            for j in range(len(self.V)):
                E_T[i][j] = self.E[j][i]
        E_1 = self.E[:][:]
        self.E = E_T[:][:]
        vertex2 = self.scc_DFS(vertex)
        self.E = E_1[:][:]
        return vertex2

    # dijkstra求最短路径
    def dijkstra(self, s):
        # 松弛操作
        relaxation = []
        for v in self.V:
            relaxation.append({'dist': inf, 'pre': None})
        relaxation[s]['dist'] = 0

        S = set()
        Q = set(self.V)
        while len(Q) != 0:
            minud = inf
            for i in range(len(relaxation)):
                if i in Q:
                    if relaxation[i]['dist'] <= minud:
                        minud = relaxation[i]['dist']
                        minu = i

            u = minu
            Q.remove(u)
            S.add(u)
            adju = self.adj(u)
            for v in adju:
                if relaxation[v['node']]['dist'] > relaxation[u]['dist'] + self.E[u][v['node']]:
                    relaxation[v['node']]['dist'] = relaxation[u]['dist'] + self.E[u][v['node']]
                    relaxation[v['node']]['pre'] = u
        return relaxation


    # 回溯根据dijkstra算法的结果找出a到b的最短路径
    def dijkstra_route(self, relaxation, a, b):
        if relaxation[b]['dist'] >= inf:
            return 0
        route = []
        pre_v = b
        while True:
            route.insert(0, pre_v)
            if pre_v == a:  # 如果已经到达源头，则停止
                break
            pre_v = relaxation[pre_v]['pre']
            if pre_v is None:  # 如果前驱节点不存在，说明路径中断
                return 0  # 或者根据需要返回一个表示失败的值

        return route

    # dijkstra求最短路径
    def Modifydijkstra(self, s, constrain):
        constrain1 = constrain[:]
        constrain2 = constrain[:]
        # 松弛操作
        relaxation = []
        for v in self.V:
            relaxation.append({'dist': inf, 'pre': None})
        relaxation[s]['dist'] = 0

        S = set()
        Q = set(self.V)
        while len(Q) != 0:
            minud = inf
            for i in range(len(relaxation)):
                if i in Q:
                    if relaxation[i]['dist'] < minud:
                        minud = relaxation[i]['dist']
                        minu = i
            u = minu
            Q.remove(u)
            S.add(u)
            adju = self.adj(u)
            for v in adju:
                if relaxation[v['node']]['dist'] > relaxation[u]['dist'] + self.E[u][v['node']]:
                    relaxation[v['node']]['dist'] = relaxation[u]['dist'] + self.E[u][v['node']]
                    relaxation[v['node']]['pre'] = u
            # 求结点剩余容量
            if u <= 32 and int((u - 1) / 8) % 2 == 1:
                if int((u - 1) / 4) % 2 == 0:
                    constrain1[(u - 1) % 4] = constrain1[(u - 1) % 4] - self.E[u - 8][u]
                else:
                    constrain2[(u - 1) % 4] = constrain1[(u - 1) % 4] - self.E[u - 8][u]
            if u > 32 and int((u - 1) / 4) % 2 == 1:
                constrain1[(u - 1) % 4] = constrain1[(u - 1) % 4] - self.E[u - 4][u]
                constrain2[(u - 1) % 4] = constrain1[(u - 1) % 4] - self.E[u - 4][u]
            # 删除边
            if u <= 32:
                if constrain1[(u - 1) % 4] < self.E[u + 8][u + 16] or constrain2[(u - 1) % 4] < self.E[u + 8][u + 16]:
                    self.E[u + 8][u + 16] = inf
            elif u <= 40:
                if constrain1[(u - 1) % 4] < self.E[u + 4][u + 8] or constrain2[(u - 1) % 4] < self.E[u + 4][u + 8]:
                    self.E[u + 4][u + 8] = inf
            print(constrain1)
            print(constrain2)

        return relaxation

    def node_weight_steiner_tree_2(self, V_cost):
        # node wight steiner_tree算法
        # 求dist和parent矩阵
        dist = self.floyd_warshall(1)
        parent = self.floyd_warshall(2)
        # 初始的树集合
        T = []
        for i in range(len(V_cost)):
            if V_cost[i] == 0:
                t = [i]
                T.append(t)

        # 树的边矩阵
        tree_matrix = np.zeros((len(V_cost), len(V_cost)))

        # 商代价
        quotient_cost = []

        # 初始化树与距离字典
        dicts_T = []
        for tree in T:
            dict_T = {}
            dict_T['Tree'] = tree
            dicts_T.append(dict_T)

        while len(T) != 1:
            min_quotient_cost = 100
            for i in self.V:
                dicts_T = []
                for j in T:
                    min_d = 100  # 结点到树中结点的最短距离
                    flag = 0
                    for k in j:
                        d = dist[i][k]
                        if d < min_d:
                            min_d = d
                            t_node = k  # 最短距离对应的树结点
                            flag = 1
                    if flag == 0:
                        return 0, 0, 0

                    distance_to_tree = min_d
                    dict_T = {}
                    dict_T['Tree'] = j
                    dict_T['t_node'] = t_node
                    dict_T['distance'] = min_d
                    dicts_T.append(dict_T)
                dicts_T.sort(key=operator.itemgetter('distance'))
                quotient_cost_1 = (V_cost[i] + dicts_T[0]['distance'] + dicts_T[1]['distance']) / len(T)
                if quotient_cost_1 < min_quotient_cost:
                    min_quotient_cost = quotient_cost_1
                    v = i  # 待插入结点v
                    select_dicts = dicts_T
            # 合并树
            v1 = v
            new_tree = []
            if v not in select_dicts[0]['Tree']:
                while v1 != select_dicts[0]['t_node']:  # v连接第一棵树
                    new_tree.append(v1)
                    v_parent = v1
                    v1 = int(parent[v1][select_dicts[0]['t_node']])
                    tree_matrix[v_parent][v1] = 1
                    tree_matrix[v1][v_parent] = 1
                tree_matrix[v_parent][v1] = 1
                tree_matrix[v1][v_parent] = 1

                v2 = v
                while v2 != select_dicts[1]['t_node']:  # v连接第二棵树
                    new_tree.append(v2)
                    v_parent = v2
                    v2 = int(parent[v2][select_dicts[1]['t_node']])
                    tree_matrix[v_parent][v2] = 1
                    tree_matrix[v2][v_parent] = 1
                tree_matrix[v_parent][v2] = 1
                tree_matrix[v2][v_parent] = 1

                new_tree.extend(select_dicts[0]['Tree'])
                new_tree.extend(select_dicts[1]['Tree'])
                T = [new_tree]
                for i in range(2, len(select_dicts)):
                    T.append(select_dicts[i]['Tree'])
            else:
                v2 = v
                while v2 != select_dicts[1]['t_node']:  # v连接第二棵树
                    new_tree.append(v2)
                    v_parent = v2
                    v2 = int(parent[v2][select_dicts[1]['t_node']])
                    tree_matrix[v_parent][v2] = 1
                    tree_matrix[v2][v_parent] = 1
                tree_matrix[v_parent][v2] = 1
                tree_matrix[v2][v_parent] = 1

                new_tree.extend(select_dicts[0]['Tree'])
                new_tree.extend(select_dicts[1]['Tree'])
                new_tree = list(set(new_tree))
                T = [new_tree]
                for i in range(2, len(select_dicts)):
                    T.append(select_dicts[i]['Tree'])

        tree_table = matrix_to_table(tree_matrix)
        # 输出steiner树
        return T, tree_matrix, tree_table

    def kuo_steiner_tree(self, destination):
        """构造一个辅助完全图"""
        complete_V_num = len(destination)
        complete_V = np.zeros(complete_V_num)
        dist, parent = self.floyd_warshall()
        for i in range(complete_V_num):
            complete_V[i] = destination[i]
        # 初始化边
        complete_E = np.zeros((complete_V_num, complete_V_num))
        for i in range(complete_V_num):
            for j in range(complete_V_num):
                if i != j and complete_E[i][j] == 0:
                    complete_E[i][j] = complete_E[j][i] = inf
        # 构造边
        for i in range(complete_V_num):
            for j in range(i + 1, complete_V_num):
                ea = int(complete_V[i])
                eb = int(complete_V[j])
                complete_E[i][j] = complete_E[j][i] = dist[ea][eb]
        # print("complete_e")
        # print(complete_E)
        complete_Graph = MDWGraph(complete_V, complete_E)
        # 求完全辅助图的最小生成树
        MST_complete, s = complete_Graph.kruskal()
        # 将最小生成树的边转化为对应辅助图最短路径的所有边
        # print("MST_completr")
        # print(MST_complete)
        MST_auxiliary = np.zeros((self.vnum, self.vnum))  # 生成树MST对应实际辅助图的图H
        for i in range(self.vnum):
            for j in range(i + 1, self.vnum):
                if MST_auxiliary[i][j] == 0:
                    MST_auxiliary[i][j] = MST_auxiliary[j][i] = inf
        for i in range(complete_V_num):
            for j in range(complete_V_num):
                if MST_complete[i][j] != inf and i != j:
                    v_i = int(complete_V[i])
                    v_j = int(complete_V[j])
                    while v_j != v_i:
                        v_pre = int(parent[v_j][v_i])
                        MST_auxiliary[v_pre][v_j] = self.E[v_pre][v_j]
                        v_j = v_pre
                    MST_auxiliary[v_pre][v_j] = self.E[v_pre][v_j]
        # print("MST_auxiliary")
        # print(MST_auxiliary)
        # print(parent)
        MST_auxiliary_Graph = MDWGraph(self.V, MST_auxiliary)
        T, sum_T = MST_auxiliary_Graph.kruskal()

        return T, sum_T

    # kruskal最小生成树算法
    def kruskal(self):
        T = np.zeros((self.vnum, self.vnum))
        for i in range(self.vnum):
            for j in range(self.vnum):
                if i != j and T[i][j] == 0:
                    T[i][j] = T[j][i] = inf
        T_set = []
        for v in range(len(self.V)):  # 将每个结点初始化为各个子树集合
            a = [v]
            a = set(a)
            T_set.append(a)
        w_sorts = {}
        for i in range(len(self.E)):  # 以权值为关键字、对应边集合为值构造字典
            for j in range(i):
                if i != j and self.E[i][j] != inf:
                    w = self.E[i][j]
                    w_e = {i: j}
                    if w in w_sorts:
                        w_sorts[w].append(w_e)
                    else:
                        w_sort = []
                        w_sort.append(w_e)
                        w_sorts[w] = w_sort
        weight = sorted(w_sorts.keys())  # 权值排序
        for i in range(len(weight)):  # 遍历对应权值的边进行最小生成树合并
            while len(w_sorts[weight[i]]) != 0:  # 当权值对应的边集合部位不为0时，不断pop出边进行合并
                e = w_sorts[weight[i]].pop()
                v1 = list(e.keys())[0]
                v2 = list(e.values())[0]
                for Ts in T_set:  # 寻找结点所属的树集合
                    if v1 in Ts:
                        v_set1 = Ts
                    if v2 in Ts:
                        v_set2 = Ts
                if v_set1 != v_set2:  # 若结点不属于同一集合则可以通过该边进行合并
                    T[v1][v2] = T[v2][v1] = weight[i]
                    v_set12 = v_set1 | v_set2
                    T_set.append(v_set12)
                    T_set.remove(v_set1)
                    T_set.remove(v_set2)

        sum_T = 0
        for i in range(len(T)):
            for j in range(len(T)):
                if T[i][j] != inf:
                    sum_T += T[i][j]
        return T, sum_T / 2

# N = 6
# V = list(range(N))
# E = np.zeros((N, N))
# for i in range(len(E)):
#     for j in range(len(E)):
#         if i != j and E[i][j] == 0:
#             E[i][j] = E[j][i] = inf
# E[0][1] = E[1][0] = 0
# E[0][5] = E[5][0] = 1
# E[1][5] = E[5][1] = 6
# E[1][2] = E[2][1] = 0
# E[1][4] = E[4][1] = 7
# E[4][5] = E[5][4] = 3
# E[2][4] = E[4][2] = 3
# E[2][3] = E[3][2] = 0
# E[3][4] = E[4][3] = 1
#
# print(V)
# print(E)
#
# G = MDWGraph(V, E)
# V_cost = [0, 1, 0, 0, 1, 1]
# rexalation = G.dijkstra(0)
# print(rexalation)
# print(G.dijkstra_route(rexalation, 0, 5))
#
# Tnode, tree_matrix,tree_table = G.node_weight_steiner_tree_2(V_cost)
# print('输出steiner树所包含的结点：')
# print(Tnode)
# print(tree_matrix)
# print('输出steiner树的邻接表：')
# print(tree_table)
# print('哭唧唧地结束了')

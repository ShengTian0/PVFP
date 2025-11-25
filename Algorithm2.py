from MGraph import MDWGraph
import numpy as np
import random

from timeit import default_timer as timer

inf = 999999  # 边权无穷大


class Algorithm2():
    # 输入原始的图
    def input_initial_Graph(self):
        N = 11
        V = list(range(N))
        E = np.zeros((N, N))
        inf = 999999
        E[0][1] = E[1][0] = E[5][6] = E[6][5] = E[6][9] = E[9][6] = 1
        E[1][2] = E[2][1] = E[1][3] = E[3][1] = E[0][3] = E[3][0] = E[3][4] = E[4][3] = E[4][6] = E[6][4] = E[7][9] = \
            E[9][
                7] = 2
        E[2][3] = E[3][2] = E[2][4] = E[4][2] = E[4][7] = E[7][4] = E[5][8] = E[8][5] = 3
        E[7][10] = E[10][7] = E[2][5] = E[5][2] = 4
        for i in range(len(E)):
            for j in range(i + 1, len(E)):
                if E[i][j] == 0:
                    E[i][j] = E[j][i] = inf
        initial_Graph = MDWGraph(V, E)
        function_V = [1, 2, 3, 4]

        # 链路资源限制
        E_constrains = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                if i != j and E[i][j] != 0 and E[i][j] != inf:
                    E_constrains[i][j] = E_constrains[j][i] = 100

        # 计算资源限制
        function_V_constrains = np.zeros(len(function_V))
        for i in range(len(function_V_constrains)):
            function_V_constrains[i] = 500
        # 设置链路时延
        link_delay = np.zeros((N, N))
        for i in range(len(E)):
            for j in range(i + 1, len(E)):
                if link_delay[i][j] == 0:
                    link_delay[i][j] = link_delay[j][i] = inf
        # for i in range(N):
        #     for j in range(i + 1, N):
        #         if E[i][j] != inf:
        #             de = random.randint(1, 10)
        #             link_delay[i][j] = link_delay[j][i] = de
        #         else:
        #             link_delay[i][j] = link_delay[j][i] = inf

        link_delay[0][1] = link_delay[1][0] = link_delay[3][4] = link_delay[4][3] = link_delay[4][7] = link_delay[7][
            4] = link_delay[2][5] = link_delay[5][2] = link_delay[6][9] = link_delay[9][6] = 2
        link_delay[2][3] = link_delay[3][2] = link_delay[2][4] = link_delay[4][2] = link_delay[5][8] = link_delay[8][
            5] = link_delay[7][9] = link_delay[9][7] = 1
        link_delay[0][3] = link_delay[3][0] = link_delay[1][2] = link_delay[2][1] = link_delay[4][6] = link_delay[6][
            4] = link_delay[7][10] = link_delay[10][7] = 3
        link_delay[1][3] = link_delay[3][1] = link_delay[5][6] = link_delay[6][5] = 4

        return initial_Graph, function_V, E_constrains, function_V_constrains, link_delay

    # 处理边资源不达标的边
    def deal_initial_Graph(self, initial_Graph, E_constrains, bk):
        V = list(range(initial_Graph.vnum))
        E = np.zeros((initial_Graph.vnum, initial_Graph.vnum))
        # 录入V数据
        for i in range(len(V)):
            V[i] = initial_Graph.V[i]

        # 录入E数据
        for i in range(len(V)):
            for j in range(i + 1, len(V)):
                if initial_Graph.E[i][j] == inf:
                    E[i][j] = E[j][i] = initial_Graph.E[i][j]
                else:
                    if E_constrains[i][j] < bk * initial_Graph.E[i][j]:
                        E[i][j] = E[j][i] = inf
                    else:
                        E[i][j] = E[j][i] = initial_Graph.E[i][j]
        new_initial_Graph = MDWGraph(V, E)

        return new_initial_Graph

    # 构造功能子图,调用返回功能子图
    def get_fuction_Graph(self, initial_Graph, function_V):
        dist = initial_Graph.floyd_warshall(1)
        # 求功能子图的边和结点
        N = len(function_V)  # 功能子图的结点数
        E = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                E[i][j] = E[j][i] = dist[function_V[i]][function_V[j]]
        fuction_Graph = MDWGraph(function_V, E)
        return fuction_Graph

    # 辅助功能
    def perm(self, arr):
        """实现全排列"""
        length = len(arr)
        if length == 1:  # 递归出口
            return [arr]

        result = []  # 存储结果
        fixed = arr[0]
        rest = arr[1:]

        for _arr in self.perm(rest):  # 遍历上层的每一个结果
            for i in range(0, length):  # 插入每一个位置得到新序列
                new_rest = _arr.copy()  # 需要复制一份
                new_rest.insert(i, fixed)
                result.append(new_rest)
        return result

    def factorial(self, x):
        """求阶乘"""
        if x == 0:
            return 1
        else:
            return x * self.factorial(x - 1)

    # 一行行输出大矩阵
    def printt(self, M):
        print('图：')
        for i in range(len(M)):
            print('第', i, '行')
            print(M[i])

    # 复制临时的功能结点剩余资源
    def copy_function_V_constrains(self, function_V_constrains):
        temp_function_V_constrains = []
        for a in function_V_constrains:
            temp_function_V_constrains.append(a)

        return temp_function_V_constrains

    # 求expanded PMOD图
    def construct_EPMOD_network(self, function_Graph, function_num, deployment_cost, bk, parallel_num,
                                parallel_position, function_V_constrains):
        if parallel_num == 0 or parallel_num == 1:
            print('No parallel function!')
            return

        dist = function_Graph.floyd_warshall(1)  # 求功能图G的最短路径矩阵

        # 构造EPMOD图
        subgraph_num = self.factorial(parallel_num)
        function_position = list(range(function_num))
        subfunction = list(range(parallel_position - 1, parallel_position + parallel_num - 1))
        subgraph_sorts = self.perm(subfunction)
        # 计算结点数量
        parallel_portion_node = subgraph_num * (len(function_Graph.V)) * parallel_num  # 并行部分的结点数
        not_parallel_portion_node = (len(function_Graph.V)) * (function_num - parallel_num)  # 非并行部分的结点数
        EPMOD_V_num = (parallel_portion_node + not_parallel_portion_node) * 2 + 1  # 求MOD图的结点数目
        EPMOD_V = list(range(EPMOD_V_num))
        EPMOD_E = np.ones((EPMOD_V_num, EPMOD_V_num)) * inf
        # 临时的功能结点剩余资源
        parallel_function_V_constrains = []
        for i in range(subgraph_num):
            parallel_function_V_constrains.append(self.copy_function_V_constrains(function_V_constrains))

        for i in range(function_num):
            # 连接并行功能前面的非并行部分
            if i < parallel_position - 1:
                # 连接部署功能线
                for a in range(function_Graph.vnum):
                    node_i = i * function_Graph.vnum * 2 + a + 1
                    node_j = i * function_Graph.vnum * 2 + a + 1 + function_Graph.vnum
                    if parallel_function_V_constrains[0][a] < deployment_cost[a][i]:
                        EPMOD_E[node_i][node_j] = inf
                        continue
                    EPMOD_E[node_i][node_j] = deployment_cost[a][i]
                    for j in range(len(parallel_function_V_constrains)):
                        parallel_function_V_constrains[j][a] = parallel_function_V_constrains[j][a] - \
                                                               deployment_cost[a][i]
                # 连接功能结点间线
                if i != parallel_position - 2:
                    for a in range(function_Graph.vnum):
                        node_i = i * function_Graph.vnum * 2 + a + 1 + function_Graph.vnum
                        for b in range(function_Graph.vnum):
                            node_j = i * function_Graph.vnum * 2 + b + 1 + 2 * function_Graph.vnum
                            EPMOD_E[node_i][node_j] = dist[a][b] * bk
                else:
                    for a in range(function_Graph.vnum):
                        node_i = i * function_Graph.vnum * 2 + a + 1 + function_Graph.vnum
                        for c in range(subgraph_num):
                            for b in range(function_Graph.vnum):
                                node_j = i * function_Graph.vnum * 2 + b + 1 + (2 + c) * function_Graph.vnum
                                EPMOD_E[node_i][node_j] = dist[a][b] * bk

            # 连接并行结点
            elif i < parallel_position + parallel_num - 1:
                # 连接部署功能线
                for c in range(subgraph_num):
                    # 引入专用于并行功能的临时的功能结点资源限制
                    parallel_temp_function_V_constrains = parallel_function_V_constrains[c]
                    for a in range(function_Graph.vnum):
                        node_i = (parallel_position - 1) * function_Graph.vnum * 2 + a + c * function_Graph.vnum + 1 + (
                                i - parallel_position + 1) * subgraph_num * function_Graph.vnum * 2
                        node_j = (
                                         parallel_position - 1) * function_Graph.vnum * 2 + a + 1 + function_Graph.vnum * c + subgraph_num * function_Graph.vnum + (
                                         i - parallel_position + 1) * subgraph_num * function_Graph.vnum * 2
                        dj = subgraph_sorts[c][i - (parallel_position - 1)]
                        if parallel_temp_function_V_constrains[a] < deployment_cost[a][dj]:
                            EPMOD_E[node_i][node_j] = inf
                            continue
                        EPMOD_E[node_i][node_j] = deployment_cost[a][dj]
                        parallel_temp_function_V_constrains[a] = parallel_temp_function_V_constrains[a] - \
                                                                 deployment_cost[a][dj]

                # 连接功能结点间线
                if i != parallel_position + parallel_num - 2:
                    for c in range(subgraph_num):
                        for a in range(function_Graph.vnum):
                            node_i = (
                                             parallel_position - 1) * function_Graph.vnum * 2 + a + 1 + function_Graph.vnum * c + subgraph_num * function_Graph.vnum + (
                                             i - parallel_position + 1) * subgraph_num * function_Graph.vnum * 2
                            for b in range(function_Graph.vnum):
                                node_j = (
                                                 parallel_position - 1) * function_Graph.vnum * 2 + 1 + function_Graph.vnum * c + subgraph_num * function_Graph.vnum + b + (
                                                 i - parallel_position + 1) * subgraph_num * function_Graph.vnum * 2 + subgraph_num * function_Graph.vnum
                                EPMOD_E[node_i][node_j] = dist[a][b] * bk
                elif i != function_num - 1:
                    for c in range(subgraph_num):
                        for a in range(function_Graph.vnum):
                            node_i = (
                                             parallel_position - 1) * function_Graph.vnum * 2 + a + 1 + function_Graph.vnum * c + subgraph_num * function_Graph.vnum + (
                                             i - parallel_position + 1) * subgraph_num * function_Graph.vnum * 2
                            for b in range(function_Graph.vnum):
                                node_j = (
                                                 parallel_position - 1) * function_Graph.vnum * 2 + subgraph_num * parallel_num * function_Graph.vnum * 2 + 1 + b
                                EPMOD_E[node_i][node_j] = dist[a][b] * bk
            # 连接剩余结点
            else:
                # 连接部署功能线
                for a in range(function_Graph.vnum):
                    node_i = (
                                     parallel_position - 1) * function_Graph.vnum * 2 + subgraph_num * parallel_num * function_Graph.vnum * 2 + a + 1 + function_Graph.vnum * (
                                     i - parallel_position - parallel_num + 1) * 2
                    node_j = (
                                     parallel_position - 1) * function_Graph.vnum * 2 + subgraph_num * parallel_num * function_Graph.vnum * 2 + a + 1 + function_Graph.vnum * (
                                     i - parallel_position - parallel_num + 1) * 2 + function_Graph.vnum
                    if parallel_function_V_constrains[0][a] < deployment_cost[a][i]:
                        EPMOD_E[node_i][node_j] = inf
                        continue
                    EPMOD_E[node_i][node_j] = deployment_cost[a][i]
                    for j in range(len(parallel_function_V_constrains)):
                        parallel_function_V_constrains[j][a] = parallel_function_V_constrains[j][a] - \
                                                               deployment_cost[a][i]
                # 连接功能结点间线
                if i != function_num - 1:
                    for a in range(function_Graph.vnum):
                        node_i = (
                                         parallel_position - 1) * function_Graph.vnum * 2 + subgraph_num * parallel_num * function_Graph.vnum * 2 + a + 1 + function_Graph.vnum * (
                                         i - parallel_position - parallel_num + 1) * 2 + function_Graph.vnum
                        for b in range(function_Graph.vnum):
                            node_j = (
                                             parallel_position - 1) * function_Graph.vnum * 2 + subgraph_num * parallel_num * function_Graph.vnum * 2 + b + 1 + function_Graph.vnum * (
                                             i - parallel_position - parallel_num + 1) * 2 + 2 * function_Graph.vnum
                            EPMOD_E[node_i][node_j] = dist[a][b] * bk
        # print(EPMOD_V_num)
        # print(EPMOD_V)
        # self.printt(EPMOD_E)

        return EPMOD_V_num, EPMOD_V, EPMOD_E

    # 构造辅助图
    def construct_auxiliary_graph(self, initial_Graph, EPMOD_V_num, EPMOD_V, EPMOD_E, function_V, parallel_num,
                                  parallel_position, function_num, sk):
        bk = sk['bk']
        # 构造新的功能节点集合
        new_function_V = []
        for v in function_V:
            new_function_V.append(v + EPMOD_V_num)

        # 构造auxiliary图
        AG_V_num = initial_Graph.vnum + EPMOD_V_num
        AG_V = list(range(AG_V_num))
        AG_temp = np.matrix(EPMOD_E)
        newcol = np.matrix(np.ones((EPMOD_V_num, initial_Graph.vnum)) * inf)
        new_col_Graph = np.c_[AG_temp, newcol]
        newrow = np.matrix(np.ones((initial_Graph.vnum, AG_V_num)) * inf)
        AG_E = np.array(np.r_[new_col_Graph, newrow])
        for i in range(initial_Graph.vnum):
            for j in range(i, initial_Graph.vnum):
                if initial_Graph.E[i][j] != inf:
                    AG_E[i + EPMOD_V_num][j + EPMOD_V_num] = AG_E[j + EPMOD_V_num][i + EPMOD_V_num] = \
                    initial_Graph.E[i][
                        j] * bk
                else:
                    AG_E[i + EPMOD_V_num][j + EPMOD_V_num] = AG_E[j + EPMOD_V_num][i + EPMOD_V_num] = inf
        # 连接虚拟功能结点到实际功能结点
        if parallel_num + parallel_position - 1 == function_num:
            subgraph_num = self.factorial(parallel_num)
            last_col_num = subgraph_num * len(function_V)
            for i in range(subgraph_num):
                for j in range(len(function_V)):
                    row = EPMOD_V_num - last_col_num + i * len(function_V) + j
                    AG_E[row][new_function_V[j]] = 0
        else:
            last_col_num = len(function_V)
            for i in range(len(function_V)):
                row = EPMOD_V_num - last_col_num + i
                AG_E[row][new_function_V[i]] = 0

        # 连接源节点与EPMOD图的边
        source = sk['source']
        initial_dist, initial_parent = initial_Graph.floyd_warshall(0)
        if parallel_position == 1:
            subgraph_num = self.factorial(parallel_num)
            for i in range(subgraph_num):
                for j in range(len(function_V)):
                    col = i * len(function_V) + j + 1
                    AG_E[0][col] = initial_dist[source][function_V[j]] * bk
        else:
            for i in range(len(function_V)):
                AG_E[0][i + 1] = initial_dist[source][function_V[i]] * bk

        Auxiliary_Graph = MDWGraph(AG_V, AG_E)
        # self.printt(Auxiliary_Graph.E)

        return Auxiliary_Graph, new_function_V, initial_parent

    # dijkstra求最短路径
    def dijkstra_get_shortest_path(self, Auxiliary_Graph, sk, EPMOD_V_num, initial_parent, EPMOD_E, function_V,
                                   parallel_position, parallel_num,
                                   function_num):
        destination = sk['destination']
        new_destination = destination + EPMOD_V_num
        dijkstra_result = Auxiliary_Graph.dijkstra(0)
        route = Auxiliary_Graph.dijkstra_route(dijkstra_result, 0, new_destination)
        weight = dijkstra_result[new_destination]['dist']
        # print(route)
        # print('花费的权值为：', weight)
        # 处理结果路径
        if route != 0:
            result_route = []
            for v in route:
                if v == 0:
                    result_route.append(sk['source'])
                elif v < EPMOD_V_num:
                    result_route.append(function_V[(v - 1) % len(function_V)])
                else:
                    result_route.append(v - EPMOD_V_num)
            real_route = []
            pre_v = sk['source']
            real_route.append(pre_v)
            for i in range(1, len(result_route)):
                v = result_route[i]
                if pre_v != v:
                    route_temp = self.find_route(initial_parent, pre_v, v)
                    for j in range(1, len(route_temp)):
                        real_route.append(route_temp[j])
                else:
                    real_route.append(v)
                pre_v = v

            # 求最短路径
            shortest_path = []
            pre_v = sk['source']
            shortest_path.append(pre_v)
            for i in range(1, len(real_route)):
                if real_route[i] != pre_v:
                    shortest_path.append(real_route[i])
                pre_v = real_route[i]
            # 求部署结点
            deploy_server = []
            flag = False
            for i in range(len(real_route)):
                if real_route[i] in function_V:
                    if flag == False and real_route[i + 1] == real_route[i]:
                        flag = True
                    elif flag == True:
                        deploy_server.append(real_route[i])
                        flag = False
            # print("最短路径为：", shortest_path)
            # print("部署的服务器结点为：", deploy_server)
            function_order = self.find_function_order(EPMOD_E, route, function_V, parallel_position, parallel_num,
                                                      function_num)
            return shortest_path, deploy_server, function_order, weight
        else:
            return 0, 0, 0, 0

    # 求部署功能的顺序
    def find_function_order(self, EPMOD_E, path, function_V, parallel_position, parallel_num, function_num):
        deploy_function = []
        parallel_function = list(np.array(list(range(parallel_num))) + parallel_position)
        parallel_function_sorts = self.perm(parallel_function)
        subgraph_num = self.factorial(parallel_num)
        parallel_range_start = ((parallel_position - 1) * len(function_V)) * 2 + 1
        parallel_range_end = ((parallel_position - 1) * len(function_V) + subgraph_num * parallel_num * len(
            function_V)) * 2 + 1
        parallel_range = list(range(parallel_range_start, parallel_range_end))
        if parallel_position == 1:
            parallel_right = parallel_position + parallel_num

        elif parallel_position + parallel_num - 1 != function_num:
            parallel_left = 1
            parallel_right = parallel_position + parallel_num
        elif parallel_position + parallel_num - 1 == function_num:
            parallel_left = 1
        f = 0
        for i in range(1, function_num + 1):
            if i < parallel_position:
                deploy_function.append(i)
            elif i <= parallel_position + parallel_num - 1:
                f += 1
                if f == parallel_num:
                    for a in path:
                        if a in parallel_range:
                            break
                    x = int((a - (parallel_position - 1) * len(function_V) * 2 - 1) / len(function_V))
                    for b in parallel_function_sorts[x]:
                        deploy_function.append(b)
            else:
                deploy_function.append(i)
        # print('部署的功能顺序为：', deploy_function)
        return deploy_function

    # 求实际图最短路径
    def find_route(self, parent, a, b):
        route = []
        route.append(b)
        while a != b:
            b_pre = int(parent[b][a])
            b = b_pre
            route.append(b)
        route.reverse()
        return route

    # 更新资源
    def update_constrains(self, new_initial_Graph, E_constrains, function_V_constrains, shortest_path, deploy_server,
                          function_order, function_V, sk):
        # 更新链路资源
        pre_v = shortest_path[0]
        bk = sk['bk']
        deployment_cost = sk['deployment_cost']
        for i in range(1, len(shortest_path)):
            v = shortest_path[i]
            E_constrains[pre_v][v] = E_constrains[pre_v][v] - bk * new_initial_Graph.E[pre_v][v]
            E_constrains[v][pre_v] = E_constrains[v][pre_v] - bk * new_initial_Graph.E[v][pre_v]
            pre_v = v

        # 更新结点资源
        for i in range(len(deploy_server)):
            fv = function_V.index(deploy_server[i])
            f = function_order[i] - 1
            function_V_constrains[fv] = function_V_constrains[fv] - deployment_cost[fv][f]

        return E_constrains, function_V_constrains

    # 处理单个请求
    def handle_a_request(self, sk, initial_Graph, function_V, E_constrains, function_V_constrains):
        source = sk['source']
        destination = sk['destination']
        SFC = sk['SFC']
        deployment_cost = sk['deployment_cost']
        bk = sk['bk']
        parallel_num = sk['parallel_num']  # 并行功能数
        function_num = sk['function_num']
        parallel_position = sk['parallel_position']
        # 更新原始图
        new_initial_Graph = self.deal_initial_Graph(initial_Graph, E_constrains, bk)
        function_Graph = self.get_fuction_Graph(new_initial_Graph, function_V)
        EPMOD_V_num, EPMOD_V, EPMOD_E = self.construct_EPMOD_network(function_Graph, function_num, deployment_cost, bk,
                                                                     parallel_num, parallel_position,
                                                                     function_V_constrains)
        Auxiliary_Graph, new_function_V, initial_parent = self.construct_auxiliary_graph(new_initial_Graph, EPMOD_V_num,
                                                                                         EPMOD_V,
                                                                                         EPMOD_E,
                                                                                         function_V, parallel_num,
                                                                                         parallel_position,
                                                                                         function_num, sk)
        shortest_path, deploy_server, function_order, weight = self.dijkstra_get_shortest_path(Auxiliary_Graph, sk,
                                                                                               EPMOD_V_num,
                                                                                               initial_parent, EPMOD_E,
                                                                                               function_V,
                                                                                               parallel_position,
                                                                                               parallel_num,
                                                                                               function_num)
        if shortest_path == 0:
            print("资源不足，拒绝请求")
            return 0, 0, 0, 0
        else:
            return shortest_path, deploy_server, function_order, weight

    # 处理时延图的请求
    def handle_a_delay_requets(self, sk, delay_Graph, function_V, E_constrains, function_V_constrains):
        deployment_cost = np.zeros((len(function_V), sk['function_num']))
        bk = sk['bk']
        alpha = sk['alpha']
        for i in range(len(deployment_cost)):
            for j in range(len(deployment_cost[i])):
                deployment_cost[i][j] = alpha[i] * bk
        parallel_num = sk['parallel_num']  # 并行功能数
        function_num = sk['function_num']
        parallel_position = sk['parallel_position']
        # 更新原始图
        new_delay_Graph = self.deal_initial_Graph(delay_Graph, E_constrains, bk)
        function_Graph = self.get_fuction_Graph(new_delay_Graph, function_V)
        EPMOD_V_num, EPMOD_V, EPMOD_E = self.construct_EPMOD_network(function_Graph, function_num, deployment_cost, bk,
                                                                     parallel_num, parallel_position,
                                                                     function_V_constrains)
        Auxiliary_Graph, new_function_V, initial_parent = self.construct_auxiliary_graph(new_delay_Graph, EPMOD_V_num,
                                                                                         EPMOD_V,
                                                                                         EPMOD_E,
                                                                                         function_V, parallel_num,
                                                                                         parallel_position,
                                                                                         function_num, sk)
        shortest_path, deploy_server, function_order, weight = self.dijkstra_get_shortest_path(Auxiliary_Graph, sk,
                                                                                               EPMOD_V_num,
                                                                                               initial_parent, EPMOD_E,
                                                                                               function_V,
                                                                                               parallel_position,
                                                                                               parallel_num,
                                                                                               function_num)
        if shortest_path == 0:
            print("资源不足，拒绝请求")
            return 0, 0, 0, 0
        else:
            return shortest_path, deploy_server, function_order, weight

    # 处理复合图的请求
    def handle_a_complex_requets(self, sk, complex_Graph, function_V, E_constrains, function_V_constrains, landa):
        deployment_delay = np.zeros((len(function_V), sk['function_num']))
        bk = sk['bk']
        alpha = sk['alpha']
        for i in range(len(deployment_delay)):
            for j in range(len(deployment_delay[i])):
                deployment_delay[i][j] = alpha[i] * bk
        deployment_cost = sk['deployment_cost'] + landa * deployment_delay
        parallel_num = sk['parallel_num']  # 并行功能数
        function_num = sk['function_num']
        parallel_position = sk['parallel_position']
        # 更新原始图
        new_complex_Graph = self.deal_initial_Graph(complex_Graph, E_constrains, bk)
        function_Graph = self.get_fuction_Graph(new_complex_Graph, function_V)
        EPMOD_V_num, EPMOD_V, EPMOD_E = self.construct_EPMOD_network(function_Graph, function_num, deployment_cost, bk,
                                                                     parallel_num, parallel_position,
                                                                     function_V_constrains)
        Auxiliary_Graph, new_function_V, initial_parent = self.construct_auxiliary_graph(new_complex_Graph, EPMOD_V_num,
                                                                                         EPMOD_V,
                                                                                         EPMOD_E,
                                                                                         function_V, parallel_num,
                                                                                         parallel_position,
                                                                                         function_num, sk)
        shortest_path, deploy_server, function_order, weight = self.dijkstra_get_shortest_path(Auxiliary_Graph, sk,
                                                                                               EPMOD_V_num,
                                                                                               initial_parent, EPMOD_E,
                                                                                               function_V,
                                                                                               parallel_position,
                                                                                               parallel_num,
                                                                                               function_num)
        if shortest_path == 0:
            print("资源不足，拒绝请求")
            return 0, 0, 0, 0
        else:
            return shortest_path, deploy_server, function_order, weight

    # 计算结果的时延
    def compute_delay(self, shortest_path, deploy_server, link_delay_matrix, sk, function_V):
        bk = sk['bk']

        # 计算链路时延
        link_delay = 0
        for i in range(1, len(shortest_path)):
            pre_v = shortest_path[i - 1]
            v = shortest_path[i]
            link_delay += link_delay_matrix[pre_v][v] * bk

        # 计算处理时延
        process_delay = 0
        alpha = sk['alpha']
        for v in deploy_server:
            i = function_V.index(v)
            process_delay += alpha[i] * bk

        # 总时延
        return link_delay + process_delay

    # floyd求结点与路径间的最短路径
    def node_to_route_dist(self, distance, route, v):
        dist = []
        for a in route:
            dist.append(distance[a][v])
        return min(dist)

    # 辅助功能
    def perm(self, arr):
        """实现全排列"""
        length = len(arr)
        if length == 1:  # 递归出口
            return [arr]

        result = []  # 存储结果
        fixed = arr[0]
        rest = arr[1:]

        for _arr in self.perm(rest):  # 遍历上层的每一个结果
            for i in range(0, length):  # 插入每一个位置得到新序列
                new_rest = _arr.copy()  # 需要复制一份
                new_rest.insert(i, fixed)
                result.append(new_rest)
        return result

    # 找到一条包含某些结点的最短路径
    def find_route_with_nodes(self, G, nodes, a, b):
        dist_matrix, parent = G.floyd_warshall(0)
        nodes_array = self.perm(nodes)
        min_dist = inf * 10086
        for value in nodes_array:
            dist = 0
            for i in range(len(value) - 1):
                dist = dist + dist_matrix[value[i]][value[i + 1]]
                if dist > min_dist:
                    i = len(value) - 2
                    break
            dist = dist + dist_matrix[a][value[0]] + dist_matrix[value[i + 1]][b]
            if dist < min_dist:
                min_dist = dist
                min_route = value
        # new_shortest_path = []
        # route_a_to_nodes = self.find_route(parent, a, min_route[0])
        # for v in route_a_to_nodes:
        #     new_shortest_path.append(v)
        # for i in range(len(min_route) - 1):
        #     route_server_to_server = self.find_route(parent, min_route[i], min_route[i + 1])
        #     for v in route_server_to_server:
        #         new_shortest_path.append(v)
        # route_nodes_to_b = self.find_route(parent, b, min_route[len(min_route) - 1])
        # for v in route_nodes_to_b:
        #     new_shortest_path.append(v)

        return min_route

    # 计算花费的权值
    def compute_cost(self, initial_Graph, shortest_path, deploy_server, function_order, sk, function_V):
        deployment_cost = sk['deployment_cost']
        weight = 0
        for i in range(len(shortest_path) - 1):
            vi = shortest_path[i]
            vj = shortest_path[i + 1]
            weight += initial_Graph.E[vi][vj] * sk['bk']
        for i in range(len(deploy_server)):
            vi = function_V.index(deploy_server[i])
            vj = function_order[i] - 1
            weight += deployment_cost[vi][vj]
        return weight

    # 处理一条时延请求
    def modify_delay_request(self, sk, initial_Graph, function_V, E_constrains, function_V_constrains, link_delay):
        shortest_path_Pc, deploy_server_Pc, function_order_Pc, weight_landa_Pc = self.handle_a_request(sk,
                                                                                                       initial_Graph,
                                                                                                       function_V,
                                                                                                       E_constrains,
                                                                                                       function_V_constrains)  # 处理单条请求不考虑时延

        if shortest_path_Pc == 0:
            return 0, 0, 0, 0
        # 如果时延直接满足,则输出结果即可
        delay_Pc = self.compute_delay(shortest_path_Pc, deploy_server_Pc, link_delay, sk, function_V)
        weight_Pc = self.compute_cost(initial_Graph, shortest_path_Pc, deploy_server_Pc, function_order_Pc, sk,
                                          function_V)
        if delay_Pc <= sk['dk']:
            E_constrains, function_V_constrains = self.update_constrains(initial_Graph, E_constrains,
                                                                         function_V_constrains,
                                                                         shortest_path_Pc, deploy_server_Pc,
                                                                         function_order_Pc, function_V, sk)
            print("*******************************最终结果********************************")
            print("最短路径为：", shortest_path_Pc)
            print("部署的服务器结点为：", deploy_server_Pc)
            print('部署的功能顺序为：', function_order_Pc)
            print('最终时延为：', delay_Pc)
            print('最终消耗原为：', weight_landa_Pc)
            print('最终消耗计算后为：', weight_Pc)
            print('剩余结点资源为：', function_V_constrains)
            print('剩余链路资源为：', E_constrains)
            return shortest_path_Pc, deploy_server_Pc, function_order_Pc, weight_Pc
        # 否则以时延作为权值求最短路劲
        else:
            print('时延为：', delay_Pc, '不满足要求，进行调整！')
            delay_E = np.zeros((initial_Graph.vnum, initial_Graph.vnum))
            for i in range(len(delay_E)):
                for j in range(i + 1, len(delay_E)):
                    if initial_Graph.E[i][j] != inf:
                        delay_E[i][j] = delay_E[j][i] = link_delay[i][j]
                    else:
                        delay_E[i][j] = delay_E[j][i] = inf
            delay_Graph = MDWGraph(initial_Graph.V, delay_E)
            # self.printt(delay_E)
            shortest_path_Pd, deploy_server_Pd, function_order_Pd, weight_landa_Pd = self.handle_a_delay_requets(sk,
                                                                                                                 delay_Graph,
                                                                                                                 function_V,
                                                                                                                 E_constrains,
                                                                                                                 function_V_constrains)
            weight_Pd = self.compute_cost(initial_Graph, shortest_path_Pd, deploy_server_Pd, function_order_Pd, sk,
                                          function_V)

            # 如果时延的最短路径还不满足,则拒绝请求
            delay_Pd = self.compute_delay(shortest_path_Pd, deploy_server_Pd, link_delay, sk, function_V)
            if delay_Pd > sk['dk']:
                print('最低时延:' + str(delay_Pd) + '。不满足,拒绝请求!')
                return 0, 0, 0, 0
            else:
                print('最低时延:' + str(delay_Pd))
                flag = False
                while flag != True:
                    print('weightpc', weight_Pc)
                    print('weightpd', weight_Pd)
                    print('delaypd', delay_Pd)
                    print('delaypc', delay_Pc)
                    landa = (weight_Pc - weight_Pd) / (delay_Pd - delay_Pc)
                    if landa < 0 and delay_Pd<sk['dk']:# gaidong
                        break
                    print(landa)
                    complex_E = np.zeros((initial_Graph.vnum, initial_Graph.vnum))
                    for i in range(len(complex_E)):
                        for j in range(i + 1, len(complex_E)):
                            if initial_Graph.E[i][j] != inf:
                                complex_E[i][j] = complex_E[j][i] = initial_Graph.E[i][j] + landa * link_delay[i][j]
                            else:
                                complex_E[i][j] = complex_E[j][i] = inf
                    # self.printt(complex_E)
                    # complex_E = initial_Graph.E + landa * link_delay
                    complex_Graph = MDWGraph(initial_Graph.V, complex_E)
                    shortest_path_Pr, deploy_server_Pr, function_order_Pr, weight_landa_Pr = self.handle_a_complex_requets(
                        sk,
                        complex_Graph,
                        function_V,
                        E_constrains,
                        function_V_constrains,
                        landa)
                    if weight_landa_Pr == weight_landa_Pd:
                        break
                    # self.printt(initial_Graph.E)
                    # self.printt(complex_E)
                    if shortest_path_Pr == 0:
                        if delay_Pd <= sk['dk']:
                            print("*******************************最终结果********************************")
                            print("最短路径为：", shortest_path_Pd)
                            print("部署的服务器结点为：", deploy_server_Pd)
                            print('部署的功能顺序为：', function_order_Pd)
                            print('最终时延为：', delay_Pd)
                            print('最终消耗为：', weight_Pd)
                            print('剩余结点资源为：', function_V_constrains)
                            print('剩余链路资源为：', E_constrains)

                            return shortest_path_Pd, deploy_server_Pd, function_order_Pd, weight_Pd
                        else:
                            print('无匹配结果！')
                            return 0, 0, 0, 0
                        continue
                    weight_Pr = self.compute_cost(initial_Graph, shortest_path_Pr, deploy_server_Pr, function_order_Pr,
                                                  sk,
                                                  function_V)
                    # if weight_landa_Pr == weight_landa_Pc or weight_landa_Pr == weight_landa_Pd:  # pd则可以gaidong
                    print('weight_landa_Pr:', weight_landa_Pr)
                    print('weight_landa_Pc:', weight_landa_Pc)
                    print('weight_landa_Pd:', weight_landa_Pd)
                    if weight_landa_Pr == weight_landa_Pc:  # or weight_landa_Pr == weight_landa_Pd:
                        flag = True
                    else:
                        delay_Pr = self.compute_delay(shortest_path_Pr, deploy_server_Pr, link_delay, sk, function_V)
                        if delay_Pr <= sk['dk']:
                            shortest_path_Pd = shortest_path_Pr
                            deploy_server_Pd = deploy_server_Pr
                            function_order_Pd = function_order_Pr
                            weight_landa_Pd = weight_landa_Pr
                            weight_Pd = weight_Pr
                            delay_Pd = delay_Pr
                        else:
                            shortest_path_Pc = shortest_path_Pr
                            deploy_server_Pc = deploy_server_Pr
                            function_order_Pc = function_order_Pr
                            weight_landa_Pc = weight_landa_Pr
                            weight_Pc = weight_Pr
                            delay_Pc = delay_Pr

                E_constrains, function_V_constrains = self.update_constrains(initial_Graph, E_constrains,
                                                                             function_V_constrains,
                                                                             shortest_path_Pd, deploy_server_Pd,
                                                                             function_order_Pd, function_V, sk)
                print("*******************************最终结果********************************")
                print("最短路径为：", shortest_path_Pd)
                print("部署的服务器结点为：", deploy_server_Pd)
                print('部署的功能顺序为：', function_order_Pd)
                print('最终时延为：', delay_Pd)
                print('最终消耗为：', weight_Pd)
                print('剩余结点资源为：', function_V_constrains)
                print('剩余链路资源为：', E_constrains)

                return shortest_path_Pd, deploy_server_Pd, function_order_Pd, weight_Pd

    # # 将nk'-nk个结点上的功能一个个部署到nk个结点上
    # def reduce_deploy_num(self, shortest_path, deploy_server, function_order, deploy_server_num, n_k, new_initial_Graph,
    #                       sk, function_V_constrains, link_delay, function_V):
    #     # 求需要重新插入的功能与新的功能结点集合
    #     # dist, parent = new_initial_Graph.floyd_warshall(0)
    #     # route = new_initial_Graph.find_floyd_route(parent, sk['source'], sk['destination'])
    #     delay_Graph = MDWGraph(list(range(len(link_delay))), link_delay)
    #     dist, parent = delay_Graph.floyd_warshall(0)
    #     route = delay_Graph.find_floyd_route(parent, sk['source'], sk['destination'])
    #     deployment_cost = sk['deployment_cost']
    #     new_deploy_server = []
    #     new_deploy_server = list(set(deploy_server))
    #     new_deploy_server_dist = []
    #     for v in new_deploy_server:
    #         new_deploy_server_dist.append(self.node_to_route_dist(dist, route, v))
    #
    #     for i in range(deploy_server_num - n_k):
    #         a = min(new_deploy_server_dist)
    #         new_deploy_server.pop(new_deploy_server_dist.index(a))
    #
    #     new_deploy_server2 = []
    #     for v in deploy_server:
    #         if v in new_deploy_server:
    #             new_deploy_server2.append(v)
    #     new_deploy_server = new_deploy_server2
    #     for i in range(len(deploy_server)):
    #         if deploy_server[i] not in new_deploy_server:
    #             flag = 0
    #             # 插入左边相邻服务器
    #             v_i = i - 1
    #             while flag == 0:
    #                 if v_i == -1:
    #                     v_i = i + 1
    #                     break
    #                 v = new_deploy_server[v_i]
    #                 v_dist = function_V.index(v)
    #                 if function_V_constrains[v_dist] > deployment_cost[v_dist][function_order[i] - 1]:
    #                     new_deploy_server.insert(v_i, v)
    #                     flag = 1
    #                 else:
    #                     v_i -= 1
    #             # 插入右边相邻服务器
    #             while flag == 0:
    #                 if v_i >= len(new_deploy_server):
    #                     print("减少部署结点时，选中的其余结点资源不足，返回原方案。")
    #                     return shortest_path, deploy_server, function_order
    #                 v = new_deploy_server[v_i]
    #                 v_dist = function_V.index(v)
    #                 if function_V_constrains[v_dist] > deployment_cost[v_i][function_order[i]]:
    #                     new_deploy_server.insert(v_i, v)
    #                     flag = 1
    #                 else:
    #                     v_i += 1
    #
    #     # 计算新的最短路径
    #     new_shortest_path = []
    #     # 源节点到第一个功能结点的最短路径
    #     route_source_to_server = self.find_route(parent, sk['source'], new_deploy_server[0])
    #     for v in route_source_to_server:
    #         new_shortest_path.append(v)
    #     # 服务器间的最短路径
    #     for i in range(len(new_deploy_server) - 1):
    #         route_server_to_server = self.find_route(parent, new_deploy_server[i], new_deploy_server[i + 1])
    #         for i in range(1, len(route_server_to_server)):
    #             new_shortest_path.append(route_server_to_server[i])
    #     # 服务器到终点的最短路径
    #     destination = sk['destination']
    #     route_server_to_destination = self.find_route(parent, new_deploy_server[len(new_deploy_server) - 1],
    #                                                   destination)
    #     for i in range(1, len(route_server_to_destination)):
    #         new_shortest_path.append(route_server_to_destination[i])
    #
    #     return new_shortest_path, new_deploy_server, function_order
    #
    # # 找出额外的nk-nk'个服务器结点来部署功能
    #
    # # 增加部署的服务器结点
    # def add_deploy_num(self, shortest_path, deploy_server, function_order, deploy_server_num, n_k, new_initial_Graph,
    #                    sk, function_V_constrains, link_delay, function_V):
    #     deployment_cost = sk['deployment_cost']
    #     new_deploy_server = []  # 新部署的结点集合
    #     # dist, parent = new_initial_Graph.floyd_warshall(0)
    #     # route = new_initial_Graph.find_floyd_route(parent, sk['source'], sk['destination'])
    #     # 改为时延最短路
    #     delay_Graph = MDWGraph(list(range(len(link_delay))), link_delay)
    #     dist, parent = delay_Graph.floyd_warshall(0)
    #     route = delay_Graph.find_floyd_route(parent, sk['source'], sk['destination'])
    #     function_V_dist = []  # 功能节点到最短路径的距离
    #     for v in function_V:
    #         function_V_dist.append(self.node_to_route_dist(dist, route, v))
    #     # 求新的部署结点
    #     new_deploy_num = 0
    #     i = 0
    #     max_deployment_cost = 0
    #     for a in deployment_cost:
    #         for b in a:
    #             if max_deployment_cost < b:
    #                 max_deployment_cost = b
    #     while new_deploy_num != (n_k - deploy_server_num):
    #         min_dist = min(function_V_dist)
    #         min_i = function_V_dist.index(min_dist)
    #         if function_V[min_i] not in deploy_server and function_V[min_i] not in new_deploy_server:
    #             if function_V_constrains[min_i] >= max_deployment_cost:
    #                 new_deploy_server.append(function_V[min_i])
    #                 new_deploy_num += 1
    #         function_V_dist[min_i] = inf
    #         i += 1
    #         if i == len(function_V):
    #             print("没有合适的功能结点，增加失败。返回原结果！")
    #             return shortest_path, deploy_server, function_order
    #     # 所有要待部署的结点
    #     all_deploy_server = list(set(deploy_server))
    #     for v in new_deploy_server:
    #         all_deploy_server.append(v)
    #
    #     # 新部署的路径
    #     new_deploy_server = self.find_route_with_nodes(new_initial_Graph, all_deploy_server, sk['source'],
    #                                                    sk['destination'])
    #
    #     # 重新部署功能
    #     print("新部署的结点：", new_deploy_server)
    #     i = 1
    #     while (len(new_deploy_server)) != (len(deploy_server)):
    #         if deploy_server[i] == deploy_server[i - 1]:
    #             new_deploy_server.insert(i, deploy_server[i])
    #         i += 1
    #         # if i == len(deploy_server):
    #         #     break
    #     # 计算新的最短路径
    #     new_shortest_path = []
    #     # 源节点到第一个功能结点的最短路径
    #     route_source_to_server = self.find_route(parent, sk['source'], new_deploy_server[0])
    #     for v in route_source_to_server:
    #         new_shortest_path.append(v)
    #     # 服务器间的最短路径
    #     for i in range(len(new_deploy_server) - 1):
    #         route_server_to_server = self.find_route(parent, new_deploy_server[i], new_deploy_server[i + 1])
    #         for i in range(1, len(route_server_to_server)):
    #             new_shortest_path.append(route_server_to_server[i])
    #     # 服务器到终点的最短路径
    #     destination = sk['destination']
    #     route_server_to_destination = self.find_route(parent, new_deploy_server[len(new_deploy_server) - 1],
    #                                                   destination)
    #     for i in range(1, len(route_server_to_destination)):
    #         new_shortest_path.append(route_server_to_destination[i])
    #
    #     # print(new_shortest_path)
    #     # print(new_deploy_server)
    #
    #     return new_shortest_path, new_deploy_server, function_order

    # # 处理时延请求
    # def handle_a_delay_requests(self, initial_Graph, function_V, sk, E_constrains, function_V_constrains, link_delay):
    #     # 不考虑时延的情况下找出最短路径
    #     shortest_path, deploy_server, function_order = self.handle_a_request(sk, initial_Graph, function_V,
    #                                                                          E_constrains,
    #                                                                          function_V_constrains)  # 处理单条请求不考虑时延
    #     new_initial_Graph = self.deal_initial_Graph(initial_Graph, E_constrains, sk['bk'])
    #     delay = self.compute_delay(shortest_path, deploy_server, function_order, link_delay, sk)
    #     if delay <= sk['dk']:
    #         return shortest_path, deploy_server, function_order
    #     # 调整最短路径至满足请求的时延
    #     n_min = 1
    #     n_max = sk['function_num']
    #     print("时延：", delay)
    #     while n_min <= n_max:
    #         deploy = list(set(deploy_server))
    #         deploy_server_num = len(deploy)
    #         n_k = int((n_min + n_max) / 2)
    #         if n_k < deploy_server_num:
    #             new_shortest_path, new_deploy_server, function_order = self.reduce_deploy_num(shortest_path,
    #                                                                                           deploy_server,
    #                                                                                           function_order,
    #                                                                                           deploy_server_num,
    #                                                                                           n_k, new_initial_Graph,
    #                                                                                           sk, function_V_constrains,
    #                                                                                           link_delay, function_V)
    #             print("减少结点的结果：")
    #             print('新的最短路径：', new_shortest_path)
    #             print('新的部署服务器：', new_deploy_server)
    #         elif n_k > deploy_server_num:
    #             new_shortest_path, new_deploy_server, function_order = self.add_deploy_num(shortest_path, deploy_server,
    #                                                                                        function_order,
    #                                                                                        deploy_server_num,
    #                                                                                        n_k, new_initial_Graph,
    #                                                                                        sk, function_V_constrains,
    #                                                                                        link_delay, function_V)
    #             print("增加结点的结果：")
    #             print('新的最短路径：', new_shortest_path)
    #             print('新的部署服务器：', new_deploy_server)
    #         else:
    #             new_shortest_path = shortest_path
    #             new_deploy_server = deploy_server
    #
    #         shortest_path = new_shortest_path
    #         deploy_server = new_deploy_server
    #         old_delay = delay
    #         delay = self.compute_delay(shortest_path, deploy_server, function_order, link_delay, sk)
    #         if delay <= sk['dk']:
    #             return shortest_path, deploy_server, function_order
    #         else:
    #             new_delay = self.compute_delay(shortest_path, deploy_server, function_order, link_delay, sk)
    #             if delay < old_delay:
    #                 n_max = n_k - 1
    #             else:
    #                 n_min = n_k + 1
    #         print("时延：", delay)
    #     return 0, 0, 0

# p = Algorithm2()
# initial_Graph, function_V, E_constrains, function_V_constrains, link_delay = p.input_initial_Graph()
# function_Graph = p.get_fuction_Graph(initial_Graph, function_V)
# parallel_num = 3  # 并行功能数
# function_num = 4
# parallel_position = 1
# # deployment_cost = [[1, 4, 3, 4, 2, 5], [2, 4, 4, 3, 4, 5], [3, 3, 3, 2, 2, 3], [2, 3, 2, 3, 4, 3]]
# deployment_cost = [[1, 4, 3, 4], [2, 4, 4, 3], [3, 3, 3, 2], [2, 3, 2, 3]]
# sk = {'source': 0, 'SFC': [1, 2, 3, 4], 'destination': 8, 'bk': 2, 'deployment_cost': deployment_cost,
#       'parallel_num': 3, 'function_num': 4, 'parallel_position': 1, 'dk': 22, 'alpha': [0.8, 0.8, 0.8, 0.8]}
# bk = sk['bk']
# print(E_constrains)
# print(function_V_constrains)
# # for i in range(10):
# #     p.handle_a_request(sk, initial_Graph, function_V, E_constrains, function_V_constrains)
# shortest_path, deploy_server, function_order, weight = p.modify_delay_request(sk, initial_Graph, function_V,
#                                                                               E_constrains, function_V_constrains,
#                                                                               link_delay)
# print(shortest_path, deploy_server, function_order, weight)

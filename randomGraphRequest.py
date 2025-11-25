from MGraph import MDWGraph
import numpy as np
import random

inf = 999999

global graphinfos


class randomGR():
    # 随机生成图：随机生成M个结点，连边概率p条边的图，为保证网络质量，每个结点至少有一条边与其他结点相连
    def random_initial_Graph(self, N, function_V_num, p, max_E_cost):
        V = list(range(N))
        while True:
            E = np.zeros((N, N))
            for i in range(len(E)):
                for j in range(len(E)):
                    if i != j and E[i][j] == 0:
                        E[i][j] = E[j][i] = inf
            # 随机产生边
            for i in range(N):
                flag = 0
                for j in range(N):
                    if j == i:
                        continue
                    prob = random.random()
                    if prob < p:
                        flag = 1
                        Eij_cost = random.randint(1, max_E_cost)  # 随机产生边权
                        E[i][j] = E[j][i] = Eij_cost
                if flag == 0:
                    j = random.randint(0, N - 1)  # 随机产生相连结点
                    if i != j:
                        Eij_cost = random.randint(1, max_E_cost)  # 随机产生边权
                        E[i][j] = E[j][i] = Eij_cost
            initial_Graph = MDWGraph(V, E)
            dist = initial_Graph.floyd_warshall(1)
            flag = False
            for i in range(N):
                if dist[0][i] > inf:
                    flag = True
            if flag == False:
                break

        # 设置链路时延
        link_delay = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if E[i][j] != inf:
                    de = random.randint(1, 10)
                    link_delay[i][j] = link_delay[j][i] = de
                else:
                    link_delay[i][j] = link_delay[j][i] = inf

        # 随机产生功能结点
        function_V = []
        while True:
            if len(function_V) < function_V_num:
                F_V = random.randint(0, N - 1)
                if F_V not in function_V:
                    function_V.append(F_V)
            else:
                break

        # 随机产生的边资源限制
        E_constrains = np.zeros((N, N))
        for i in range(len(E)):
            for j in range(i + 1, len(E)):
                if E[i][j] != inf and i != j:
                    # E_constrains[i][j] = E_constrains[j][i] = random.randint(1000, 10000)
                    E_constrains[i][j] = E_constrains[j][i] = random.randint(50, 150)

        # 随机产生功能结点限制
        function_V_constrains = []
        for i in range(function_V_num):
            # function_V_constrains.append(random.randint(4000, 12000))
            # function_V_constrains.append(random.randint(10000, 20000))
            function_V_constrains.append(1000)
        # 输出随机产生的图
        print('随机产生的结点个数：', initial_Graph.vnum)
        print('产生结点集合为：', initial_Graph.V)
        print('产生的边集合为：')
        print(initial_Graph.E)
        print('产生的链路时延是：', link_delay)
        print('产生的功能结点分别是：', function_V)
        print('随机产生的边资源限制:\n', E_constrains)
        print('随机产生功能结点限制:\n', function_V_constrains)

        return initial_Graph, function_V, E_constrains, function_V_constrains, link_delay

    # 随机产生请求request
    def random_request(self, N, function_num, function_V_num, max_bk, deployment_costs,
                       parallel_num, function_V, parallel_position) -> object:
        request = {}
        while True:
            source = random.randint(0, N - 1)
            if source not in function_V:
                request['source'] = source
                break
        SFC = []
        for i in range(function_num):
            SFC.append(i + 1)
        request['SFC'] = SFC
        destination = -1
        while destination == -1:
            d = random.randint(0, N - 1)
            if d not in function_V and d != source:
                destination = d
        request['destination'] = destination
        bk = random.randint(1, max_bk)
        request['bk'] = bk
        deployment_SFC = random.randint(0, len(deployment_costs) - 1)
        deployment_cost = deployment_costs[deployment_SFC]
        request['deployment_SFC'] = deployment_SFC
        request['deployment_cost'] = deployment_cost
        request['parallel_num'] = parallel_num
        request['parallel_position'] = parallel_position
        request['dk'] = random.randint(150, 200)
        request['function_num'] = function_num
        alpha = np.zeros(function_V_num)
        for i in range(function_V_num):
            alpha[i] = random.randint(5, 20) / 10
        request['alpha'] = alpha
        print('随机产生的请求为：', request)
        return request

    # 随机产生一个deployment_cost集合
    def get_deployment_cost(self, n, function_V_num, function_num, max_deployment_cost):
        deployment_costs = []
        for i in range(n):
            deployment_cost = np.zeros((function_V_num, function_num))
            for i in range(function_V_num):
                for j in range(function_num):
                    deployment_cost[i][j] = (random.randint(1, max_deployment_cost))
            deployment_costs.append(deployment_cost)

        return deployment_costs

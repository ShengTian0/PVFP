from Algorithm2 import Algorithm2
from randomGraphRequest import randomGR
from constrast1.ConstrastAlgorithm1 import ConstrastAlgorithm1
from constract2.ConstrastAlgorithm2 import ConstrastAlgorithm2

p = Algorithm2()
p2=ConstrastAlgorithm1()
p3=ConstrastAlgorithm2()
rGR = randomGR()
N = 50 # 结点数
function_V_num = 5  # 服务器结点数
pro = 0.03  # 边连接概率
max_E_cost = 10  # 最大的边权值
function_num = 4  # 网络功能数
max_bk = 17  # 最大的包大小
parallel_num = 3  # 并行功能数
parallel_position = 1  # 并行功能起始位置
SFC_num = 10  # 功能链数目
max_deployment_cost = 10  # 最大的结点功能部署代价

total_cost1=0
total_cost2=0
total_cost3=0

time1=0
time_msa=0
time_gsa=0
for i in range(50):
    requests=[]
    initial_Graph, function_V, E_constrains, function_V_constrains, link_delay = rGR.random_initial_Graph(N, function_V_num,
                                                                                                          pro, max_E_cost)
    deployment_costs = rGR.get_deployment_cost(SFC_num, function_V_num, function_num, max_deployment_cost)
    sk = rGR.random_request(N, function_num, function_V_num, max_bk, deployment_costs, parallel_num, function_V,
                            parallel_position)

    requests.append(sk)
    weight=0
    shortest_path, deploy_server, function_order, weight = p.modify_delay_request(sk, initial_Graph, function_V,
                                                                                  E_constrains, function_V_constrains,
                                                                                  link_delay)
    if weight!=0:
        total_cost1 += weight#算法1
        time1+=1
    # weight,admit_num=p2.admit_series_requests_1(initial_Graph, function_V, E_constrains, function_V_constrains, link_delay, requests)#gsa
    # total_cost2 +=weight
    # time_gsa+=admit_num


print("Algorithm1  total_cost:", total_cost1,"处理请求:", time1,"次")
print("average cost:", total_cost1*10/time1)
# print("Algorithm2 gsa total_cost:", total_cost2,"处理请求",time_gsa,"次")


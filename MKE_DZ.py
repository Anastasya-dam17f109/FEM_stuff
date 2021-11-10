import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import os
import os.path


coord_max = np.array([40,200,40], float)
nodes = {}
mesh_elems = {}
mesh_B_dict ={}
mesh_list_dict = {}
teta_grad = {}
on_s_t = {}
on_s_q = {}
on_s_s = {}
q = {}
h = int(input("Введите размер конечного элемента: "))
f_gl_std = []
lambda_ = 135
alpha_t = 10
f_h = 0
q_h = 30
teta_inf = 273.15
teta_e = 298.15
aelem = []
j_ptr = []
i_ptr = []
teta = []
ksi = 0.0001

# триангуляция заданной области

def triangulation():
    #создание узлов
    coord_buf = np.zeros(3, dtype=float)
    coord_up = np.zeros(4, dtype=int)
    coord_down = np.zeros(4, dtype=int)
    x_n = int(coord_max[0] // h)+1
    if coord_max[0] / h > x_n:
        x_n += 1
    y_n = int(coord_max[1] // h)+1
    if coord_max[1] / h > y_n:
        y_n += 1
    z_n = int(coord_max[2] // h)+1
    if coord_max[2] / h > z_n:
        z_n += 1
    counter = 0
    for i in range(y_n):
        coord_buf[2] = coord_max[2]
        for j in range(z_n):
            coord_buf[0] = 0
            for k in range(x_n):
                nodes.update({counter:coord_buf.copy()})
                counter += 1
                if k != x_n-1:
                    coord_buf[0] += h
                else:
                    coord_buf[0] = coord_max[0]
            if j != z_n - 1:
                coord_buf[2] -= h
            else:
                coord_buf[2] = 0
        if i != y_n - 1:
            coord_buf[1] += h
        else:
            coord_buf[1] = coord_max[1]
    counter_mesh = 0
    # создание конечных элементов
    for i in range(y_n-1):
        for j in range(z_n-1):
            for k in range(x_n-1):
                for l in range(2):
                    coord_up[l] = ((i + l)*x_n+j)*z_n+k
                    coord_up[3-l] = coord_up[l] + 1
                    coord_down[l] = ((i + l) * x_n + j + 1) *z_n + k
                    coord_down[3-l] = coord_down[l] + 1
                mesh_elems.update({counter_mesh    : np.array([coord_up[0],coord_up[1],coord_down[0],coord_down[3]])})
                mesh_elems.update({counter_mesh + 1: np.array([coord_up[0], coord_up[1], coord_up[3], coord_down[3]])})
                mesh_elems.update({counter_mesh + 2: np.array([coord_down[1], coord_up[1], coord_down[0], coord_down[3]])})
                mesh_elems.update({counter_mesh + 3: np.array([coord_down[1], coord_up[1], coord_down[2], coord_down[3]])})
                mesh_elems.update({counter_mesh + 4: np.array([coord_up[2], coord_up[1], coord_down[2], coord_down[3]])})
                mesh_elems.update({counter_mesh + 5: np.array([coord_up[2], coord_up[1], coord_up[3], coord_down[3]])})
                counter_mesh += 6

#возвращение списка узлов конечного элемента , которые принадлежат sigma_teta

def on_sigma_teta(idx_el):
    node_list = mesh_elems.get(idx_el)
    surf = []
    buf = 0
    for i in range(4):
        if nodes.get(node_list[i])[1]==0:
            surf.append(node_list[i])
        else:
            buf = node_list[i]
    if len(surf) ==3:
        surf.append(buf)
        return surf
    else:
        return []

#возвращение списка узлов конечного элемента , которые принадлежат sigma_q

def on_sigma_q(idx_el):
    node_list = mesh_elems.get(idx_el)
    surf = []
    buf = 0
    for i in range(4):
        if nodes.get(node_list[i])[1]==coord_max[1]:
            surf.append(node_list[i])
        else:
            buf = node_list[i]
    if len(surf) == 3:
        surf.append(buf)
        return surf
    else:
        return []

#возвращение списка узлов конечного элемента , которые принадлежат боковой поверхности

def on_sigma_side(idx_el):
    node_list = mesh_elems.get(idx_el)
    surf = []
    id_ = [0,0,2,2]
    vals_ = [0,coord_max[0],0,coord_max[2]]
    res_surf = []
    for i in range(4):
        surf = []
        for j in range(4):
            if nodes.get(node_list[j])[id_[i]]==vals_[i]:
                surf.append(node_list[j])
        if len(surf) == 3:
            res_surf.append(surf.copy())
    if len( res_surf) >0:
        return res_surf
    else:
        return []

# проверка принадлежности узла поверхности sigma_teta

def is_on_sigma_teta(idx_node):
    if nodes.get(idx_node)[1] == 0:
        return True
    else:
        return False

# вычисление барицентрических координат  узлов элемента

def baricentric(node_list):
    v_matrix = np.ones(16, dtype = float).reshape(4,4)
    res_matrix = np.zeros(16, dtype = float).reshape(4,4)
    for i in range(4):
        v_matrix[i,1:] = nodes.get(node_list[i])
    return np.linalg.inv(v_matrix.T)

#  вычисление объема конечного элемента

def calc_V(matr):
    p1 = np.array(nodes[matr[0]])
    p2 = np.array(nodes[matr[1]]) - p1
    p3 = np.array(nodes[matr[2]]) - p1
    p4 = np.array(nodes[matr[3]]) - p1
    m = np.array([p2,p3,p4])
    return abs((np.linalg.det(m)))/6

# вычисление площади поверхностного элемента

def calc_S(v1,v2,v3):
    p1 = np.array(nodes[v1])
    p2 = np.array(nodes[v2])-p1
    p3 = np.array(nodes[v3])-p1
    v_res = np.cross(p2,p3)
    return np.sqrt(np.dot(v_res,v_res))*0.5

# сборка глобальной матрицы

def create_global_system():
    gl_matr_std    = np.zeros(len(nodes)*len(nodes)).reshape(len(nodes),len(nodes))
    mesh_V_dict    = {}
    mesh_S_dict    = {}
    for i in range(len(mesh_elems)):
        node_list = []
        sigma_q_elems = on_sigma_q(i)
        S = 0
        if len(sigma_q_elems) != 0:
            node_list.append(sigma_q_elems[0])
            node_list.append(sigma_q_elems[3])
            node_list.append(sigma_q_elems[1])
            node_list.append(sigma_q_elems[2])
            S = calc_S(sigma_q_elems[0],sigma_q_elems[1],sigma_q_elems[2])
        else:
            node_list = mesh_elems.get(i).copy()
        B = baricentric(node_list.copy())
        V = calc_V(node_list.copy())
        f = (f_h*V/4)*np.array([1.0,1.0,1.0,1.0])-((q_h+alpha_t*teta_inf)*S/3)*np.array([1.0,0.0,1.0,1.0])
        mesh_B_dict.update({i: B})
        mesh_list_dict.update({i: node_list})
        mesh_V_dict.update({i: V})
        mesh_S_dict.update({i: S})
        for j in range(4):
            f_gl_std[node_list[j]] += f[j]

    for i in range(len(mesh_elems)):
        if len(on_sigma_teta(i)) != 0:
            f = np.array([teta_e, teta_e, teta_e])
            node_list = on_sigma_teta(i)
            for j in range(3):
                i_gl = node_list[j]
                gl_matr_std[i_gl,i_gl] = 1
                f_gl_std[i_gl] = f[j]

    for i in range(len(mesh_elems)):
        B = mesh_B_dict.get(i).copy().T
        B_t = mesh_B_dict.get(i)[:, 1:]
        B = B[1:, :]
        G = lambda_ * mesh_V_dict.get(i) * np.dot(B_t, B) - alpha_t * (mesh_S_dict.get(i) / 12) * np.array(
            [[2.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 2.0, 1.0], [1.0, 0.0, 1.0, 2.0]])

        for j in range(4):
            i_gl = mesh_list_dict.get(i)[j]
            if not is_on_sigma_teta(i_gl):
                for k in range(4):
                    j_gl = mesh_list_dict.get(i)[k]
                    if not is_on_sigma_teta(j_gl):
                        gl_matr_std[i_gl, j_gl] += G[j, k]
                    else:
                        f_gl_std[i_gl] -= G[j, k]*teta_e
    #print( f_gl_std)

   # print(mesh_S_dict)
    i_ptr.append(0)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if gl_matr_std[i,j] != 0:
                aelem.append(gl_matr_std[i,j])
                j_ptr.append(j)
        i_ptr.append(len(aelem))

# выполнение умножения в формате csr

def multiply_csr(x):
    n = len(x)
    z = np.zeros(n)
    for i in range(n):
        for j in range(i_ptr[i],i_ptr[i+1]):
            z[i] += x[j_ptr[j]]*aelem[j]
    return z

# решение глобальной слау на основе метода сопряженных градиентов

def solve_system():
    n       = len(f_gl_std)
    b       = np.dot(f_gl_std,f_gl_std)
    x       = f_gl_std.copy()
    r_i     = f_gl_std-multiply_csr(x)
    d_ri    = 0
    z       = r_i.copy()
    alpha_k = 0
    beta_k  = 0
    flag    = ksi + 1
    step    = 0
    while(flag > ksi):
        buf = multiply_csr(z).copy()
        step += 1
        alpha_k = np.dot(r_i, r_i)/np.dot(buf,z.copy())
        x = x + alpha_k*z.copy()
        d_ri = np.dot(r_i.copy(), r_i.copy())

        r_i = r_i-alpha_k*buf
        beta_k = np.dot(r_i.copy(), r_i.copy())/d_ri
        z = r_i + beta_k*z
        flag = np.sqrt(np.dot(r_i.copy(), r_i.copy())/b)
    print("Решение найдено")
    return x

# вычисление аппроксимации q и grad_teta

def find_approximation():
    for i in range(len(mesh_list_dict)):
        node_list = mesh_list_dict.get(i)
        teta_list = np.array([[0.0],[0.0],[0.0],[0.0]])
        for j in range(4):
            teta_list[j,0] = teta[node_list[j]]
        buf_teta = np.dot(mesh_B_dict.get(i).T[1:, :], teta_list)
        buf_teta = [buf_teta[0,0],buf_teta[1,0],buf_teta[2,0]]
        for j in range(4):
            if teta_grad.get(node_list[j])is not None:
                p = teta_grad.get(node_list[j])
                p.append(buf_teta)
                teta_grad.update({node_list[j]: p})
            else:
                teta_grad.update({node_list[j]: [buf_teta]})
        if len(on_sigma_teta(i)) !=0:
            on_s_t.update({len(on_s_t):on_sigma_teta(i)[:3]})
        if len(on_sigma_q(i)) != 0:
            on_s_q.update({len(on_s_q): on_sigma_q(i)[:3]})
        if len(on_sigma_side(i)) != 0:
            for j in range(len(on_sigma_side(i))):
                on_s_s.update({len(on_s_s): on_sigma_side(i)[j]})
    for i in range(len(nodes)):
         length = len(teta_grad.get(i))
         buf_teta1 = np.zeros(3)
         for j in range(length):
             for k in range(3):
                buf_teta1[k] += teta_grad.get(i)[j][k]/length
         teta_grad.update({i: buf_teta1.copy()})
         q.update({i: buf_teta1.copy()*lambda_})

# вывод результата в файл формата mv2

def print_in_mv2():
    with open("D://result_mke.txt",'w') as file:
        file.write(str(len(nodes))+' 3 3 teta grad_t_x grad_t_y grad_t_z q_x q_y q_z \n')
        for i in range(len(nodes)):
            str_buf1 = ''
            str_buf3 = ''
            str_buf4 = ''
            for j in range(3):
                str_buf1 += str(nodes.get(i)[j])+' '
                str_buf3 += str(teta_grad.get(i)[j]) + ' '
                str_buf4 += str(q.get(i)[j]) + ' '
            file.write( str(i+1) + ' ' + str_buf1 + str(teta[i]) + ' ' + str_buf3 + str_buf4 + '\n')

        file.write(str(len(on_s_s)+ len(on_s_q)+ len(on_s_t)) + ' 3 3 BC_id mat_id mat_id_Out\n')
        counter = 1
        for i in range(len(on_s_t)):
            str_buf1 = ''
            for j in range(3):
                str_buf1 += str(on_s_t.get(i)[j]+1) + ' '
            file.write(str(counter) + ' ' + str_buf1 + ' 0 1 0\n')
            counter += 1
        for i in range(len(on_s_q)):
            str_buf1 = ''
            for j in range(3):
                str_buf1 += str(on_s_q.get(i)[j]+1) + ' '
            file.write(str(counter) + ' ' + str_buf1 + ' 0 1 0\n')
            counter += 1
        for i in range(len(on_s_s)):
            str_buf1 = ''
            for j in range(3):
                str_buf1 += str(on_s_s.get(i)[j]+1) + ' '
            file.write(str(counter) + ' ' + str_buf1 + ' 0 1 0\n')
            counter += 1
    file.close()
    if os.path.exists("D://result_mke.mv2"):
        os.remove("D://result_mke.mv2")
    os.rename("D://result_mke.txt","D://result_mke.mv2")


# решение поставленной задачи

triangulation()
f_gl_std = np.zeros(len(nodes))
create_global_system()

aelem = np.array(aelem)
i_ptr = np.array(i_ptr)
j_ptr = np.array(j_ptr)

teta = solve_system()
find_approximation()
print_in_mv2()





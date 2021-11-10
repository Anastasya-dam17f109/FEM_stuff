import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import os
import os.path


coord_min = np.array([300,0], float)
coord_max = np.array([320, 200], float)
nodes = {}
mesh_elems = {}
mesh_B_dict ={}
mesh_Bk_dict ={}
mesh_list_dict = {}
E_ = 212
nu_ = 0.29
mu  = 0
l_1 = E_*nu_/((1+nu_)*(1-2*nu_))
l_2 = E_/(2*(1+nu_))
C_ =  np.array([[l_1+2*l_2, l_1,l_1, 0],[ l_1,l_1+2*l_2,l_1, 0],[ l_1,l_1, l_1+2*l_2,0],[0,0,0,l_2]], float)

f_h = np.array([0,0], float)
u_e = 0
te_1 = -0.0001*np.array([1,0],float)
te_2 = 0.0001*np.array([1,0],float)
on_s_sigma = {}
on_s_u = {}
u_vec = []
epsilon = {}
sigma = {}
q = {}
h = float(input("Введите размер конечного элемента по r : "))
f_gl_std = []
gl_matr_std  = []


aelem = []
j_ptr = []
i_ptr = []

ksi = 0.000001

# триангуляция заданной области

def triangulation():
    #создание узлов
    coord_buf = np.zeros(2, dtype=float)
    coord_up = np.zeros(2, dtype=int)
    coord_down = np.zeros(2, dtype=int)
    h_z = int(h)
    r_n = int((coord_max[0]-coord_min[0])// h)+1
    if (coord_max[0]-coord_min[0]) / h > r_n:
        r_n += 1
    z_n = int((coord_max[1]-coord_min[1]) // h_z)+1
    if (coord_max[1]-coord_min[1]) / h_z > z_n:
        z_n += 1
    counter = 0
    coord_buf[1] = coord_min[1]
    for i in range(z_n):
        coord_buf[0] = coord_max[0]
        for j in range(r_n):
            nodes.update({counter : coord_buf.copy()})
            counter += 1
            if j != r_n-1:
                coord_buf[0] -= h
            else:
                coord_buf[0] = coord_min[1]
        if i != z_n - 1:
            coord_buf[1] += h_z
        else:
            coord_buf[1] = coord_max[1]
    counter_mesh = 0
    # создание конечных элементов
    for i in range(z_n-1):
        for k in range(r_n-1):
            for l in range(2):
                coord_up[l] = (i + l)*r_n+k
                coord_down[l] = (i + l)*r_n + k + 1
            mesh_elems.update({counter_mesh    : np.array([coord_up[0],coord_up[1],coord_down[0]])})
            mesh_elems.update({counter_mesh + 1: np.array([coord_up[1], coord_down[0], coord_down[1]])})
            counter_mesh += 2

#возвращение списка узлов конечного элемента , которые принадлежат sigma_u

def on_sigma_u(idx_el):
    node_list = mesh_elems.get(idx_el)
    curve = []
    buf = 0
    for i in range(3):
        if nodes.get(node_list[i])[1] == coord_min[1] or nodes.get(node_list[i])[1] == coord_max[1]:
            curve.append(node_list[i])
        else:
            buf = node_list[i]
    if len(curve) == 2:
        curve.append(buf)
        return curve
    else:
        return []

#возвращение списка узлов конечного элемента , которые принадлежат sigma_sigma

def on_sigma_sigma(idx_el):
    node_list = mesh_elems.get(idx_el)
    curve = []
    buf = 0
    for i in range(3):
        if nodes.get(node_list[i])[0] == coord_min[0] or nodes.get(node_list[i])[0] == coord_max[0]:
            curve.append(node_list[i])
        else:
            buf = node_list[i]
    if len(curve) == 2:
        curve.append(buf)
        return curve
    else:
        return []


# проверка принадлежности узла поверхности sigma_u

def is_on_sigma_u(idx_node,id):
    if nodes.get(idx_node)[1] == coord_min[1] or nodes.get(idx_node)[1] == coord_max[1]:
        if id % 2 != 0:
            return True
        else:
            return False
    else:
        return False

# вычисление барицентрических координат  узлов элемента

def baricentric(node_list):
    v_matrix = np.ones(9, dtype = float).reshape(3,3)
    res_matrix = np.zeros(9, dtype = float).reshape(3,3)
    for i in range(3):
        v_matrix[i,1:] = nodes.get(node_list[i])
    return np.linalg.inv(v_matrix.T)


# вычисление площади конечного элемента

def calc_S(v1,v2,v3):
    p1 = np.array(nodes[v1])
    p2 = np.array(nodes[v2])-p1
    p3 = np.array(nodes[v3])-p1
    v_res = np.cross(p2,p3)
    return np.sqrt(np.dot(v_res,v_res))*0.5

# вычисление длины стороны конечного элемента

def calc_L(v1,v2):
    v_res = np.array(nodes[v1]) - np.array(nodes[v2])
    return np.sqrt(np.dot(v_res,v_res))

# сборка глобальной матрицы

def create_global_system():
    global gl_matr_std
    mesh_L_dict    = {}
    mesh_S_dict    = {}
    mesh_ri_dict   = {}
    for i in range(len(mesh_elems)):
        node_list = []
        sigma_s_elems = on_sigma_sigma(i)
        S = 0
        L = 0
        if len(sigma_s_elems) != 0:
            node_list.append(sigma_s_elems[2])
            node_list.append(sigma_s_elems[0])
            node_list.append(sigma_s_elems[1])
            L = calc_L(sigma_s_elems[0],sigma_s_elems[1])
        else:
            node_list = mesh_elems.get(i).copy()
        B = baricentric(node_list.copy()).T
        S = calc_S(node_list[0],node_list[1],node_list[2])
        # ri
        a_i = np.array([[B[0,0]],[B[0,1]]])
        b_matr = B[1:,:2].copy().T
        b_matr = np.linalg.inv(b_matr)
        a_i =  np.dot(b_matr,a_i)
        r_i = np.array([0,b_matr[0,1]-a_i[0],-a_i[0]],float)
        buf_v1 = np.zeros(6)
        for j in range(6):
            buf_v1[j] += r_i[0]+r_i[1]+r_i[2]
        for j in range(3):
            for k in range(2):
                buf_v1[j*2+k] += r_i[j]
        buf_v2 = buf_v1.copy()
        for j in range(3):
            for k in range(2):
                buf_v1[j*2+k] = buf_v1[j*2+k]* f_h[k]
                if(j == 0):
                    buf_v2[j * 2 + k] = 0
                else:
                    if(L != 0):
                        if nodes. get(node_list[j])[0] == coord_max[0]:
                            buf_v2[j * 2 + k] = buf_v2[j*2+k]*te_1[k]

                        else:
                            buf_v2[j * 2 + k] = buf_v2[j * 2 + k] * te_2[k]
        print((math.pi*L/3)*buf_v2,node_list)
        f = -(S/12)*buf_v1+(math.pi*L/3)*buf_v2
        mesh_B_dict.update({i: B})
        mesh_ri_dict.update({i: r_i})
        mesh_list_dict.update({i: node_list})
        mesh_L_dict.update({i: L})
        mesh_S_dict.update({i: S})
        for j in range(3):
            f_gl_std[node_list[j]*2] += f[2*j]
            f_gl_std[2*node_list[j]+1] += f[2 * j+1]
    print(f_gl_std)
    for i in range(len(mesh_elems)):
        if len(on_sigma_u(i)) != 0:
            f = np.array([u_e, u_e,u_e, u_e])
            node_list = on_sigma_u(i)
            for j in range(2):
                i_gl = node_list[j]
                #gl_matr_std[2*i_gl,2*i_gl] = 1
                gl_matr_std[2 * i_gl+1, 2 * i_gl+1] = 1
               # f_gl_std[2 *i_gl] = f[2 *j]
                f_gl_std[2 * i_gl + 1] = f[2 * j + 1]
    #print( f_gl_std)
    for i in range(2 * len(nodes)):
        if abs(f_gl_std[i])<10**(-7):
            f_gl_std[i] = 0
    print(f_gl_std)
    for i in range(len(mesh_elems)):
        B = mesh_B_dict.get(i).copy()
        r_w = np.zeros(2)
        phi = np.zeros(3)
        for j in range(3):
            r_w[0] += nodes.get(mesh_list_dict.get(i)[j])[0]/3
            r_w[1] += nodes.get(mesh_list_dict.get(i)[j])[1]/3
        for j in range(3):
            phi[j] = (B[0,j]+B[1,j]*r_w[0] +B[2,j]*r_w[1])/r_w[0]

        B_k = np.zeros(24).reshape(4,6)
        for j in range(3):
            B_k[0, j*2] = B[1,j]
            B_k[1, j * 2] = phi[j]
            B_k[3, j * 2+1] = B[1, j]
            B_k[3, j * 2] = B[2, j]
            B_k[2, j * 2 + 1] = B[2, j]
        B_t = B_k.T
        mesh_Bk_dict.update({i :B_k})
        L_buf =  np.zeros(36).reshape(6,6)
        for j in range(2,6):
            L_buf[j, j] = 3 * mesh_ri_dict.get(i)[1] + mesh_ri_dict.get(i)[2]

        L_buf[2, 4] = mesh_ri_dict.get(i)[1]+mesh_ri_dict.get(i)[2]
        L_buf[3, 5] = mesh_ri_dict.get(i)[1]+mesh_ri_dict.get(i)[2]
        L_buf[4, 2] = mesh_ri_dict.get(i)[1]+mesh_ri_dict.get(i)[2]
        L_buf[5, 3] = mesh_ri_dict.get(i)[1]+mesh_ri_dict.get(i)[2]
        G =  2*math.pi*r_w[0]* mesh_S_dict.get(i) *np.dot(np.dot(B_t, C_),B_k) + mu*math.pi * (mesh_L_dict.get(i) / 6) * L_buf
        node_list = []
        for j in range(3):
            node_list.append( mesh_list_dict.get(i)[j])
            node_list.append(mesh_list_dict.get(i)[j])
        for j in range(6):
            i_gl = node_list[j]
            if not is_on_sigma_u(i_gl,j):
                i_gl = 2*i_gl
                if j%2 !=0:
                    i_gl += 1
                for k in range(6):
                    j_gl = node_list[k]
                    if not is_on_sigma_u(j_gl,k):
                        j_gl = 2*j_gl
                        if k % 2 != 0:
                            j_gl += 1
                        gl_matr_std[i_gl, j_gl] += G[j, k]
                    else:
                        j_gl = 2 * j_gl
                        if k % 2 != 0:
                            j_gl += 1
                        f_gl_std[i_gl] -= G[j, k]*f_gl_std[j_gl].copy()

    i_ptr.append(0)
    for i in range(2*len(nodes)):
        for j in range(2*len(nodes)):
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
        u_list = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
        for j in range(3):
            u_list[2*j,0] = u_vec[2*node_list[j]]

            u_list[2 * j+1, 0] = u_vec[2 * node_list[j]+1]


        buf_eps = np.dot(mesh_Bk_dict.get(i), u_list)
        buf_sigma = np.dot(C_, buf_eps.copy())
        buf_eps = [buf_eps[0,0],buf_eps[1,0],buf_eps[2,0],buf_eps[3,0]]
        buf_sigma =[buf_sigma[0,0],buf_sigma[1,0],buf_sigma[2,0],buf_sigma[3,0]]
        for j in range(3):
            if epsilon.get(node_list[j])is not None:
                p = epsilon.get(node_list[j])
                p.append(buf_eps)
                epsilon.update({node_list[j]: p})
            else:
                epsilon.update({node_list[j]: [buf_eps]})
            if sigma.get(node_list[j])is not None:
                p = sigma.get(node_list[j])
                p.append(buf_sigma)
                sigma.update({node_list[j]: p})
            else:
                sigma.update({node_list[j]: [buf_sigma]})

        if len(on_sigma_sigma(i)) !=0:
            on_s_sigma.update({len(on_s_sigma):on_sigma_sigma(i)[:2]})
        if len(on_sigma_u(i)) != 0:
            on_s_u.update({len(on_s_u): on_sigma_u(i)[:2]})

    for i in range(len(nodes)):
         length1 = len(epsilon.get(i))
         length2 = len(sigma.get(i))
         buf_1 = np.zeros(4)
         buf_2 = np.zeros(4)
         for j in range(length1):
             for k in range(4):
                buf_1[k] += epsilon.get(i)[j][k]/length1
                buf_2[k] += sigma.get(i)[j][k] / length2
         epsilon.update({i: buf_1.copy()})
         sigma.update({i: buf_2.copy()})


# вывод результата в файл формата mv2

def print_in_mv2():
    with open("D://result_mke.txt",'w') as file:
        file.write(str(len(nodes))+' 3 10 u1 u2 eps11 eps22 eps33 eps12  sigma11 sigma22 sigma33 sigma12\n')
        for i in range(len(nodes)):
            str_buf1 = ''
            str_buf2 = ''
            str_buf3 = ''
            str_buf4 = ''
            for j in range(2):
                str_buf1 += str(nodes.get(i)[j])+' '
                #str_buf2 +=  str(u_vec[2*i+j]) + ' '
                str_buf3 += str(epsilon.get(i)[2*j]) + ' '+str(epsilon.get(i)[2*j+1]) + ' '
                str_buf4 += str(sigma.get(i)[2*j]) + ' ' + str(sigma.get(i)[2*j+1]) + ' '
            str_buf2 = '{0:.8f} {1:.8f} '.format(u_vec[2 * i + 0], u_vec[2 * i + 1])
            str_buf1 +=  '0 '
            file.write( str(i+1) + ' ' + str_buf1  + str_buf2 + str_buf3 + str_buf4 + '\n')
        file.write(str(len(mesh_elems)) + ' 3 3 BC_id mat_id mat_id_Out\n')
        for i in range(len(mesh_elems)):
            str_buf1 = ''
            for j in range(3):
                str_buf1 += str(mesh_list_dict.get(i)[j]+1)+' '
            file.write(str(i+1) + ' ' + str_buf1 + '0 1 0\n')


    file.close()
    if os.path.exists("D://result_mke.mv2"):
        os.remove("D://result_mke.mv2")
    os.rename("D://result_mke.txt","D://result_mke.mv2")


# решение поставленной задачи

triangulation()

f_gl_std = np.zeros(2*len(nodes))
gl_matr_std    = np.zeros(2*len(nodes)*2*len(nodes)).reshape(2*len(nodes),2*len(nodes))
create_global_system()

aelem = np.array(aelem)
i_ptr = np.array(i_ptr)
j_ptr = np.array(j_ptr)

#u_vec = solve_system()
u_vec =  np.linalg.solve(gl_matr_std , f_gl_std)
#print(u_vec)
find_approximation()
print_in_mv2()





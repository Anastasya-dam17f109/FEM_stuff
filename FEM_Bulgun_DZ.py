import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import os
import os.path


coord_min = np.array([0,0], float)
coord_max = np.array([200, 100], float)
nodes = {}
mesh_elems = {}
mesh_B_dict ={}
mesh_Bk_dict ={}
mesh_list_dict = {}

E_ = 212
nu_ = 0.29
mu  = 0.001
#mu = 0
l_1 = E_*nu_/((1+nu_)*(1-2*nu_))
l_2 = E_/(2.0*(1+nu_))
C_ =  np.array([[l_1+2*l_2, l_1,0],[ l_1,l_1+2*l_2,0],[0,0,l_2]], float)
print(1/(l_1+2*l_2))
f_h = np.array([0,0], float)
u_e = 0.0
te_1 = -0.0001*np.array([1,0],float)
#te_2 = 0.1*np.array([0,1],float)
on_s_sigma = {}
on_s_u = {}
u_vec = []
epsilon = {}
sigma = {}
q = {}
h = float(input("Введите размер конечного элемента: "))
gl_mar_std = []
f_gl_std = []



aelem = []
j_ptr = []
i_ptr = []

ksi = 0.0000001

# триангуляция заданной области

def triangulation():
    #создание узлов
    coord_buf = np.zeros(2, dtype=float)
    coord_up = np.zeros(2, dtype=int)
    coord_down = np.zeros(2, dtype=int)
    x_n = int((coord_max[0]-coord_min[0])// h)+1
    if (coord_max[0]-coord_min[0]) / h >x_n:
        x_n += 1
    y_n = int((coord_max[1]-coord_min[1]) // h)+1
    if (coord_max[1]-coord_min[1]) / h > x_n:
        y_n += 1
    counter = 0
    coord_buf[1] = coord_min[1]
    for i in range(x_n):
        coord_buf[1] = coord_max[1]
        for j in range(y_n):
            nodes.update({counter : coord_buf.copy()})
            counter += 1
            if j != y_n-1:
                coord_buf[1] -= h
            else:
                coord_buf[1] = coord_min[1]
        if i != x_n - 1:
            coord_buf[0] += h
        else:
            coord_buf[0] = coord_max[0]
    counter_mesh = 0
    # создание конечных элементов
    for i in range(x_n-1):
        for k in range(y_n-1):
            for l in range(2):
                coord_up[l] = (i + l)*y_n+k
                coord_down[l] = (i + l)*y_n + k + 1
            mesh_elems.update({counter_mesh    : np.array([coord_up[0],coord_up[1],coord_down[0]])})
            mesh_elems.update({counter_mesh + 1: np.array([coord_up[1], coord_down[0], coord_down[1]])})
            counter_mesh += 2

#возвращение списка узлов конечного элемента , которые принадлежат sigma_u

def on_sigma_u(idx_el):
    node_list = mesh_elems.get(idx_el)
    curve = []
    curve1 = []
    result = []
    fl = False
    fl1 = False
    buf = 0
    for i in range(3):
        if nodes.get(node_list[i])[1] == coord_min[1] or nodes.get(node_list[i])[1] == coord_max[1]:
            curve.append(node_list[i])
        else:
            buf = node_list[i]
    if len(curve) == 2:
        curve.append(buf)
        result.append(curve)
    for i in range(3):
        if nodes.get(node_list[i])[0] == coord_min[0] :
            curve1.append(node_list[i])
        else:
            buf = node_list[i]
    if len(curve1) == 2:
        curve1.append(buf)
        result.append(curve1)

    return result

#возвращение списка узлов конечного элемента , которые принадлежат sigma_sigma

def on_sigma_sigma(idx_el):
    node_list = mesh_elems.get(idx_el)
    curve = []
    buf = 0
    for i in range(3):
        if  nodes.get(node_list[i])[0] == coord_max[0]:
            curve.append(node_list[i])
        else:
            buf = node_list[i]
    if len(curve) == 2:
        curve.append(buf)
        return curve
    else:
        return []


# проверка принадлежности узла поверхности sigma_u

def is_on_sigma_u(idx_node):
    if nodes.get(idx_node)[1] == coord_min[1] or nodes.get(idx_node)[1] == coord_max[1] or nodes.get(idx_node)[0] == coord_min[0]:
        return True
    else:
        return False

def is_on_sigma_u_coord(idx_node,coord):
    if nodes.get(idx_node)[1] == coord_min[1] or nodes.get(idx_node)[1] == coord_max[1] :
        if coord % 2 !=0:
            return True
        else:
            if nodes.get(idx_node)[0] == coord_min[0]:
                if coord % 2 == 0:
                    return True
                else:
                    return False
            return False
    else:
        if  nodes.get(idx_node)[0] == coord_min[0]:
            if coord % 2 == 0:
                return True
            else:
                return False
        else:
            return False



def is_on_sigma_u_coord_1(idx_node,coord):

    if coord % 2 == 0:
        if nodes.get(idx_node)[0] == coord_min[0]:
            return True
        else:
            return False
    else:
        if nodes.get(idx_node)[1] == coord_min[1] or nodes.get(idx_node)[1] == coord_max[1] :
            return True
        else:
            return False

# вычисление барицентрических координат  узлов элемента

def baricentric(node_list):
    v_matrix = np.ones(9, dtype = float).reshape(3,3)
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

    global gl_mar_std
    mesh_L_dict    = {}
    mesh_S_dict    = {}
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
        buf_v1 = np.zeros(6)
        buf_v2 = np.zeros(6)
        for j in range(3):
            for k in range(2):
                buf_v1[j*2+k] =  f_h[k]/3
                if(j == 0):
                    buf_v1[j * 2 + k] = 0
                    buf_v2[j * 2 + k] = 0
                else:
                    if(L != 0):
                        buf_v2[j * 2 + k] = te_1[k]/2.0
        f = -S*buf_v1+(L)*buf_v2
        mesh_B_dict.update({i: B})
        mesh_list_dict.update({i: node_list})
        mesh_L_dict.update({i: L})
        mesh_S_dict.update({i: S})
        for j in range(3):
            f_gl_std[node_list[j]*2] += f[2*j]
            f_gl_std[2*node_list[j]+1] += f[2 * j+1]
    print( f_gl_std)
    for i in range(len(mesh_elems)):
        if len(on_sigma_u(i)) != 0:
            f = np.array([u_e, u_e, u_e, u_e])
            for l in range(len(on_sigma_u(i))):
                node_list = on_sigma_u(i)[l]
                for j in range(2):
                    i_gl = node_list[j]
                    if nodes.get(i_gl)[1] == coord_min[1] or nodes.get(i_gl)[1] == coord_max[1]:
                        gl_mar_std[2*i_gl+1,2*i_gl+1] = 1
                        f_gl_std[2 * i_gl+1] = f[2 * j+1]

                    if nodes.get(i_gl)[0] == coord_min[0]:
                        gl_mar_std[2 * i_gl, 2 * i_gl] = 1
                        f_gl_std[2 * i_gl] = f[2 * j]


    for i in range(len(mesh_elems)):
        B = mesh_B_dict.get(i).copy()
        B_k = np.zeros(18).reshape(3,6)
        #print("hhhhh")
        #print(B)
        for j in range(3):
            #if (abs(B[1,j])>10**(-6)):
            B_k[0, j*2] = B[1,j]
            B_k[2, j * 2 + 1] = B[1, j]
            #if (abs(B[2,j])>10**(-6)):
            B_k[1, j * 2+1] = B[2,j]

            B_k[2, j * 2] = B[2,j]
        #print(B_k)
        B_t = B_k.copy().T
        mesh_Bk_dict.update({i :B_k})
        L_buf =  np.zeros(36).reshape(6,6)
        for j in range(2,6):
            L_buf[j, j] = 2
        L_buf[2, 4] = 1
        L_buf[3, 5] = 1
        L_buf[4, 2] = 1
        L_buf[5, 3] = 1
        G =  mesh_S_dict.get(i) *np.dot(np.dot(B_t, C_),B_k) + mu * (mesh_L_dict.get(i) /6) * L_buf
        print(mu * (mesh_L_dict.get(i) /6) * L_buf)
        print( mesh_S_dict.get(i) *np.dot(np.dot(B_t, C_),B_k))
        node_list = []
        for j in range(3):
            node_list.append( mesh_list_dict.get(i)[j])
            node_list.append(mesh_list_dict.get(i)[j])
        for j in range(6):
            i_gl = node_list[j]
            if not is_on_sigma_u_coord_1(i_gl, j):
                i_gl = 2 * i_gl
                if j % 2 != 0:
                    i_gl += 1
                for k in range(6):
                    j_gl = node_list[k]
                    if not is_on_sigma_u_coord_1(j_gl, k):
                        print("vvvvvv")
                        print(node_list[j])
                        print(node_list[k])
                        j_gl = 2 * j_gl
                        if k % 2 != 0:
                            j_gl += 1

                        gl_mar_std[i_gl, j_gl] += G[j, k]
                    else:
                        j_gl = 2 * j_gl
                        if k % 2 != 0:
                            j_gl += 1
                        f_gl_std[i_gl] -= G[j, k] * f_gl_std[j_gl].copy()

    #gl_mar_std =  gl_mar_std.T
    with open("D://result_mke1.txt", 'w') as file:

        for i in range(2*len(nodes)):
            str_1 = ""
            for j in range(2*len(nodes)):
                str_1 += str(gl_mar_std[i,j])+" "
            str_1 += "\n"
            file.write(str_1)
    file.close()
    '''print(gl_matr_std)
    print(f_gl_std)'''
    i_ptr.append(0)
    for i in range(2*len(nodes)):
        for j in range(2*len(nodes)):
            if gl_mar_std[i,j] != 0:
                aelem.append(gl_mar_std[i,j])
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

# вычисление аппроксимации epsilon и sigma

def find_approximation():
    for i in range(len(mesh_list_dict)):
        node_list = mesh_list_dict.get(i)
        u_list = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
        for j in range(3):
            u_list[2*j,0] = u_vec[2*node_list[j]]
            u_list[2 * j+1, 0] = u_vec[2 * node_list[j]+1]

        buf_eps = np.dot(mesh_Bk_dict.get(i), u_list)
        buf_sigma = np.dot(C_, buf_eps.copy())
        buf_eps = [buf_eps[0,0],buf_eps[1,0],buf_eps[2,0]]
        buf_sigma =[buf_sigma[0,0],buf_sigma[1,0],buf_sigma[2,0]]
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

    for i in range(len(nodes)):
         length1 = len(epsilon.get(i))
         length2 = len(sigma.get(i))
         buf_1 = np.zeros(3)
         buf_2 = np.zeros(3)
         for j in range(length1):
             for k in range(3):
                buf_1[k] += epsilon.get(i)[j][k]/length1
                buf_2[k] += sigma.get(i)[j][k] / length2
         epsilon.update({i: buf_1.copy()})
         sigma.update({i: buf_2.copy()})


# вывод результата в файл формата mv2

def print_in_mv2():
    with open("D://result_mke.txt",'w') as file:
        file.write(str(len(nodes))+' 3 8 u1 u2 eps11 eps22 eps33 sigma11 sigma22 sigma33 \n')
        for i in range(len(nodes)):
            str_buf1 = ''
            str_buf2 = ''
            str_buf3 = ''
            str_buf4 = ''
            for j in range(2):
                str_buf1 += str(nodes.get(i)[j])+' '
                #str_buf2 +=  str(u_vec[2*i+j]) + ' '

            str_buf2 ='{0:.16f} {1:.16f} '.format(u_vec[2*i+0],u_vec[2*i+1])
            for j in range(3):
                str_buf3 += str(epsilon.get(i)[j]) + ' '

                str_buf4 += str(sigma.get(i)[j]) + ' '
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
gl_mar_std   = np.zeros(2*len(nodes)*2*len(nodes)).reshape(2*len(nodes),2*len(nodes))

create_global_system()

aelem = np.array(aelem)
i_ptr = np.array(i_ptr)
j_ptr = np.array(j_ptr)

#u_vec = solve_system()
u_vec =  np.linalg.solve(gl_mar_std , f_gl_std)
#print(u_vec)
find_approximation()
print_in_mv2()

''' 
for j in range(6):
    i_gl = node_list[j]
    if not is_on_sigma_u_coord(i_gl,j):
        i_gl = 2*i_gl
        if j%2 !=0:
            i_gl += 1
        for k in range(6):
            j_gl = node_list[k]
            if not is_on_sigma_u_coord(j_gl,k):
                j_gl = 2*j_gl
                if k % 2 != 0:
                    j_gl += 1
                gl_matr_std[i_gl, j_gl] += G[j, k]
            else:
                j_gl = 2 * j_gl
                if k % 2 != 0:
                    j_gl += 1
                f_gl_std[i_gl] -= G[j, k]*f_gl_std[j_gl].copy()
'''

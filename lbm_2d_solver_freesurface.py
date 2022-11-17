import taichi as ti
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu, dynamic_index=True, kernel_profiler=False, print_ir=False)

nx = 512
ny = 512

# 2D velocity
v = ti.Vector.field(2, ti.f32, shape=(nx, ny))
# density
rho = ti.field(ti.f32, shape=(nx, ny))
# double buffered mass, can be swapped quickly.
mass = ti.field(ti.f32, shape=(2, nx, ny))
# ϵ
volume_fraction = ti.field(ti.f32, shape=(nx, ny))

# grid_type: 0 for fluid, 1 for interface, 2 for gas, 3 for solid(not imple yet).
type_mask = ti.field(ti.i8, shape=(nx, ny))

display_var = ti.field(ti.f32, shape=(nx, ny))

# double buffered f_x, can be swapped quickly.
f_x = ti.field(ti.f32, shape=(2, nx, ny, 9))

eq_v_weight = ti.field(ti.f32, shape=9)
lattice_vector = ti.Vector.field(2, ti.f32, shape=9)
reverse_direction = ti.field(ti.i8, shape=9)

# Viscosity define
niu = 0.02
# 由流体粘度 计算流体松弛时间tau
tau = 3.0 * niu + 0.5
inv_tau = 1.0 / tau
rho_gas = 0.01

np_arr = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
                   1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float32)
eq_v_weight.from_numpy(np_arr)

np_arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                   [-1, 1], [-1, -1], [1, -1]], dtype=np.float)
lattice_vector.from_numpy(np_arr)

np_arr = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
reverse_direction.from_numpy(np_arr)

steps = 1000000

# Init rho and velocity
cycle_position = ti.Vector([128, 256])


@ti.kernel
def init():
    for i, j in rho:
        v[i, j] = ti.Vector([0.0, 0.0])
        rho[i, j] = 1.0
        type_mask[i, j] = 0
        # vector = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)])
        # vector = vector - cycle_position
        # d = ti.math.dot(vector, vector)
        # if d <= 15.0:
        #     solid_mask[i, j] = 1


@ti.func
def f_eq(i, j, direction):
    u_dot_c = ti.Vector.dot(lattice_vector[direction], v[i, j])
    u_dot_u = ti.Vector.dot(v[i, j], v[i, j])
    return eq_v_weight[direction] * rho[i, j] * (1.0 + 3.0 * u_dot_c + 4.5 * u_dot_c ** 2 - 1.5 * u_dot_u)


@ti.func
def f_eq_gas(rho_value, v_value, direction):
    u_dot_c = ti.Vector.dot(lattice_vector[direction], v_value)
    u_dot_u = ti.Vector.dot(v_value, v_value)
    return eq_v_weight[direction] * rho_value * (1.0 + 3.0 * u_dot_c + 4.5 * u_dot_c ** 2 - 1.5 * u_dot_u)


fx_bank_sel = ti.field(ti.i32, shape=())
fx_bank_sel[None] = 0


@ti.kernel
def collision_stream():
    next_bank = fx_bank_sel[None] + 1
    next_bank = next_bank % 2
    # 不在最外一层进行stream与collision，便于边界处理
    for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
        if type_mask[i, j] != 2:
            for direction in ti.static(range(9)):
                i_prev = i - ti.cast(lattice_vector[direction][0], ti.i32)
                j_prev = j - ti.cast(lattice_vector[direction][1], ti.i32)
                # dt = 1
                if type_mask[i_prev, j_prev] == 2:
                    f_x[next_bank, i, j, direction] = f_eq_gas(rho_gas, v[i, j], direction) + f_eq(i_prev, j_prev,
                                                                                                   reverse_direction[
                                                                                                       direction]) - \
                                                      f_x[fx_bank_sel[None], i, j, direction]
                else:
                    f_x[next_bank, i, j, direction] = (1 - inv_tau) * f_x[
                        fx_bank_sel[None], i_prev, j_prev, direction] + inv_tau * f_eq(i_prev, j_prev, direction)
    fx_bank_sel[None] = next_bank


mass_bank_sel = ti.field(ti.i32, shape=())
mass_bank_sel[None] = 0


@ti.kernel
def mass_stream():
    # 不在最外一层进行stream与collision，便于边界处理
    fx_bank = fx_bank_sel[None]
    mass_bank = mass_bank_sel[None]
    mass_bank_next = (mass_bank + 1) % 2
    for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
        if type_mask[i, j] == 2 or type_mask[i, j] == 3:
            continue
        dm = 0.
        i_next_f = i + lattice_vector[direction][0]
        i_next = ti.cast(i_next_f, ti.i32)
        j_next_f = j + lattice_vector[direction][1]
        j_next = ti.cast(j_next_f, ti.i32)
        if type_mask[i, j] == 0:
            for direction in ti.static(range(9)):
                sdm = 0.
                sdm -= f_x[fx_bank, i, j, direction]
                sdm += f_x[fx_bank, i_next, j_next, reverse_direction[direction]]
                dm += sdm
        else:
            for direction in ti.static(range(9)):
                sdm = 0.
                sdm -= f_x[fx_bank, i, j, direction]
                sdm += f_x[fx_bank, i_next, j_next, reverse_direction[direction]]
                if type_mask[i_next, j_next] == 0:
                    dm += sdm
                elif type_mask[i_next, j_next] == 1:
                    dm += sdm * 0.5 * (volume_fraction[i, j] + volume_fraction[i_next, j_next])
        mass[mass_bank_next, i, j] = mass[mass_bank, i, j] + dm


@ti.kernel
def update_rho_v():
    # 先不计算边界层
    new_sel = fx_bank_sel[None]
    for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
        local_rho = 0.0
        local_v = ti.Vector([0.0, 0.0])
        for direction in ti.static(range(9)):
            f = f_x[new_sel, i, j, direction]
            local_rho += f
            local_v += lattice_vector[direction] * f
        local_v /= local_rho
        rho[i, j] = local_rho
        v[i, j] = local_v


@ti.func
def boundary_dirichlet(boundary_v, i_b, j_b, i_inside, j_inside):
    v[i_b, j_b] = boundary_v
    rho[i_b, j_b] = rho[i_inside, j_inside]
    for direction in ti.static(range(9)):
        f_x[fx_bank_sel[None], i_b, j_b, direction] = f_eq(i_b, j_b, direction) - f_eq(i_inside, j_inside, direction) \
                                                      + f_x[fx_bank_sel[None], i_inside, j_inside, direction]


@ti.kernel
def get_display_var():
    for i, j in ti.ndrange(nx, ny):
        display_var[i, j] = ti.sqrt(v[i, j][0] ** 2.0 + v[i, j][1] ** 2.0)


# Boundary condition
@ti.kernel
def boundary_condition():
    # Left and right
    for j in ti.ndrange(1, ny - 1):
        boundary_dirichlet(ti.Vector([0.0, 0.0]), 0, j, 1, j)
        boundary_dirichlet(ti.Vector([0.0, 0.0]), nx - 1, j, nx - 2, j)
    # Top and bottom
    for i in ti.ndrange(nx):
        boundary_dirichlet(ti.Vector([0.0, 0.0]), i, 0, i, 1)
        boundary_dirichlet(ti.Vector([0.2, 0.0]), i, ny - 1, i, ny - 2)


def solve():
    gui = ti.GUI('lbm-2d', (nx, ny))
    init()
    for i in range(steps):
        collision_stream()
        update_rho_v()
        boundary_condition()
        if (i % 100 == 0):
            print(str(i) + ' updates \n')
            get_display_var()
            img = cm.plasma(display_var.to_numpy() / 0.15)
            gui.set_image(img)
            gui.show()


solve()

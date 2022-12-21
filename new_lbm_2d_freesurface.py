import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

eq_v_weight_d2q9 = ti.field(ti.f32, shape=9)
lattice_vector_d2q9 = ti.Vector.field(2, ti.i8, shape=9)
np_arr = np.array([[+1, +0], [+0, +1], [+1, +1], [-1, +1],
                   [0, 0],
                   [+1, -1], [-1, -1], [+0, -1], [-1, +0]], dtype=np.int32)
lattice_vector_d2q9.from_numpy(np_arr)
np_arr = np.array([1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0,
                   4.0 / 9.0,
                   1.0 / 36.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 9.0], dtype=np.float32)
eq_v_weight_d2q9.from_numpy(np_arr)


@ti.data_oriented
class LbmD2Q9FreeSurface:

    def __init__(self, x_size: int, y_size: int):
        self.nx = x_size
        self.ny = y_size
        self.f_x = ti.field(ti.f32, shape=(2, self.nx, self.ny, 9))
        self.mass = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.mass_exchange = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.fraction = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.display_var = ti.field(ti.f32, shape=(self.nx, self.ny))

        # Viscosity define
        self.niu = 0.0255
        # 由流体粘度 计算流体松弛时间tau
        # self.tau = 3.0 * self.niu + 0.5
        # self.inv_tau = 1.0 / self.tau
        self.tau = ti.field(ti.f32, shape=())
        self.inv_tau = ti.field(ti.f32, shape=())

        self.tau[None] = 3.0 * self.niu + 0.5
        self.inv_tau[None] = 1.0 / self.tau[None]

        self.global_force = ti.field(ti.f32, shape=2)
        self.global_force[1] = -0.00777

        # Lower 4 flag bits is used to mark empty block, fluid block or interface block. (x & 15)
        # 0 for fluid 1 for interface 2 for empty.
        # +4 0 for interface to fluid, 2 for interface to empty
        # +8 0 for standstill wall.

        # Higher 4 flag bits is used to mark neighborhood status. (x >> 4)
        # 0 for standard, 1 for no_fluid_neighbors, 2 for no_empty_neighbors
        self.flag = ti.field(ti.u8, shape=(self.nx, self.ny))
        self.f_x_bank_sel = ti.field(ti.i32, shape=())
        self.timestep = ti.field(ti.f32, shape=())
        self.timestep[None] = 1.0

    @ti.func
    def get_density(self, bank, x, y) -> ti.f32:
        m = 0.0
        for direction in ti.static(range(9)):
            m += self.f_x[bank, x, y, direction]
        return m

    @ti.func
    def get_velocity(self, bank: int, x: int, y: int, d: ti.f32):
        v = ti.Vector([0., 0.], dt=ti.f32)
        assert d > 0
        for direction in ti.static(range(9)):
            v += ti.Vector([lattice_vector_d2q9[direction][0] * self.f_x[bank, x, y, direction],
                            lattice_vector_d2q9[direction][1] * self.f_x[bank, x, y, direction]], dt=ti.f32)
        v /= d
        return v

    @ti.func
    def cal_feq(self, density, velocity, direction):
        u_dot_c = lattice_vector_d2q9[direction][0] * velocity[0] + lattice_vector_d2q9[direction][1] * velocity[1]
        u_dot_u = ti.Vector.dot(velocity, velocity)
        val = eq_v_weight_d2q9[direction] * density * (1.0 + 3.0 * u_dot_c + 4.5 * u_dot_c ** 2 - 1.5 * u_dot_u)
        assert 0 <= val <= 1.1
        return val

    @ti.func
    def set_post_collision_distributions_bgk(self, bank, x, y, direction, inv_tau, f_eq):
        self.f_x[bank, x, y, direction] = self.f_x[bank, x, y, direction] - inv_tau * (
                self.f_x[bank, x, y, direction] - f_eq)

    @ti.func
    def get_lower_flag(self, x, y):
        # if self.flag[x, y] & 15 > 7:
        #     print(self.flag[x, y] & 15, self.flag[x, y])
        return self.flag[x, y] & 15

    @ti.func
    def get_higher_flag(self, x, y):
        return self.flag[x, y] >> 4

    @ti.func
    def set_lower_flag(self, x, y, value):
        self.flag[x, y] = ti.u8(self.get_higher_flag(x, y) << 4) | ti.u8(value)

    @ti.func
    def set_higher_flag(self, x, y, value):
        self.flag[x, y] = ti.u8(self.get_lower_flag(x, y)) | ti.u8(value << 4)

    @ti.func
    def get_neighbor_index_p(self, x, y, direction_index):
        # p means positive
        return x + lattice_vector_d2q9[direction_index][0], y + lattice_vector_d2q9[direction_index][1]

    @ti.func
    def get_neighbor_index_n(self, x, y, direction_index):
        # n means negative
        return x - lattice_vector_d2q9[direction_index][0], y - lattice_vector_d2q9[direction_index][1]

    @ti.func
    def get_fluid_fraction(self, x, y):
        return min(1.0, max(0.0, self.fraction[x, y]))

    @ti.func
    def get_surface_normal(self, x, y):
        upper_fraction = self.get_fluid_fraction(x, y + 1)
        lower_fraction = self.get_fluid_fraction(x, y - 1)
        left_fraction = self.get_fluid_fraction(x - 1, y)
        right_fraction = self.get_fluid_fraction(x + 1, y)
        return -0.5 * ti.Vector([right_fraction - left_fraction, upper_fraction - lower_fraction])

    @ti.func
    def cal_se(self, x, y, direction_index):
        f_x_bank = self.f_x_bank_sel[None]
        n_x, n_y = self.get_neighbor_index_p(x, y, direction_index)
        ret_value = self.f_x[f_x_bank, n_x, n_y, 8 - direction_index]
        if self.get_higher_flag(x, y) == self.get_higher_flag(n_x, n_y):
            ret_value -= self.f_x[f_x_bank, x, y, direction_index]

        elif (self.get_higher_flag(n_x, n_y) == 0 and self.get_higher_flag(x, y) == 1) or (
                self.get_higher_flag(n_x, n_y) == 2 and (self.get_higher_flag(x, y) <= 1)):
            ret_value = -self.f_x[f_x_bank, x, y, direction_index]

        return ret_value

    @ti.kernel
    def std_streaming(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        # for x, y in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y in ti.ndrange(self.nx, self.ny):
            if self.get_lower_flag(x, y) <= 1:
                # standard streaming step
                for direction in ti.static(range(9)):
                    n_x, n_y = self.get_neighbor_index_n(x, y, direction)
                    assert self.f_x[f_x_bank, n_x, n_y, direction] >= 0.
                    self.f_x[f_x_bank_next, x, y, direction] = self.f_x[f_x_bank, n_x, n_y, direction]

            if self.get_lower_flag(x, y) == 1:
                normal = self.get_surface_normal(x, y)
                has_fluid_neighbors = False
                has_empty_neighbors = False
                atmosphere_pressure = 1.0
                cur_density = self.get_density(f_x_bank, x, y)
                velocity = self.get_velocity(f_x_bank, x, y, cur_density)
                for direction in ti.static([0, 1, 2, 3, 5, 6, 7, 8]):
                    n_x, n_y = self.get_neighbor_index_n(x, y, direction)
                    neighbor_is_empty = self.get_lower_flag(n_x, n_y) == 2
                    neighbor_is_fluid = self.get_lower_flag(n_x, n_y) == 0
                    has_empty_neighbors = has_empty_neighbors or neighbor_is_empty
                    has_fluid_neighbors = has_fluid_neighbors or neighbor_is_fluid

                    dot = lattice_vector_d2q9[8 - direction][0] * normal[0] + lattice_vector_d2q9[8 - direction][1] * \
                          normal[1]
                    in_normal_direction = dot > 0.0

                    if in_normal_direction or neighbor_is_empty:
                        # distribution function need to be reconstruction
                        self.f_x[f_x_bank_next, x, y, direction] = self.cal_feq(atmosphere_pressure, velocity,
                                                                                8 - direction) + self.cal_feq(
                            atmosphere_pressure, velocity, direction) - self.f_x[f_x_bank, x, y, 8 - direction]
                is_standard_cell = has_fluid_neighbors and has_empty_neighbors
                if is_standard_cell:
                    self.set_higher_flag(x, y, 0)
                elif not has_fluid_neighbors:
                    self.set_higher_flag(x, y, 1)
                elif not has_empty_neighbors:
                    self.set_higher_flag(x, y, 2)

    @ti.kernel
    def stream_mass(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        self.f_x_bank_sel[None] = f_x_bank_next
        # for x, y in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y in ti.ndrange(self.nx, self.ny):
            delta_mass = 0.0
            if self.get_lower_flag(x, y) != 1:
                continue
            cur_fluid_fraction = self.get_fluid_fraction(x, y)
            for direction in ti.static([0, 1, 2, 3, 5, 6, 7, 8]):
                n_x, n_y = self.get_neighbor_index_p(x, y, direction)
                if self.get_lower_flag(n_x, n_y) == 0:
                    delta_mass += self.f_x[f_x_bank, n_x, n_y, 8 - direction] - self.f_x[
                        f_x_bank, x, y, direction]
                elif self.get_lower_flag(n_x, n_y) == 1:
                    # exchange mass betweem interface
                    neigh_fluid_fraction = self.get_fluid_fraction(n_x, n_y)
                    s_e = self.cal_se(x, y, direction)
                    # s_e = self.f_x[f_x_bank, n_x, n_y, 8 - direction] - self.f_x[
                    #     f_x_bank, x, y, direction]
                    delta_mass += s_e * 0.5 * (cur_fluid_fraction + neigh_fluid_fraction)
            self.mass[x, y] = self.mass[x, y] + delta_mass
        return

    @ti.kernel
    def collision(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        # for x, y in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y in ti.ndrange(self.nx, self.ny):
            if self.get_lower_flag(x, y) <= 1:
                cur_density = self.get_density(f_x_bank, x, y)
                cur_velocity = self.get_velocity(f_x_bank, x, y, cur_density)
                cur_velocity = cur_velocity + ti.Vector([self.global_force[0] * self.tau[None],
                                                         self.global_force[1] * self.tau[None]])
                for direction in ti.static(range(9)):
                    f_eq = self.cal_feq(cur_density, cur_velocity, direction)
                    self.set_post_collision_distributions_bgk(f_x_bank, x, y, direction, self.inv_tau[None], f_eq)
                cur_density = self.get_density(f_x_bank, x, y)
                if self.get_lower_flag(x, y) == 0:
                    self.mass[x, y] = cur_density
                else:
                    self.fraction[x, y] = self.mass[x, y] / cur_density
        return

    @ti.kernel
    def potential_update(self):
        # for x, y in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y in ti.ndrange(self.nx, self.ny):
            if self.get_lower_flag(x, y) != 1:
                continue
            cur_density = 0.0
            if self.fraction[x, y] != 0:
                cur_density = self.mass[x, y] / self.fraction[x, y]
            else:
                cur_density = 0.0

            if self.mass[x, y] > 1.0001 * cur_density:
                self.set_lower_flag(x, y, 0 + 4)
            elif self.mass[x, y] < -0.0001 * cur_density:
                self.set_lower_flag(x, y, 2 + 4)

            if self.mass[x, y] < 0.1 * cur_density and self.get_higher_flag(x, y) == 1:
                self.set_lower_flag(x, y, 2 + 4)
            elif self.mass[x, y] > 0.9 * cur_density and self.get_higher_flag(x, y) == 2:
                self.set_lower_flag(x, y, 0 + 4)
        return

    @ti.func
    def interpolate_empty_cell(self, bank: int, x: int, y: int):
        num_neighs = 0
        avg_density = ti.f32(0.)
        avg_vel = ti.Vector([0., 0.], dt=ti.f32)
        for direction in range(9):
            if direction == 4:
                continue
            n_x, n_y = self.get_neighbor_index_p(x, y, direction)
            n_flag = self.get_lower_flag(n_x, n_y)
            if n_flag == 0 or n_flag == 1 or n_flag == 0 + 4:
                neigh_density = self.get_density(bank, n_x, n_y)
                neigh_velocity = self.get_velocity(bank, n_x, n_y, neigh_density)
                # neigh_velocity = ti.Vector([0., 0.])
                num_neighs = num_neighs + 1
                avg_density = avg_density + neigh_density
                avg_vel = avg_vel + neigh_velocity
        assert num_neighs != 0
        avg_vel /= num_neighs
        avg_density /= num_neighs
        self.fraction[x, y] = self.mass[x, y] / avg_density
        for direction in ti.static(range(9)):
            self.f_x[bank, x, y, direction] = self.cal_feq(avg_density, avg_vel, direction)
            # self.f_x[bank, x, y, direction] = 0
        return

    @ti.kernel
    def flag_reinit_1(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        # for x, y in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y in ti.ndrange(self.nx, self.ny):
            if self.get_lower_flag(x, y) == 0 + 4:
                for direction in range(9):
                    if direction == 4:
                        continue
                    n_x, n_y = self.get_neighbor_index_p(x, y, direction)
                    if self.get_lower_flag(n_x, n_y) == 2:
                        self.set_lower_flag(n_x, n_y, 1)
                        self.mass[n_x, n_y] = 0.0
                        self.fraction[n_x, n_y] = 0.0
                        self.interpolate_empty_cell(f_x_bank, n_x, n_y)
                    elif self.get_lower_flag(n_x, n_y) == 2 + 4:
                        self.set_lower_flag(n_x, n_y, 1)
        return

    @ti.kernel
    def flag_reinit_2(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        # for x, y in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y in ti.ndrange(self.nx, self.ny):
            if self.get_lower_flag(x, y) == 2 + 4:
                for direction in ti.static([0, 1, 2, 3, 5, 6, 7, 8]):
                    n_x, n_y = self.get_neighbor_index_p(x, y, direction)
                    if self.get_lower_flag(n_x, n_y) == 0:
                        self.set_lower_flag(n_x, n_y, 1)
                        self.mass[n_x, n_y] = self.get_density(f_x_bank, n_x, n_y)
                        self.fraction[n_x, n_y] = 1.0
        return

    @ti.func
    def mass_distribute_single(self, bank, x, y, is_filled):
        excess_mass = 0.
        density = self.get_density(bank, x, y)
        if is_filled:
            excess_mass = self.mass[x, y] - density
            # atomic operation
            self.mass_exchange[x, y] -= excess_mass
        else:
            excess_mass = self.mass[x, y]
            # atomic operation
            self.mass_exchange[x, y] -= excess_mass

        normal = self.get_surface_normal(x, y)
        weight = ti.Vector([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        weight_back = ti.Vector([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        normalizer = 0.0
        for direction in ti.static([0, 1, 2, 3, 5, 6, 7, 8]):
            is_valid = True
            n_x, n_y = self.get_neighbor_index_p(x, y, direction)
            if self.get_lower_flag(n_x, n_y) != 1:
                is_valid = False
            if is_valid:
                weight_back[direction] = 1.0
                dot = normal[0] * lattice_vector_d2q9[direction][0] + normal[1] * lattice_vector_d2q9[direction][1]
                if is_filled:
                    weight[direction] = max(0.0, dot)
                else:
                    weight[direction] = -min(0.0, dot)
                normalizer += weight[direction]

        is_valid_n = True
        if normalizer == 0.0:
            weight = weight_back
            for direction in ti.static([0, 1, 2, 3, 5, 6, 7, 8]):
                normalizer += weight[direction]

        if normalizer == 0.0:
            is_valid_n = False
        if is_valid_n:
            # redistribute weights as non-interface cells have weights 0
            for direction in ti.static(range(9)):
                n_x, n_y = self.get_neighbor_index_p(x, y, direction)
                self.mass_exchange[n_x, n_y] += (weight[direction] / normalizer) * excess_mass
        return

    @ti.kernel
    def mass_distribute(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        # for x, y in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y in ti.ndrange(self.nx, self.ny):
            if self.get_lower_flag(x, y) == 0 + 4:
                self.mass_distribute_single(f_x_bank, x, y, True)
                self.set_lower_flag(x, y, 0)
            elif self.get_lower_flag(x, y) == 2 + 4:
                self.mass_distribute_single(f_x_bank, x, y, False)
                self.set_lower_flag(x, y, 2)
        for x, y in ti.ndrange(self.nx, self.ny):
            self.mass[x, y] = self.mass[x, y] + self.mass_exchange[x, y]
            self.fraction[x, y] = self.mass[x, y] / self.get_density(f_x_bank, x, y)
            self.mass_exchange[x, y] = 0
        return

    @ti.kernel
    def adapt_timestep(self, allow_increase: ti.u8):
        f_x_bank = self.f_x_bank_sel[None]
        maximum_velocity = 0.0
        for x, y in ti.ndrange(self.nx, self.ny):
            if self.get_lower_flag(x, y) > 2:
                continue
            density = self.get_density(f_x_bank, x, y)
            velocity_vec = self.get_velocity(f_x_bank, x, y, density)
            velocity = velocity_vec[0] * velocity_vec[0] + velocity_vec[1] * velocity_vec[1]
            ti.atomic_max(maximum_velocity, velocity)
        maximum_velocity = ti.sqrt(maximum_velocity)
        critical_velocity = 0.5 / 3
        multiplier = 4.0 / 5.0
        upper_limit = critical_velocity / multiplier
        lower_limit = critical_velocity * multiplier

        old_time_step = self.timestep[None]
        new_time_step = self.timestep[None]

        is_invalid = False
        if maximum_velocity > upper_limit:
            new_time_step *= multiplier
        elif maximum_velocity < lower_limit and allow_increase != 0:
            new_time_step /= multiplier
        else:
            is_invalid = True
        if not is_invalid:
            time_ratio = new_time_step / old_time_step
            new_inv_tau = (time_ratio * ((self.inv_tau[None]) - 0.5) + 0.5)
            new_tau = 1 / new_inv_tau
            self.global_force[0] = time_ratio * time_ratio * self.global_force[0]
            self.global_force[1] = time_ratio * time_ratio * self.global_force[1]
            total_fluid_volume = 0.0
            total_mass = 0.0
            for x, y in ti.ndrange(self.nx, self.ny):
                if self.get_lower_flag(x, y) == 0:
                    total_mass += self.mass[x, y]
                    total_fluid_volume += 1.0
                elif self.get_lower_flag(x, y) == 1:
                    total_mass += self.mass[x, y]
                    total_fluid_volume += self.fraction[x, y]

            median_density = total_mass / total_fluid_volume

            # rescale distributions and densities
            for x, y in ti.ndrange(self.nx, self.ny):
                if self.get_lower_flag(x, y) > 2:
                    continue
                old_density = self.get_density(f_x_bank, x, y)
                new_density = time_ratio * (old_density - median_density) + median_density

                old_velocity = self.get_velocity(f_x_bank, x, y, old_density)
                new_velocity = old_velocity * time_ratio

                tau_radio = time_ratio * (new_tau * self.inv_tau[None])
                for direction in ti.static(range(9)):
                    old_feq = self.cal_feq(old_density, old_velocity, direction)
                    new_feq = self.cal_feq(new_density, new_velocity, direction)
                    feq_ratio = new_feq / old_feq
                    self.f_x[f_x_bank, x, y, direction] = feq_ratio * (
                            old_feq + tau_radio * (self.f_x[f_x_bank, x, y, direction] - old_feq))

                if self.get_lower_flag(x, y) == 1:
                    self.mass[x, y] = self.mass[x, y] * (old_density / new_density)
                    self.fraction[x, y] = self.mass[x, y] / new_density

            print('timestep changed from', self.timestep[None], 'to', new_time_step)
            print('max_velocity:', maximum_velocity)
            self.timestep[None] = new_time_step
            self.tau[None] = new_tau
            self.inv_tau[None] = new_inv_tau
        return

    @ti.func
    def no_slip_single(self, bank, x, y):
        for direction in ti.static(range(9)):
            n_x, n_y = self.get_neighbor_index_p(x, y, direction)
            if not (n_x < 0 or n_y < 0 or n_x >= self.nx or n_y >= self.ny):
                self.f_x[bank, x, y, direction] = self.f_x[bank, n_x, n_y, 8 - direction]
                if self.get_lower_flag(n_x, n_y) == 2:
                    self.f_x[bank, x, y, direction] = 0
        return

    @ti.kernel
    def boundary_condition(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        for x, y in ti.ndrange(self.nx, self.ny):
            if self.get_lower_flag(x, y) == 8 + 0:
                self.no_slip_single(f_x_bank, x, y)
        return

    @ti.kernel
    def init_drop(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        for x, y in ti.ndrange(self.nx, self.ny):
            if x <= 0 or x == self.nx - 1 or y == 0 or y == self.ny - 1:
                self.flag[x, y] = ti.u8(8)
                self.mass[x, y] = 0.0
                self.fraction[x, y] = 0.0
                for direction in ti.static(range(9)):
                    self.f_x[f_x_bank, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
            elif y < self.ny / 16 or (x - self.nx / 2) ** 2 + (y - 64 - self.ny / 2) ** 2 < 400:
                self.flag[x, y] = ti.u8(0)
                self.mass[x, y] = 1.0
                self.fraction[x, y] = 1.0
                for direction in ti.static(range(9)):
                    self.f_x[f_x_bank, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
            elif (x - self.nx / 2) ** 2 + (y - 64 - self.ny / 2) ** 2 < 441:
                self.flag[x, y] = ti.u8(1)
                self.mass[x, y] = 0.5
                self.fraction[x, y] = 0.5
                for direction in ti.static(range(9)):
                    self.f_x[f_x_bank, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)

            elif y > self.ny / 16:
                self.flag[x, y] = ti.u8(2)
                self.mass[x, y] = 0.0
                self.fraction[x, y] = 0.0
                for direction in ti.static(range(9)):
                    self.f_x[f_x_bank, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
            else:
                self.flag[x, y] = ti.u8(1)
                self.mass[x, y] = 0.5
                self.fraction[x, y] = 0.5
                for direction in ti.static(range(9)):
                    self.f_x[f_x_bank, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)

        return

    @ti.kernel
    def init_water(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        for x, y in ti.ndrange(self.nx, self.ny):
            if x <= 0 or x == self.nx - 1 or y == 0 or y == self.ny - 1:
                self.flag[x, y] = ti.u8(8)
                self.mass[x, y] = 0.0
                self.fraction[x, y] = 0.0
                for direction in ti.static(range(9)):
                    self.f_x[f_x_bank, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
            elif x > self.nx / 2:
                self.flag[x, y] = ti.u8(0)
                self.mass[x, y] = 1.0
                self.fraction[x, y] = 1.0
                for direction in ti.static(range(9)):
                    self.f_x[f_x_bank, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
            elif x < self.nx / 2:
                self.flag[x, y] = ti.u8(2)
                self.mass[x, y] = 0.0
                self.fraction[x, y] = 0.0
                for direction in ti.static(range(9)):
                    self.f_x[f_x_bank, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
            else:
                self.flag[x, y] = ti.u8(1)
                self.mass[x, y] = 0.5
                self.fraction[x, y] = 0.5
                for direction in ti.static(range(9)):
                    self.f_x[f_x_bank, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, direction] = self.cal_feq(1.0, ti.Vector([0., 0.]), direction)

        return

    @ti.kernel
    def get_display_var(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        for x, y in ti.ndrange(self.nx, self.ny):
            density = self.get_density(f_x_bank, x, y)
            velocity = self.get_velocity(f_x_bank, x, y, density)
            if self.get_lower_flag(x, y) > 1:
                density = 0.0
                velocity = ti.Vector([0., 0.])
            if self.get_lower_flag(x, y) == 1:
                density = 1.0
                velocity = ti.Vector([1., 1.])
            self.display_var[x, y] = ti.sqrt(velocity[0] ** 2 + velocity[1] ** 2) * 10
            # self.display_var[x, y] = self.flag[x, y] < 2
            # self.display_var[x, y] = self.get_lower_flag(x, y) == 1
            # self.display_var[x, y] = self.mass[x, y]
            # self.display_var[x, y] = ti.abs(self.mass[x, y] - density)
            # self.display_var[x, y] = self.fraction[x, y]
            # self.display_var[x, y] = density
        return

    def solve(self):
        gui = ti.GUI('lbm-2d-freesurface', (self.nx, self.ny))
        self.init_drop()
        self.get_display_var()
        gui.set_image(self.display_var)
        gui.show()
        time_record = 0.0
        last_print_time = 0.0
        increase_next = 0.0
        for i in range(10000000):
            self.std_streaming()
            if i == 0:
                print(1, i)
            self.stream_mass()
            if i == 0:
                print(2, i)
            if i == 0:
                print(3, i)
            self.collision()
            if i == 0:
                print(4, i)
            self.potential_update()
            if i == 0:
                print(5, i)
            self.flag_reinit_1()
            if i == 0:
                print(6, i)
            self.flag_reinit_2()
            if i == 0:
                print(7, i)
            self.mass_distribute()
            if i == 0:
                print(8, i)

            step_size_before = self.timestep[None]
            allow_increase = increase_next > i
            time_record += step_size_before

            self.adapt_timestep(allow_increase)

            if step_size_before > self.timestep[None]:
                increase_next = i + 200

            self.boundary_condition()
            if i == 0:
                print(9, i)

            if time_record - last_print_time >= 10.0:
                last_print_time = time_record
                self.get_display_var()
                # print(10, i)
                # img = cm.plasma(self.display_var.to_numpy() / 0.15)
                img = self.display_var
                gui.set_image(img)
                gui.show()


simulator_obj = LbmD2Q9FreeSurface(256, 256)
simulator_obj.solve()

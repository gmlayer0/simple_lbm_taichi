import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, offline_cache=True)

eq_v_weight_d3q19 = ti.field(ti.f32, shape=19)
lattice_vector_d3q19 = ti.Vector.field(3, ti.i8, shape=19)
np_arr = np.array([[0, -1, -1], [-1, 0, -1], [0, 0, -1], [1, 0, -1],
                   [0, 1, -1], [-1, -1, 0], [0, -1, 0], [1, -1, 0],
                   [-1, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 1, 0],
                   [0, 1, 0], [1, 1, 0], [0, -1, 1], [-1, 0, 1],
                   [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.int32)
lattice_vector_d3q19.from_numpy(np_arr)
np_arr = np.array([(1. / 36), (1. / 36), (2. / 36), (1. / 36),
                   (1. / 36), (1. / 36), (2. / 36), (1. / 36),
                   (2. / 36), (12. / 36), (2. / 36), (1. / 36),
                   (2. / 36), (1. / 36), (1. / 36), (1. / 36),
                   (2. / 36), (1. / 36), (1. / 36)], dtype=np.float32)
eq_v_weight_d3q19.from_numpy(np_arr)


@ti.data_oriented
class LbmD2Q9FreeSurface:

    def __init__(self, x_size: int, y_size: int, z_size: int):
        self.nx = x_size
        self.ny = y_size
        self.nz = z_size
        self.f_x = ti.field(ti.f32, shape=(2, self.nx, self.ny, self.nz, 19))
        self.mass = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
        self.mass_exchange = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
        self.fraction = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
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

        self.global_force = ti.field(ti.f32, shape=3)
        self.global_force[2] = -0.00777

        # Lower 4 flag bits is used to mark empty block, fluid block or interface block. (x & 15)
        # 0 for fluid 1 for interface 2 for empty.
        # +4 0 for interface to fluid, 2 for interface to empty
        # +8 0 for standstill wall.

        # Higher 4 flag bits is used to mark neighborhood status. (x >> 4)
        # 0 for standard, 1 for no_fluid    _neighbors, 2 for no_empty_neighbors
        self.flag = ti.field(ti.u8, shape=(self.nx, self.ny, self.nz))
        self.f_x_bank_sel = ti.field(ti.i32, shape=())
        self.timestep = ti.field(ti.f32, shape=())
        self.timestep[None] = 1.0

    @ti.func
    def get_density(self, bank, x, y, z) -> ti.f32:
        m = 0.0
        for direction in ti.static(range(19)):
            m += self.f_x[bank, x, y, z, direction]
        return m

    @ti.func
    def get_velocity(self, bank: int, x: int, y: int, z: int, d: ti.f32):
        v = ti.Vector([0., 0., 0.], dt=ti.f32)
        assert d > 0
        for direction in ti.static(range(19)):
            v += ti.Vector([lattice_vector_d3q19[direction][0] * self.f_x[bank, x, y, z, direction],
                            lattice_vector_d3q19[direction][1] * self.f_x[bank, x, y, z, direction],
                            lattice_vector_d3q19[direction][2] * self.f_x[bank, x, y, z, direction]], dt=ti.f32)
        v /= d
        return v

    @ti.func
    def cal_feq(self, density, velocity, direction):
        u_dot_c = lattice_vector_d3q19[direction][0] * velocity[0] + lattice_vector_d3q19[direction][1] * velocity[1] + \
                  lattice_vector_d3q19[direction][2] * velocity[2]
        u_dot_u = ti.Vector.dot(velocity, velocity)
        val = eq_v_weight_d3q19[direction] * density * (1.0 + 3.0 * u_dot_c + 4.5 * u_dot_c ** 2 - 1.5 * u_dot_u)
        assert 0 <= val <= 1.1
        return val

    @ti.func
    def set_post_collision_distributions_bgk(self, bank, x, y, z, direction, inv_tau, f_eq):
        self.f_x[bank, x, y, z, direction] = self.f_x[bank, x, y, z, direction] - inv_tau * (
                self.f_x[bank, x, y, z, direction] - f_eq)

    @ti.func
    def get_lower_flag(self, x, y, z):
        return self.flag[x, y, z] & 15

    @ti.func
    def get_higher_flag(self, x, y, z):
        return self.flag[x, y, z] >> 4

    @ti.func
    def set_lower_flag(self, x, y, z, value):
        self.flag[x, y, z] = ti.u8(self.get_higher_flag(x, y, z) << 4) | ti.u8(value)

    @ti.func
    def set_higher_flag(self, x, y, z, value):
        self.flag[x, y, z] = ti.u8(self.get_lower_flag(x, y, z)) | ti.u8(value << 4)

    @ti.func
    def get_neighbor_index_p(self, x, y, z, direction_index):
        # p means positive
        return x + lattice_vector_d3q19[direction_index][0], y + lattice_vector_d3q19[direction_index][1], \
               z + lattice_vector_d3q19[direction_index][2]

    @ti.func
    def get_neighbor_index_n(self, x, y, z, direction_index):
        # n means negative
        return x - lattice_vector_d3q19[direction_index][0], y - lattice_vector_d3q19[direction_index][1], \
               z - lattice_vector_d3q19[direction_index][2]

    @ti.func
    def get_fluid_fraction(self, x, y, z):
        return min(1.0, max(0.0, self.fraction[x, y, z]))

    @ti.func
    def get_surface_normal(self, x, y, z):
        upper_fraction = self.get_fluid_fraction(x, y, z + 1)
        lower_fraction = self.get_fluid_fraction(x, y, z - 1)
        front_fraction = self.get_fluid_fraction(x, y + 1, z)
        back_fraction = self.get_fluid_fraction(x, y - 1, z)
        right_fraction = self.get_fluid_fraction(x + 1, y, z)
        left_fraction = self.get_fluid_fraction(x - 1, y, z)
        return -0.5 * ti.Vector(
            [right_fraction - left_fraction, front_fraction - back_fraction, upper_fraction - lower_fraction])

    @ti.func
    def cal_se(self, x, y, z, direction_index):
        f_x_bank = self.f_x_bank_sel[None]
        n_x, n_y, n_z = self.get_neighbor_index_p(x, y, z, direction_index)
        ret_value = self.f_x[f_x_bank, n_x, n_y, n_z, 18 - direction_index]
        if self.get_higher_flag(x, y, z) == self.get_higher_flag(n_x, n_y, n_z):
            ret_value -= self.f_x[f_x_bank, x, y, z, direction_index]

        elif (self.get_higher_flag(n_x, n_y, n_z) == 0 and self.get_higher_flag(x, y, z) == 1) or (
                self.get_higher_flag(n_x, n_y, n_z) == 2 and (self.get_higher_flag(x, y, z) <= 1)):
            ret_value = -self.f_x[f_x_bank, x, y, z, direction_index]

        return ret_value

    @ti.kernel
    def std_streaming(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        # for x, y, z in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if self.get_lower_flag(x, y, z) <= 1:
                # standard streaming step
                for direction in ti.static(range(19)):
                    n_x, n_y, n_z = self.get_neighbor_index_n(x, y, z, direction)
                    assert self.f_x[f_x_bank, n_x, n_y, n_z, direction] >= 0.
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.f_x[f_x_bank, n_x, n_y, n_z, direction]

            if self.get_lower_flag(x, y, z) == 1:
                normal = self.get_surface_normal(x, y, z)
                has_fluid_neighbors = False
                has_empty_neighbors = False
                atmosphere_pressure = 1.0
                cur_density = self.get_density(f_x_bank, x, y, z)
                velocity = self.get_velocity(f_x_bank, x, y, z, cur_density)
                for direction in ti.static([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]):
                    n_x, n_y, n_z = self.get_neighbor_index_n(x, y, z, direction)
                    neighbor_is_empty = self.get_lower_flag(n_x, n_y, n_z) == 2
                    neighbor_is_fluid = self.get_lower_flag(n_x, n_y, n_z) == 0
                    has_empty_neighbors = has_empty_neighbors or neighbor_is_empty
                    has_fluid_neighbors = has_fluid_neighbors or neighbor_is_fluid

                    dot = lattice_vector_d3q19[18 - direction][0] * normal[0] + lattice_vector_d3q19[18 - direction][
                        1] * \
                          normal[1] + lattice_vector_d3q19[18 - direction][2] * normal[2]
                    in_normal_direction = dot > 0.0

                    if in_normal_direction or neighbor_is_empty:
                        self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(atmosphere_pressure, velocity,
                                                                                   18 - direction) + self.cal_feq(
                            atmosphere_pressure, velocity, direction) - self.f_x[f_x_bank, x, y, z, 18 - direction]
                is_standard_cell = has_fluid_neighbors and has_empty_neighbors
                if is_standard_cell:
                    self.set_higher_flag(x, y, z, 0)
                elif not has_fluid_neighbors:
                    self.set_higher_flag(x, y, z, 1)
                elif not has_empty_neighbors:
                    self.set_higher_flag(x, y, z, 2)

    @ti.kernel
    def stream_mass(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        self.f_x_bank_sel[None] = f_x_bank_next
        # for x, y, z in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            delta_mass = 0.0
            if self.get_lower_flag(x, y, z) != 1:
                continue
            cur_fluid_fraction = self.get_fluid_fraction(x, y, z)
            for direction in ti.static([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]):
                n_x, n_y, n_z = self.get_neighbor_index_p(x, y, z, direction)
                if self.get_lower_flag(n_x, n_y, n_z) == 0:
                    delta_mass += self.f_x[f_x_bank, n_x, n_y, n_z, 18 - direction] - self.f_x[
                        f_x_bank, x, y, z, direction]
                elif self.get_lower_flag(n_x, n_y, n_z) == 1:
                    # exchange mass betweem interface
                    neigh_fluid_fraction = self.get_fluid_fraction(n_x, n_y, n_z)
                    s_e = self.cal_se(x, y, z, direction)
                    # s_e = self.f_x[f_x_bank, n_x, n_y, n_z, 8 - direction] - self.f_x[
                    #     f_x_bank, x, y, z, direction]
                    delta_mass += s_e * 0.5 * (cur_fluid_fraction + neigh_fluid_fraction)
            self.mass[x, y, z] = self.mass[x, y, z] + delta_mass
        return

    @ti.kernel
    def collision(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        # for x, y, z in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if self.get_lower_flag(x, y, z) <= 1:
                cur_density = self.get_density(f_x_bank, x, y, z)
                cur_velocity = self.get_velocity(f_x_bank, x, y, z, cur_density)
                cur_velocity = cur_velocity + ti.Vector([self.global_force[0] * self.tau[None],
                                                         self.global_force[1] * self.tau[None],
                                                         self.global_force[2] * self.tau[None]])
                for direction in ti.static(range(19)):
                    f_eq = self.cal_feq(cur_density, cur_velocity, direction)
                    self.set_post_collision_distributions_bgk(f_x_bank, x, y, z, direction, self.inv_tau[None], f_eq)
                cur_density = self.get_density(f_x_bank, x, y, z)
                if self.get_lower_flag(x, y, z) == 0:
                    self.mass[x, y, z] = cur_density
                else:
                    self.fraction[x, y, z] = self.mass[x, y, z] / cur_density
        return

    @ti.kernel
    def potential_update(self):
        # for x, y, z in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if self.get_lower_flag(x, y, z) != 1:
                continue
            cur_density = 0.0
            if self.fraction[x, y, z] != 0:
                cur_density = self.mass[x, y, z] / self.fraction[x, y, z]
            else:
                cur_density = 0.0

            if self.mass[x, y, z] > 1.0001 * cur_density:
                self.set_lower_flag(x, y, z, 0 + 4)
            elif self.mass[x, y, z] < -0.0001 * cur_density:
                self.set_lower_flag(x, y, z, 2 + 4)

            if self.mass[x, y, z] < 0.1 * cur_density and self.get_higher_flag(x, y, z) == 1:
                self.set_lower_flag(x, y, z, 2 + 4)
            elif self.mass[x, y, z] > 0.9 * cur_density and self.get_higher_flag(x, y, z) == 2:
                self.set_lower_flag(x, y, z, 0 + 4)
        return

    @ti.func
    def interpolate_empty_cell(self, bank: int, x: int, y: int, z: int):
        num_neighs = 0
        avg_density = ti.f32(0.)
        avg_vel = ti.Vector([0., 0., 0.], dt=ti.f32)
        for direction in range(19):
            if direction == 9:
                continue
            n_x, n_y, n_z = self.get_neighbor_index_p(x, y, z, direction)
            n_flag = self.get_lower_flag(n_x, n_y, n_z)
            if n_flag == 0 or n_flag == 1 or n_flag == 0 + 4:
                neigh_density = self.get_density(bank, n_x, n_y, n_z)
                neigh_velocity = self.get_velocity(bank, n_x, n_y, n_z, neigh_density)
                # neigh_velocity = ti.Vector([0., 0.])
                num_neighs = num_neighs + 1
                avg_density = avg_density + neigh_density
                avg_vel = avg_vel + neigh_velocity
        assert num_neighs != 0
        avg_vel /= num_neighs
        avg_density /= num_neighs
        self.fraction[x, y, z] = self.mass[x, y, z] / avg_density
        for direction in ti.static(range(19)):
            self.f_x[bank, x, y, z, direction] = self.cal_feq(avg_density, avg_vel, direction)
            # self.f_x[bank, x, y, z, direction] = 0
        return

    @ti.kernel
    def flag_reinit_1(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        # for x, y, z in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if self.get_lower_flag(x, y, z) == 0 + 4:
                for direction in range(19):
                    if direction == 9:
                        continue
                    n_x, n_y, n_z = self.get_neighbor_index_p(x, y, z, direction)
                    if self.get_lower_flag(n_x, n_y, n_z) == 2:
                        self.set_lower_flag(n_x, n_y, n_z, 1)
                        self.mass[n_x, n_y, n_z] = 0.0
                        self.fraction[n_x, n_y, n_z] = 0.0
                        self.interpolate_empty_cell(f_x_bank, n_x, n_y, n_z)
                    elif self.get_lower_flag(n_x, n_y, n_z) == 2 + 4:
                        self.set_lower_flag(n_x, n_y, n_z, 1)
        return

    @ti.kernel
    def flag_reinit_2(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        # for x, y, z in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if self.get_lower_flag(x, y, z) == 2 + 4:
                for direction in ti.static([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]):
                    n_x, n_y, n_z = self.get_neighbor_index_p(x, y, z, direction)
                    if self.get_lower_flag(n_x, n_y, n_z) == 0:
                        self.set_lower_flag(n_x, n_y, n_z, 1)
                        self.mass[n_x, n_y, n_z] = self.get_density(f_x_bank, n_x, n_y, n_z)
                        self.fraction[n_x, n_y, n_z] = 1.0
        return

    @ti.func
    def mass_distribute_single(self, bank, x, y, z, is_filled):
        excess_mass = 0.
        density = self.get_density(bank, x, y, z)
        if is_filled:
            excess_mass = self.mass[x, y, z] - density
            # atomic operation
            self.mass_exchange[x, y, z] -= excess_mass
        else:
            excess_mass = self.mass[x, y, z]
            # atomic operation
            self.mass_exchange[x, y, z] -= excess_mass

        normal = self.get_surface_normal(x, y, z)
        weight = ti.Vector([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        weight_back = ti.Vector([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        normalizer = 0.0
        for direction in ti.static([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]):
            is_valid = True
            n_x, n_y, n_z = self.get_neighbor_index_p(x, y, z, direction)
            if self.get_lower_flag(n_x, n_y, n_z) != 1:
                is_valid = False
            if is_valid:
                weight_back[direction] = 1.0
                dot = normal[0] * lattice_vector_d3q19[direction][0] + normal[1] * lattice_vector_d3q19[direction][1] + \
                      normal[2] * lattice_vector_d3q19[direction][2]
                if is_filled:
                    weight[direction] = max(0.0, dot)
                else:
                    weight[direction] = -min(0.0, dot)
                normalizer += weight[direction]

        is_valid_n = True
        if normalizer == 0.0:
            weight = weight_back
            for direction in ti.static([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]):
                normalizer += weight[direction]

        if normalizer == 0.0:
            is_valid_n = False
        if is_valid_n:
            # redistribute weights as non-interface cells have weights 0
            for direction in ti.static(range(19)):
                n_x, n_y, n_z = self.get_neighbor_index_p(x, y, z, direction)
                self.mass_exchange[n_x, n_y, n_z] += (weight[direction] / normalizer) * excess_mass
        return

    @ti.kernel
    def mass_distribute(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        # for x, y, z in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if self.get_lower_flag(x, y, z) == 0 + 4:
                self.mass_distribute_single(f_x_bank, x, y, z, True)
                self.set_lower_flag(x, y, z, 0)
            elif self.get_lower_flag(x, y, z) == 2 + 4:
                self.mass_distribute_single(f_x_bank, x, y, z, False)
                self.set_lower_flag(x, y, z, 2)
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            self.mass[x, y, z] = self.mass[x, y, z] + self.mass_exchange[x, y, z]
            self.fraction[x, y, z] = self.mass[x, y, z] / self.get_density(f_x_bank, x, y, z)
            self.mass_exchange[x, y, z] = 0
        return

    @ti.kernel
    def adapt_timestep(self, allow_increase: ti.u8):
        f_x_bank = self.f_x_bank_sel[None]
        maximum_velocity = 0.0
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if self.get_lower_flag(x, y, z) > 2:
                continue
            density = self.get_density(f_x_bank, x, y, z)
            velocity_vec = self.get_velocity(f_x_bank, x, y, z, density)
            velocity = velocity_vec[0] * velocity_vec[0] + velocity_vec[1] * velocity_vec[1] + velocity_vec[2] * \
                       velocity_vec[2]
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
            self.global_force[2] = time_ratio * time_ratio * self.global_force[2]
            total_fluid_volume = 0.0
            total_mass = 0.0
            for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
                if self.get_lower_flag(x, y, z) == 0:
                    total_mass += self.mass[x, y, z]
                    total_fluid_volume += 1.0
                elif self.get_lower_flag(x, y, z) == 1:
                    total_mass += self.mass[x, y, z]
                    total_fluid_volume += self.fraction[x, y, z]

            median_density = total_mass / total_fluid_volume

            # rescale distributions and densities
            for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
                if self.get_lower_flag(x, y, z) > 2:
                    continue
                old_density = self.get_density(f_x_bank, x, y, z)
                new_density = time_ratio * (old_density - median_density) + median_density

                old_velocity = self.get_velocity(f_x_bank, x, y, z, old_density)
                new_velocity = old_velocity * time_ratio

                tau_radio = time_ratio * (new_tau * self.inv_tau[None])
                for direction in ti.static(range(19)):
                    old_feq = self.cal_feq(old_density, old_velocity, direction)
                    new_feq = self.cal_feq(new_density, new_velocity, direction)
                    feq_ratio = new_feq / old_feq
                    self.f_x[f_x_bank, x, y, z, direction] = feq_ratio * (
                            old_feq + tau_radio * (self.f_x[f_x_bank, x, y, z, direction] - old_feq))

                if self.get_lower_flag(x, y, z) == 1:
                    self.mass[x, y, z] = self.mass[x, y, z] * (old_density / new_density)
                    self.fraction[x, y, z] = self.mass[x, y, z] / new_density

            print('timestep changed from', self.timestep[None], 'to', new_time_step)
            print('max_velocity:', maximum_velocity)
            self.timestep[None] = new_time_step
            self.tau[None] = new_tau
            self.inv_tau[None] = new_inv_tau
        return

    @ti.func
    def no_slip_single(self, bank, x, y, z):
        for direction in ti.static(range(19)):
            n_x, n_y, n_z = self.get_neighbor_index_p(x, y, z, direction)
            if not (n_x < 0 or n_y < 0 or n_x >= self.nx or n_y >= self.ny):
                self.f_x[bank, x, y, z, direction] = self.f_x[bank, n_x, n_y, n_z, 18 - direction]
                if self.get_lower_flag(n_x, n_y, n_z) == 2:
                    self.f_x[bank, x, y, z, direction] = 0
        return

    @ti.kernel
    def boundary_condition(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if self.get_lower_flag(x, y, z) == 8 + 0:
                self.no_slip_single(f_x_bank, x, y, z)
        return

    @ti.kernel
    def init_drop(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if x <= 0 or x == self.nx - 1 or y == 0 or y == self.ny - 1 or z == 0 or z == self.nz - 1:
                self.flag[x, y, z] = ti.u8(8)
                self.mass[x, y, z] = 0.0
                self.fraction[x, y, z] = 0.0
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
            elif z < self.nz / 8 or (x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - 8) ** 2 < 36:
                self.flag[x, y, z] = ti.u8(0)
                self.mass[x, y, z] = 1.0
                self.fraction[x, y, z] = 1.0
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
            elif (x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - 8) ** 2 < 50:
                self.flag[x, y, z] = ti.u8(1)
                self.mass[x, y, z] = 0.5
                self.fraction[x, y, z] = 0.5
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)

            elif z > self.ny / 8:
                self.flag[x, y, z] = ti.u8(2)
                self.mass[x, y, z] = 0.0
                self.fraction[x, y, z] = 0.0
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
            else:
                self.flag[x, y, z] = ti.u8(1)
                self.mass[x, y, z] = 0.5
                self.fraction[x, y, z] = 0.5
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)

        return

    @ti.kernel
    def init_water(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if x <= 0 or x == self.nx - 1 or y == 0 or y == self.ny - 1:
                self.flag[x, y, z] = ti.u8(8)
                self.mass[x, y, z] = 0.0
                self.fraction[x, y, z] = 0.0
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]),
                                                                                  direction)
            elif x > self.nx / 2:
                self.flag[x, y, z] = ti.u8(0)
                self.mass[x, y, z] = 1.0
                self.fraction[x, y, z] = 1.0
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]),
                                                                                  direction)
            elif x < self.nx / 2:
                self.flag[x, y, z] = ti.u8(2)
                self.mass[x, y, z] = 0.0
                self.fraction[x, y, z] = 0.0
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]),
                                                                                  direction)
            else:
                self.flag[x, y, z] = ti.u8(1)
                self.mass[x, y, z] = 0.5
                self.fraction[x, y, z] = 0.5
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]),
                                                                                  direction)

        return

    @ti.kernel
    def get_display_var(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        for x, y in ti.ndrange(self.nx, self.ny):
            sub_m = 0.
            for z in range(self.nz):
                density = self.get_density(f_x_bank, x, y, z)
                velocity = self.get_velocity(f_x_bank, x, y, z, density)
                if self.get_lower_flag(x, y, z) > 1:
                    density = 0.0
                    velocity = ti.Vector([0., 0., 0.])
                if self.get_lower_flag(x, y, z) == 1:
                    density = 1.0
                    velocity = ti.Vector([1., 1., 1.])
                if self.flag[x, y, z] == 0:
                    sub_m = z
                # sub_m += self.flag[x, y, z] < 2
                # sub_m += self.get_lower_flag(x, y, z) == 1
                # sub_m += self.mass[x, y, z]
                # sub_m += ti.abs(self.mass[x, y, z] - density)
                # sub_m += self.fraction[x, y, z]
                # sub_m += density
                # sub_m += ti.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2) * 10
            self.display_var[x, y] = sub_m * 1.0 / 32
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

            if time_record - last_print_time >= 1.0:
                last_print_time = time_record
                self.get_display_var()
                print(time_record)
                # img = cm.plasma(self.display_var.to_numpy() / 0.15)
                img = self.display_var
                gui.set_image(img)
                gui.show()


simulator_obj = LbmD2Q9FreeSurface(64, 64, 32)
simulator_obj.solve()

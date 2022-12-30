import mcubes
import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, offline_cache=True, device_memory_fraction=0.8)

eq_v_weight_d3q19 = ti.field(ti.f32, shape=19)
lattice_vector_d3q19 = ti.Vector.field(3, ti.i8, shape=19)
marching_cube_p_offset = ti.Vector.field(3, ti.f32, shape=8)
marching_cube_edge_lut = ti.field(ti.u16, shape=128)
marching_cube_triangle_lut = ti.field(ti.u8, shape=1920)
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
np_arr = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]],
                  dtype=np.float32)
marching_cube_p_offset.from_numpy(np_arr)
np_arr = np.array(
    [0x000, 0x109, 0x203, 0x30A, 0x406, 0x50F, 0x605, 0x70C, 0x80C, 0x905, 0xA0F, 0xB06, 0xC0A, 0xD03, 0xE09, 0xF00,
     0x190, 0x099, 0x393, 0x29A, 0x596, 0x49F, 0x795, 0x69C, 0x99C, 0x895, 0xB9F, 0xA96, 0xD9A, 0xC93, 0xF99, 0xE90,
     0x230, 0x339, 0x033, 0x13A, 0x636, 0x73F, 0x435, 0x53C, 0xA3C, 0xB35, 0x83F, 0x936, 0xE3A, 0xF33, 0xC39, 0xD30,
     0x3A0, 0x2A9, 0x1A3, 0x0AA, 0x7A6, 0x6AF, 0x5A5, 0x4AC, 0xBAC, 0xAA5, 0x9AF, 0x8A6, 0xFAA, 0xEA3, 0xDA9, 0xCA0,
     0x460, 0x569, 0x663, 0x76A, 0x066, 0x16F, 0x265, 0x36C, 0xC6C, 0xD65, 0xE6F, 0xF66, 0x86A, 0x963, 0xA69, 0xB60,
     0x5F0, 0x4F9, 0x7F3, 0x6FA, 0x1F6, 0x0FF, 0x3F5, 0x2FC, 0xDFC, 0xCF5, 0xFFF, 0xEF6, 0x9FA, 0x8F3, 0xBF9, 0xAF0,
     0x650, 0x759, 0x453, 0x55A, 0x256, 0x35F, 0x055, 0x15C, 0xE5C, 0xF55, 0xC5F, 0xD56, 0xA5A, 0xB53, 0x859, 0x950,
     0x7C0, 0x6C9, 0x5C3, 0x4CA, 0x3C6, 0x2CF, 0x1C5, 0x0CC, 0xFCC, 0xEC5, 0xDCF, 0xCC6, 0xBCA, 0xAC3, 0x9C9, 0x8C0],
    dtype=np.int32)
marching_cube_edge_lut.from_numpy(np_arr)
np_arr = np.array(
    [255, 255, 255, 255, 255, 255, 255, 15, 56, 255, 255, 255, 255, 255, 255, 16, 249, 255, 255, 255, 255, 255, 31, 56,
     137, 241, 255, 255, 255, 255, 33, 250, 255, 255, 255, 255, 255, 15, 56, 33, 250, 255, 255, 255, 255, 41, 10, 146,
     255, 255, 255, 255, 47, 56, 162, 168, 137, 255, 255, 255, 179, 242, 255, 255, 255, 255, 255, 15, 43, 184, 240, 255,
     255, 255, 255, 145, 32, 179, 255, 255, 255, 255, 31, 43, 145, 155, 184, 255, 255, 255, 163, 177, 58, 255, 255, 255,
     255, 15, 26, 128, 138, 171, 255, 255, 255, 147, 48, 155, 171, 249, 255, 255, 159, 168, 138, 251, 255, 255, 255,
     255, 116, 248, 255, 255, 255, 255, 255, 79, 3, 55, 244, 255, 255, 255, 255, 16, 137, 116, 255, 255, 255, 255, 79,
     145,
     116, 113, 19, 255, 255, 255, 33, 138, 116, 255, 255, 255, 255, 63, 116, 3, 20, 162, 255, 255, 255, 41, 154, 32, 72,
     247, 255, 255, 47, 154, 146, 39, 55, 151, 244, 255, 72, 55, 43, 255, 255, 255, 255, 191, 116, 43, 36, 64,
     255, 255, 255, 9, 129, 116, 50, 251, 255, 255, 79, 183, 73, 155, 43, 41, 241, 255, 163, 49, 171, 135, 244, 255,
     255, 31, 171, 65, 27, 64, 183, 244, 255, 116, 152, 176, 185, 186, 48, 255, 79, 183, 180, 153, 171, 255, 255, 255,
     89, 244, 255, 255, 255, 255, 255, 159, 69, 128, 243, 255, 255, 255, 255, 80, 20, 5, 255, 255, 255, 255, 143, 69,
     56, 53, 81, 255, 255, 255, 33, 154, 69, 255, 255, 255, 255, 63, 128, 33, 74, 89, 255, 255, 255, 37, 90, 36,
     4, 242, 255, 255, 47, 90, 35, 53, 69, 67, 248, 255, 89, 36, 179, 255, 255, 255, 255, 15, 43, 128, 75, 89, 255, 255,
     255, 80, 4, 81, 50, 251, 255, 255, 47, 81, 82, 40, 184, 132, 245, 255, 58, 171, 49, 89, 244, 255,
     255, 79, 89, 128, 129, 26, 184, 250, 255, 69, 80, 176, 181, 186, 48, 255, 95, 132, 133, 170, 184, 255, 255, 255,
     121, 88, 151, 255, 255, 255, 255, 159, 3, 89, 83, 55, 255, 255, 255, 112, 8, 113, 81, 247, 255, 255, 31, 53,
     83, 247, 255, 255, 255, 255, 121, 152, 117, 26, 242, 255, 255, 175, 33, 89, 80, 3, 117, 243, 255, 8, 130, 82, 88,
     167, 37, 255, 47, 90, 82, 51, 117, 255, 255, 255, 151, 117, 152, 179, 242, 255, 255, 159, 117, 121, 146, 2,
     114, 251, 255, 50, 11, 129, 113, 24, 117, 255, 191, 18, 27, 119, 81, 255, 255, 255, 89, 136, 117, 26, 163, 179,
     255, 95, 7, 5, 121, 11, 1, 186, 10, 171, 176, 48, 90, 128, 112, 117, 176, 90, 183, 245, 255, 255, 255, 255,
     106, 245, 255, 255, 255, 255, 255, 15, 56, 165, 246, 255, 255, 255, 255, 9, 81, 106, 255, 255, 255, 255, 31, 56,
     145, 88, 106, 255, 255, 255, 97, 37, 22, 255, 255, 255, 255, 31, 86, 33, 54, 128, 255, 255, 255, 105, 149, 96,
     32, 246, 255, 255, 95, 137, 133, 82, 98, 35, 248, 255, 50, 171, 86, 255, 255, 255, 255, 191, 128, 43, 160, 86, 255,
     255, 255, 16, 41, 179, 165, 246, 255, 255, 95, 106, 145, 146, 43, 137, 251, 255, 54, 107, 53, 21, 243, 255,
     255, 15, 184, 176, 5, 21, 181, 246, 255, 179, 6, 99, 96, 5, 149, 255, 111, 149, 150, 187, 137, 255, 255, 255, 165,
     70, 135, 255, 255, 255, 255, 79, 3, 116, 99, 165, 255, 255, 255, 145, 80, 106, 72, 247, 255, 255, 175, 86,
     145, 23, 55, 151, 244, 255, 22, 98, 21, 116, 248, 255, 255, 31, 82, 37, 54, 64, 67, 247, 255, 72, 151, 80, 96, 5,
     98, 255, 127, 147, 151, 52, 146, 149, 38, 150, 179, 114, 72, 106, 245, 255, 255, 95, 106, 116, 66, 2,
     114, 251, 255, 16, 73, 135, 50, 91, 106, 255, 159, 18, 185, 146, 180, 183, 84, 106, 72, 55, 91, 83, 81, 107, 255,
     95, 177, 181, 22, 176, 183, 4, 180, 80, 9, 86, 48, 182, 54, 72, 103, 149, 150, 75, 151, 183, 249, 255,
     74, 105, 164, 255, 255, 255, 255, 79, 106, 148, 10, 56, 255, 255, 255, 10, 161, 6, 70, 240, 255, 255, 143, 19, 24,
     134, 70, 22, 250, 255, 65, 25, 66, 98, 244, 255, 255, 63, 128, 33, 41, 148, 98, 244, 255, 32, 68, 98,
     255, 255, 255, 255, 143, 35, 40, 68, 98, 255, 255, 255, 74, 169, 70, 43, 243, 255, 255, 15, 40, 130, 75, 169, 164,
     246, 255, 179, 2, 97, 96, 100, 161, 255, 111, 20, 22, 74, 24, 18, 139, 27, 105, 148, 99, 25, 179, 54,
     255, 143, 27, 24, 176, 22, 25, 100, 20, 179, 54, 6, 96, 244, 255, 255, 111, 132, 107, 248, 255, 255, 255, 255, 167,
     118, 168, 152, 250, 255, 255, 15, 55, 160, 7, 169, 118, 250, 255, 106, 23, 122, 113, 24, 8, 255, 175, 118,
     122, 17, 55, 255, 255, 255, 33, 22, 134, 129, 137, 118, 255, 47, 150, 146, 97, 151, 144, 115, 147, 135, 112, 96, 6,
     242, 255, 255, 127, 35, 118, 242, 255, 255, 255, 255, 50, 171, 134, 138, 137, 118, 255, 47, 112, 114, 11, 121,
     118, 154, 122, 129, 16, 135, 161, 103, 167, 50, 187, 18, 27, 167, 22, 118, 241, 255, 152, 134, 118, 25, 182, 54,
     49, 6, 25, 107, 247, 255, 255, 255, 255, 135, 112, 96, 179, 176, 6, 255, 127, 107, 255, 255, 255, 255, 255, 255,
     103, 251, 255, 255, 255, 255, 255, 63, 128, 123, 246, 255, 255, 255, 255, 16, 185, 103, 255, 255, 255, 255, 143,
     145, 56, 177, 103, 255, 255, 255, 26, 98, 123, 255, 255, 255, 255, 31, 162, 3, 104, 123, 255, 255, 255, 146, 32,
     154,
     182, 247, 255, 255, 111, 123, 162, 163, 56, 154, 248, 255, 39, 99, 114, 255, 255, 255, 255, 127, 128, 103, 96, 2,
     255, 255, 255, 114, 38, 115, 16, 249, 255, 255, 31, 38, 129, 22, 137, 120, 246, 255, 122, 166, 113, 49, 247, 255,
     255, 175, 103, 113, 26, 120, 1, 248, 255, 48, 7, 167, 160, 105, 122, 255, 127, 166, 167, 136, 154, 255, 255, 255,
     134, 180, 104, 255, 255, 255, 255, 63, 182, 3, 6, 100, 255, 255, 255, 104, 139, 100, 9, 241, 255, 255, 159, 100,
     105, 147, 19, 59, 246, 255, 134, 100, 139, 162, 241, 255, 255, 31, 162, 3, 11, 182, 64, 246, 255, 180, 72, 182, 32,
     41, 154, 255, 175, 57, 58, 146, 52, 59, 70, 54, 40, 131, 36, 100, 242, 255, 255, 15, 36, 100, 242, 255,
     255, 255, 255, 145, 32, 67, 66, 70, 131, 255, 31, 73, 65, 34, 100, 255, 255, 255, 24, 131, 22, 72, 102, 26, 255,
     175, 1, 10, 102, 64, 255, 255, 255, 100, 67, 131, 166, 3, 147, 154, 163, 73, 166, 244, 255, 255, 255, 255,
     148, 117, 182, 255, 255, 255, 255, 15, 56, 148, 181, 103, 255, 255, 255, 5, 81, 4, 103, 251, 255, 255, 191, 103,
     56, 52, 69, 19, 245, 255, 89, 164, 33, 103, 251, 255, 255, 111, 123, 33, 10, 56, 148, 245, 255, 103, 91, 164,
     36, 74, 32, 255, 63, 132, 83, 52, 82, 90, 178, 103, 39, 115, 38, 69, 249, 255, 255, 159, 69, 128, 6, 38, 134, 247,
     255, 99, 50, 103, 81, 80, 4, 255, 111, 130, 134, 39, 129, 132, 21, 133, 89, 164, 97, 113, 22, 115,
     255, 31, 166, 113, 22, 112, 120, 144, 69, 4, 74, 90, 48, 106, 122, 115, 122, 166, 167, 88, 164, 132, 250, 255, 150,
     101, 155, 139, 249, 255, 255, 63, 182, 96, 3, 101, 144, 245, 255, 176, 8, 181, 16, 85, 182, 255, 111, 59,
     54, 85, 19, 255, 255, 255, 33, 154, 181, 185, 184, 101, 255, 15, 59, 96, 11, 105, 101, 25, 162, 139, 181, 101, 8,
     165, 37, 32, 101, 59, 54, 37, 58, 90, 243, 255, 133, 89, 130, 101, 50, 40, 255, 159, 101, 105, 0, 38,
     255, 255, 255, 81, 24, 8, 101, 56, 40, 38, 24, 101, 18, 246, 255, 255, 255, 255, 49, 22, 166, 131, 86, 150, 152,
     166, 1, 10, 150, 5, 101, 240, 255, 48, 88, 166, 255, 255, 255, 255, 175, 101, 255, 255, 255, 255, 255, 255,
     91, 122, 181, 255, 255, 255, 255, 191, 165, 123, 133, 3, 255, 255, 255, 181, 87, 186, 145, 240, 255, 255, 175, 87,
     186, 151, 24, 56, 241, 255, 27, 178, 23, 87, 241, 255, 255, 15, 56, 33, 23, 87, 39, 251, 255, 121, 149, 114,
     9, 34, 123, 255, 127, 37, 39, 91, 41, 35, 152, 40, 82, 42, 83, 115, 245, 255, 255, 143, 2, 88, 130, 87, 42, 245,
     255, 9, 81, 58, 53, 55, 42, 255, 159, 40, 41, 129, 39, 42, 117, 37, 49, 53, 87, 255, 255, 255,
     255, 15, 120, 112, 17, 87, 255, 255, 255, 9, 147, 83, 53, 247, 255, 255, 159, 120, 149, 247, 255, 255, 255, 255,
     133, 84, 138, 186, 248, 255, 255, 95, 64, 181, 80, 186, 59, 240, 255, 16, 137, 164, 168, 171, 84, 255, 175, 75,
     74, 181, 67, 73, 49, 65, 82, 33, 88, 178, 72, 133, 255, 15, 180, 176, 67, 181, 178, 81, 177, 32, 5, 149, 178, 69,
     133, 139, 149, 84, 178, 243, 255, 255, 255, 255, 82, 58, 37, 67, 53, 72, 255, 95, 42, 37, 68, 2,
     255, 255, 255, 163, 50, 165, 131, 69, 133, 16, 89, 42, 37, 20, 41, 73, 242, 255, 72, 133, 53, 83, 241, 255, 255,
     15, 84, 1, 245, 255, 255, 255, 255, 72, 133, 53, 9, 5, 83, 255, 159, 84, 255, 255, 255, 255, 255, 255,
     180, 71, 185, 169, 251, 255, 255, 15, 56, 148, 151, 123, 169, 251, 255, 161, 27, 75, 65, 112, 180, 255, 63, 65, 67,
     24, 74, 71, 171, 75, 180, 151, 75, 41, 155, 33, 255, 159, 71, 185, 151, 177, 178, 1, 56, 123, 180, 36,
     66, 240, 255, 255, 191, 71, 75, 130, 67, 35, 244, 255, 146, 42, 151, 50, 119, 148, 255, 159, 122, 121, 164, 114,
     120, 32, 112, 115, 58, 42, 71, 26, 10, 4, 26, 42, 120, 244, 255, 255, 255, 255, 148, 65, 113, 23, 243, 255,
     255, 79, 25, 20, 7, 24, 120, 241, 255, 4, 115, 52, 255, 255, 255, 255, 79, 120, 255, 255, 255, 255, 255, 255, 169,
     168, 139, 255, 255, 255, 255, 63, 144, 147, 187, 169, 255, 255, 255, 16, 10, 138, 168, 251, 255, 255, 63, 161,
     59, 250, 255, 255, 255, 255, 33, 27, 155, 185, 248, 255, 255, 63, 144, 147, 27, 146, 178, 249, 255, 32, 139, 176,
     255, 255, 255, 255, 63, 178, 255, 255, 255, 255, 255, 255, 50, 40, 168, 138, 249, 255, 255, 159, 42, 144, 242, 255,
     255, 255, 255, 50, 40, 168, 16, 24, 138, 255, 31, 42, 255, 255, 255, 255, 255, 255, 49, 152, 129, 255, 255, 255,
     255, 15, 25, 255, 255, 255, 255, 255, 255, 48, 248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
     255])
marching_cube_triangle_lut.from_numpy(np_arr)

p = ti.Vector.field(3, ti.f32, shape=8)
p[0] = marching_cube_p_offset[0]
p[1] = marching_cube_p_offset[1]
p[2] = marching_cube_p_offset[2]
p[3] = marching_cube_p_offset[3]
p[4] = marching_cube_p_offset[4]
p[5] = marching_cube_p_offset[5]
p[6] = marching_cube_p_offset[6]
p[7] = marching_cube_p_offset[7]


@ti.data_oriented
class LbmD3Q19FreeSurface:

    def __init__(self, x_size: int, y_size: int, z_size: int, render_time_step: int):
        # self.triangle_list = ti.field(ti.f32)
        # ti.root.dynamic(ti.i, x_size * y_size * z_size * 12 * 3 * 3, chunk_size=128).place(self.triangle_list)
        self.render_time_step = render_time_step
        self.nx = x_size
        self.ny = y_size
        self.nz = z_size
        self.f_x = ti.field(ti.f32, shape=(2, self.nx, self.ny, self.nz, 19))
        self.mass = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
        self.mass_exchange = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
        self.fraction = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
        self.display_var = ti.field(ti.f32, shape=(self.nx, self.ny))

        # Viscosity define
        self.niu = 0.2
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
        self.next_timestep = ti.field(ti.f32, shape=())
        self.next_timestep[None] = 1.0

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
                for direction in range(19):
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
                for direction in range(19):
                    if direction == 9:
                        continue
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
            for direction in range(19):
                if direction == 9:
                    continue
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
                for direction in range(19):
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
                for direction in range(19):
                    if direction == 9:
                        continue
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
            cur_density = self.get_density(f_x_bank, x, y, z)
            self.mass[x, y, z] = self.mass[x, y, z] + self.mass_exchange[x, y, z]
            self.fraction[x, y, z] = self.mass[x, y, z] / cur_density
            self.mass_exchange[x, y, z] = 0
        return

    @ti.kernel
    def adapt_timestep_1(self, allow_increase: ti.u8):
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
        if maximum_velocity > upper_limit:
            new_time_step *= multiplier
        elif maximum_velocity < lower_limit and allow_increase != 0 and old_time_step < self.render_time_step:
            new_time_step /= multiplier
        self.next_timestep[None] = new_time_step

    @ti.kernel
    def adapt_timestep_2(self):
        f_x_bank = self.f_x_bank_sel[None]
        old_time_step = self.timestep[None]
        new_time_step = self.next_timestep[None]
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
            for direction in range(19):
                old_feq = self.cal_feq(old_density, old_velocity, direction)
                new_feq = self.cal_feq(new_density, new_velocity, direction)
                feq_ratio = new_feq / old_feq
                self.f_x[f_x_bank, x, y, z, direction] = feq_ratio * (
                        old_feq + tau_radio * (self.f_x[f_x_bank, x, y, z, direction] - old_feq))

            if self.get_lower_flag(x, y, z) == 1:
                self.mass[x, y, z] = self.mass[x, y, z] * (old_density / new_density)
                self.fraction[x, y, z] = self.mass[x, y, z] / new_density

        print('timestep changed from', self.timestep[None], 'to', new_time_step)
        self.timestep[None] = new_time_step
        self.tau[None] = new_tau
        self.inv_tau[None] = new_inv_tau
        return

    @ti.func
    def no_slip_single(self, bank, x, y, z):
        for direction in range(19):
            n_x, n_y, n_z = self.get_neighbor_index_p(x, y, z, direction)
            if not (n_x < 0 or n_y < 0 or n_z < 0 or n_x >= self.nx or n_y >= self.ny or n_z >= self.nz):
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
                for direction in range(19):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
            elif z < self.nz / 8 or (x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - 80) ** 2 < 400:
                self.flag[x, y, z] = ti.u8(0)
                self.mass[x, y, z] = 1.0
                self.fraction[x, y, z] = 1.0
                for direction in range(19):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
            elif (x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - 80) ** 2 < 450:
                self.flag[x, y, z] = ti.u8(1)
                self.mass[x, y, z] = 0.5
                self.fraction[x, y, z] = 0.5
                for direction in range(19):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)

            elif z > self.ny / 8:
                self.flag[x, y, z] = ti.u8(2)
                self.mass[x, y, z] = 0.0
                self.fraction[x, y, z] = 0.0
                for direction in range(19):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
            else:
                self.flag[x, y, z] = ti.u8(1)
                self.mass[x, y, z] = 0.5
                self.fraction[x, y, z] = 0.5
                for direction in range(19):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)

        return

    @ti.kernel
    def init_water(self):
        f_x_bank = self.f_x_bank_sel[None]
        f_x_bank_next = (f_x_bank + 1) & 1
        for x, y, z in ti.ndrange(self.nx, self.ny, self.nz):
            if x <= 0 or x == self.nx - 1 or y == 0 or y == self.ny - 1 or z == 0 or z == self.nz - 1:
                self.flag[x, y, z] = ti.u8(8)
                self.mass[x, y, z] = 0.0
                self.fraction[x, y, z] = 0.0
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]),
                                                                               direction)
            elif x > self.nx / 2:
                self.flag[x, y, z] = ti.u8(0)
                self.mass[x, y, z] = 1.0
                self.fraction[x, y, z] = 1.0
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]),
                                                                               direction)
            elif x < self.nx / 2:
                self.flag[x, y, z] = ti.u8(2)
                self.mass[x, y, z] = 0.0
                self.fraction[x, y, z] = 0.0
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]),
                                                                               direction)
            else:
                self.flag[x, y, z] = ti.u8(1)
                self.mass[x, y, z] = 0.5
                self.fraction[x, y, z] = 0.5
                for direction in ti.static(range(19)):
                    self.f_x[f_x_bank, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]), direction)
                    self.f_x[f_x_bank_next, x, y, z, direction] = self.cal_feq(1.0, ti.Vector([0., 0., 0.]),
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
        # self.init_drop()
        self.init_water()
        self.get_display_var()
        gui.set_image(self.display_var)
        gui.show()
        time_record = 0.0
        last_print_time = 0.0
        increase_next = 0.0
        i = -1
        render_cnt = 0
        while True:
            i = i + 1
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

            self.adapt_timestep_1(allow_increase)

            if self.timestep[None] != self.next_timestep[None]:
                self.adapt_timestep_2()

            if step_size_before > self.timestep[None]:
                increase_next = i + 200

            self.boundary_condition()
            if i == 0:
                print(9, i)

            if time_record - last_print_time >= self.render_time_step:
                file_name = '.\\output\\test_%d.obj' % render_cnt
                # np.save(file_name, self.fraction.to_numpy())
                # smoothed_sphere = mcubes.smooth(self.fraction.to_numpy())
                smoothed_sphere = self.fraction.to_numpy()
                vertices, triangles = mcubes.marching_cubes(smoothed_sphere, 0)
                mcubes.export_obj(vertices, triangles, file_name)
                print(file_name)
                last_print_time = time_record
                self.get_display_var()
                print(time_record)
                # img = cm.plasma(self.display_var.to_numpy() / 0.15)
                img = self.display_var
                gui.set_image(img)
                gui.show()
                render_cnt += 1


simulator_obj = LbmD3Q19FreeSurface(128, 128, 128, 10)
simulator_obj.solve()

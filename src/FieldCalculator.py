import numpy as np
import FieldOperator as fo


def field(xy_max, n_xy, objects, t=-1, nabla='', function='', plane='xy'):
    x, y = mesh(xy_max, n_xy)
    x_field, y_field, z_field = np.zeros_like(x), np.zeros_like(y), np.zeros_like(x)
    for object in objects:
        for i in range(len(x)):
            for j in range(len(y)):
                field = field_point(x[i][j], y[i][j], plane,
                                    object, nabla, function, t, xy_max, n_xy)
                x_field[i][j] += field[0]
                y_field[i][j] += field[1]
                z_field[i][j] += field[2]
    return [x_field, y_field, z_field]


def field_point(x, y, plane, object, nabla, function, t, xy_max, n_xy):
    f = getattr(object, function)
    if nabla == '':
        if t >= 0:
            if plane == 'xz':
                field = np.real(f(x, 0., y, t))
                tmp = field_round(field[0], field[2], xy_max, n_xy, object)
                field[0] = tmp[0]
                field[2] = tmp[1]
            elif plane == 'yz':
                field = np.real(f(0., x, y, t))
            else:
                field = np.real(f(x, y, 0., t))
                tmp = field_round(field[0], field[1], xy_max, n_xy, object)
                field[0] = tmp[0]
                field[1] = tmp[1]
        else:
            if plane == 'xz':
                field = f(x, 0., y)
            elif plane == 'yz':
                field = f(0., x, y)
            else:
                field = f(x, y, 0.)
    else:
        operator = getattr(fo, nabla)
        if plane == 'xz':
            field = operator(x, 0., y, f)
        elif plane == 'yz':
            field = operator(0., x, y, f)
        else:
            field = operator(x, y, 0., f)
    return field


def mesh(xy_max, n_xy):
    xx, yy = np.meshgrid(np.linspace(-xy_max, xy_max, n_xy),
                         np.linspace(-xy_max, xy_max, n_xy))
    return xx, yy


def field3d(xyz_max, n_xyz, objects, t=-1, nabla='', function=''):
    x, y, z = mesh3d(xyz_max, n_xyz)
    x_field, y_field, z_field = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    for object in objects:
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    field = field_point3d(x[i][j][k], y[i][j][k], z[i][j][k],
                                          object, nabla, function, t)
                    x_field[i][j][k] += field[0]
                    y_field[i][j][k] += field[1]
                    z_field[i][j][k] += field[2]
    return [x_field, y_field, z_field]


def field_point3d(x, y, z, object, nabla, function, t):
    f = getattr(object, function)
    if nabla == '':
        if t >= 0:
            field = np.real(f(x, y, z, t))
        else:
            field = f(x, y, z)
    else:
        operator = getattr(fo, nabla)
        field = operator(x, y, z, f)
    return field


def mesh3d(xyz_max, n_xyz):
    xxx, yyy, zzz = np.meshgrid(np.linspace(-xyz_max, xyz_max, n_xyz),
                                np.linspace(-xyz_max, xyz_max, n_xyz),
                                np.linspace(-xyz_max, xyz_max, n_xyz),
                                indexing='ij')
    return xxx, yyy, zzz


def field_round(f_x, f_y, xy_max, n_xy, object):
    unit = (xy_max / n_xy) * object.get_factor()
    f_xy_norm = np.sqrt(f_x ** 2 + f_y ** 2)
    if f_xy_norm > unit:
        f_x = (f_x / f_xy_norm) * unit
        f_y = (f_y / f_xy_norm) * unit
    return f_x, f_y


def radius_unit(xy_max, n_xy):
    x, y = mesh(xy_max, n_xy)
    phi = np.arctan2(y, x)
    e_r = np.array([np.cos(phi), np.sin(phi)])
    return e_r


def phi_unit(xy_max, n_xy):
    x, y = mesh(xy_max, n_xy)
    phi = np.arctan2(y, x)
    e_phi = np.array([-np.sin(phi), np.cos(phi)])
    return e_phi

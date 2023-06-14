import numpy as np
from bresenham import bresenham
from utils import draw_line

def generate_scene(scene_height, scene_width, max_shape):
    scene = np.zeros((scene_height, scene_width), dtype=np.complex64)
    nr_of_shapes = np.random.randint(1, max_shape+1) # nr of shapes to create, maximum max_shape
    nr_of_shapes = max_shape
    shape_types = [0,1,2,3]
    for i in range(nr_of_shapes):
        np.random.seed() # reset seed
        # shape_type = np.random.randint(5)
        shape_type = np.random.choice(6, p=[0.17, 0.17, 0.17, 0.17, 0.17, 0.15])
        # shape_type=3
        shape_type = shape_types[i]

        row = np.random.randint(scene_height - 5) # generate row position
        col = np.random.randint(scene_width - 5) # generate col position
        l = np.random.randint(10, np.floor(np.minimum(scene_height, scene_width)/2)) # generate size of shape

        # boundary conditions
        if row + l >= scene_height:
           row = row - l 
    
        if col + l >= scene_width:
           col = col - l

        # randomise intensity inside the unit circle
        radius = np.random.rand() # random radius 0-1
        phi = np.linspace(0, 2*np.pi, 1000)
        phi = phi[np.random.randint(np.max(phi.shape))]; # random angle 0-2pi
        
        real_part = radius * np.cos(phi)
        imag_part = radius * np.sin(phi)
        intensity = real_part + imag_part * 1j # random complex intensity

        if shape_type == 0: # rectangle
            scene[row, col:col+l+1] = intensity
            scene[row:row+l+1, col] = intensity
            scene[row+l, col:col+l+1] = intensity
            scene[row:row+l+1, col+l] = intensity

        elif shape_type == 1: # circle
            x = np.arange(scene_width)
            y = np.arange(scene_height)
            [columnsInImage, rowsInImage] = np.meshgrid(x, y)
            circle1 = (rowsInImage - row)**2 + (columnsInImage - col)**2 <= l**2
            circle2 = (rowsInImage - row)**2 + (columnsInImage - col)**2 <= (l-1)**2
            circle3 = circle1.astype(int) - circle2.astype(int)
            circle3 = circle3 * intensity
            scene = scene + circle3

        elif shape_type == 2: # triangle
            p1 = np.array([row, row + np.floor(col/2)], dtype=int) # top of triangle
            p2 = np.array([row + l, col], dtype=int)               # bottom left
            p3 = np.array([row + l, col + l], dtype=int)           # bottom right

            # dumb workaround since bresenham yields coordinates one pair at a time instead of returning all pairs
            coords_generator = bresenham(p1[1],p1[0],p2[1],p2[0]) # top-botleft
            coords = np.empty((0, 2), dtype=int)
            for item in coords_generator:
                coords = np.append(coords, [item], axis=0)
            scene = draw_line(scene, coords[:, 0], coords[:, 1], intensity)  # coords[:, 0] contains the x-coords and coords[:, 1] contains the y-coords

            coords_generator = bresenham(p2[1],p2[0],p3[1],p3[0]) # botleft-botright
            coords = np.empty((0, 2), dtype=int)
            for item in coords_generator:
                coords = np.append(coords, [item], axis=0)
            scene = draw_line(scene, coords[:, 0], coords[:, 1], intensity)

            coords_generator = bresenham(p1[1],p1[0],p3[1],p3[0]) # top-botright
            coords = np.empty((0, 2), dtype=int)
            for item in coords_generator:
                coords = np.append(coords, [item], axis=0)
            scene = draw_line(scene, coords[:, 0], coords[:, 1], intensity)

        elif shape_type == 3: # line between 2 random points
            p1 = np.array([row, col])
            p2 = np.array([np.random.randint(scene_height), np.random.randint(scene_width)])
            coords_generator = bresenham(p1[1],p1[0],p2[1],p2[0])
            coords = np.empty((0, 2), dtype=int)
            for item in coords_generator:
                coords = np.append(coords, [item], axis=0)
            scene = draw_line(scene, coords[:, 0], coords[:, 1], intensity)

        elif shape_type == 4: # weird 4-point object
            nr_of_points = 4
            rows = []
            cols = []

            for j in range(nr_of_points): # randomise 4 points
                row = np.random.randint(scene_height) # generate row position
                col = np.random.randint(scene_width) # generate col position
                rows.append(row)
                cols.append(col)

            # random intensity
            radius = np.random.rand() # random radius 0-1
            phi = np.linspace(0, 2*np.pi, 1000)
            phi = phi[np.random.randint(len(phi))] # random angle 0-2pi

            real_part = radius * np.cos(phi)
            imag_part = radius * np.sin(phi)
            intensity = real_part + imag_part * 1j # random complex intensity
            
            # draw lines from p1->p2->p3->p4
            for j in range(nr_of_points - 1):
                coords_generator = bresenham(cols[j],rows[j],cols[j+1],rows[j+1])
                coords = np.empty((0, 2), dtype=int)
                for item in coords_generator:
                    coords = np.append(coords, [item], axis=0)
                scene = draw_line(scene, coords[:, 0], coords[:, 1], intensity)

            # draw line from p4->p1
            coords_generator = bresenham(cols[3],rows[3],cols[0],rows[0])
            coords = np.empty((0, 2), dtype=int)
            for item in coords_generator:
                coords = np.append(coords, [item], axis=0)
            scene = draw_line(scene, coords[:, 0], coords[:, 1], intensity)

        else: # random points
            nr_of_points = np.random.randint(1, 10)            
            
            for j in range(nr_of_points):
                np.random.seed() # reset seed
                # random position
                row = np.random.randint(scene_height//8, scene_height - scene_height//8) # generate row position
                col = np.random.randint(scene_width//8, scene_width - scene_width//8) # generate col position
                # random intensity
                radius = np.random.rand() # random radius 0-1
                phi = np.linspace(0, 2*np.pi, 1000)
                phi = phi[np.random.randint(len(phi))] # random angle 0-2pi

                real_part = radius * np.cos(phi)
                imag_part = radius * np.sin(phi)
                intensity = real_part + imag_part * 1j # random complex intensity

                scene[row, col] = intensity

    return scene


def generate_line(scene_height, scene_width):
    scene = np.zeros((scene_height, scene_width), dtype=np.complex64)    

    cols = [70, 130]
    row = scene_height//2
    intensity = 0.1 + 0.4j

    # draw line from p4->p1
    coords_generator = bresenham(cols[1],row,cols[0],row)
    coords = np.empty((0, 2), dtype=int)
    for item in coords_generator:
        coords = np.append(coords, [item], axis=0)
    scene = draw_line(scene, coords[:, 0], coords[:, 1], intensity)

    return scene


def generate_point_scene(scene_height, scene_width):
    scene = np.zeros((scene_height, scene_width), dtype=np.complex64)
    nr_of_points = np.random.randint(1, 15)

    phis = np.linspace(0, 2*np.pi, 1000)
    for j in range(nr_of_points):
        np.random.seed() # reset seed
        # random position        
        row = np.random.randint(scene_height//8, scene_height - scene_height//8) # generate row position
        col = np.random.randint(scene_width//8, scene_width - scene_width//8) # generate col position
        # random intensity
        radius = np.random.rand() # random radius 0-1
        
        phi = phis[np.random.randint(len(phis))] # random angle 0-2pi

        real_part = radius * np.cos(phi)
        imag_part = radius * np.sin(phi)
        intensity = real_part + imag_part * 1j # random complex intensity

        scene[row, col] = intensity

    return scene


def generate_point_scene_same_row(scene_height, scene_width):
    scene = np.zeros((scene_height, scene_width), dtype=np.complex64)
    nr_of_points = np.random.randint(2, 5)

    phis = np.linspace(0, 2*np.pi, 1000)
    row = np.random.randint(scene_height//8, scene_height - scene_height//8) # generate row position
    for j in range(nr_of_points):
        np.random.seed() # reset seed
        # random position        
        col = np.random.randint(scene_width//8, scene_width - scene_width//8) # generate col position
        # random intensity
        radius = np.random.rand() # random radius 0-1
        
        phi = phis[np.random.randint(len(phis))] # random angle 0-2pi

        real_part = radius * np.cos(phi)
        imag_part = radius * np.sin(phi)
        intensity = real_part + imag_part * 1j # random complex intensity

        scene[row, col] = intensity

    return scene


def generate_point_scene_same_row_closer(scene_height, scene_width):
    phis = np.linspace(0, 2*np.pi, 1000)
    radius = np.random.rand() # random radius 0-1        
    phi = phis[np.random.randint(len(phis))] # random angle 0-2pi
    real_part = radius * np.cos(phi)
    imag_part = radius * np.sin(phi)
    intensity = real_part + imag_part * 1j # random complex intensity
    scenes = []
    row = scene_height//2
    for j in range(7, 0, -1):
        scene = np.zeros((scene_height, scene_width), dtype=np.complex64)
        np.random.seed() # reset seed
        # random position        
        col1 = scene_width//2 - 2**(j-1)
        col2 = scene_width//2 + 2**(j-1)        

        scene[row, col1] = intensity
        scene[row, col2] = intensity

        scenes.append(scene)

    return scenes


def generate_point_scene_same_row_closer2(scene_height, scene_width, start):
    phis = np.linspace(0, 2*np.pi, 1000)
    radius = np.random.rand() # random radius 0-1        
    phi = phis[np.random.randint(len(phis))] # random angle 0-2pi
    real_part = radius * np.cos(phi)
    imag_part = radius * np.sin(phi)
    intensity = real_part + imag_part * 1j # random complex intensity
    intensity = 0.5 + 0.5j
    scenes = []
    row = scene_height//2
    for j in range(start, 70, -1):
        scene = np.zeros((scene_height, scene_width), dtype=np.complex64)
        np.random.seed() # reset seed
        # random position        
        col1 = scene_width//2
        col2 = scene_width//2 - (j - 1)

        scene[row, col1] = intensity
        scene[row, col2] = intensity

        scenes.append(scene)

    return scenes


def generate_point_scene_same_col(scene_height, scene_width):
    scene = np.zeros((scene_height, scene_width), dtype=np.complex64)
    nr_of_points = np.random.randint(2, 5)

    phis = np.linspace(0, 2*np.pi, 1000)
    
    col = np.random.randint(scene_width//8, scene_width - scene_width//8) # generate col position
    for j in range(nr_of_points):
        np.random.seed() # reset seed
        # random position        
        row = np.random.randint(scene_height//8, scene_height - scene_height//8) # generate row position    
        # random intensity
        radius = np.random.rand() # random radius 0-1
        
        phi = phis[np.random.randint(len(phis))] # random angle 0-2pi

        real_part = radius * np.cos(phi)
        imag_part = radius * np.sin(phi)
        intensity = real_part + imag_part * 1j # random complex intensity

        scene[row, col] = intensity

    return scene


def generate_point_scene_same_col_closer2(scene_height, scene_width, start):
    phis = np.linspace(0, 2*np.pi, 1000)
    radius = np.random.rand() # random radius 0-1        
    phi = phis[np.random.randint(len(phis))] # random angle 0-2pi
    real_part = radius * np.cos(phi)
    imag_part = radius * np.sin(phi)
    intensity = real_part + imag_part * 1j # random complex intensity
    intensity = 0.5 + 0.5j
    scenes = []
    col = scene_width//2
    row1 = scene_height//2
    for j in range(start, 70, -1):
        scene = np.zeros((scene_height, scene_width), dtype=np.complex64)
        np.random.seed() # reset seed
        
        row2 = scene_height//2 - (j - 1)

        scene[row1, col] = intensity
        scene[row2, col] = intensity

        scenes.append(scene)

    return scenes


def generate_point_scene_same_col_closer(scene_height, scene_width):
    phis = np.linspace(0, 2*np.pi, 1000)
    radius = np.random.rand() # random radius 0-1        
    phi = phis[np.random.randint(len(phis))] # random angle 0-2pi
    real_part = radius * np.cos(phi)
    imag_part = radius * np.sin(phi)
    intensity = real_part + imag_part * 1j # random complex intensity
    scenes = []
    col = scene_width//2
    for j in range(7, 0, -1):
        scene = np.zeros((scene_height, scene_width), dtype=np.complex64)
        np.random.seed() # reset seed
        # random position        
        row1 = scene_height//2 - 2**(j-1)
        row2 = scene_height//2 + 2**(j-1)        

        scene[row1, col] = intensity
        scene[row2, col] = intensity

        scenes.append(scene)

    return scenes


def generate_single_point_scene(scene_height, scene_width):
    scene = np.zeros((scene_height, scene_width), dtype=np.complex64)
    row = scene_height//2
    col = scene_width//2
    intensity = 1 + 0j 

    scene[row, col] = intensity

    return scene

def generate_offgrid_point(x_range, y_range, off_mode = "both"):
    x_ind = np.random.randint(1, len(x_range))
    y_ind = np.random.randint(1, len(y_range))
    x_pos = x_range[x_ind-1]
    y_pos = y_range[y_ind-1]
    dx = x_range[-1] - x_range[-2]
    dy = y_range[-1] - y_range[-2]
    x_pos = x_pos - dx/2
    y_pos = y_pos - dy/2
    # random intensity
    radius = np.random.rand() # random radius 0-1
    phi = np.linspace(0, 2*np.pi, 1000)
    phi = phi[np.random.randint(len(phi))] # random angle 0-2pi

    real_part = radius * np.cos(phi)
    imag_part = radius * np.sin(phi)
    intensity = real_part + imag_part * 1j # random complex intensity
    
    return x_pos, y_pos, intensity
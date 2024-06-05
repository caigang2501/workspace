class RubiksCube:
    def __init__(self):
        self.faces = {
            'U': [['W'] * 3 for _ in range(3)],  # 上（白色）
            'D': [['Y'] * 3 for _ in range(3)],  # 下（黄色）
            'F': [['R'] * 3 for _ in range(3)],  # 前（红色）
            'B': [['O'] * 3 for _ in range(3)],  # 后（橙色）
            'L': [['G'] * 3 for _ in range(3)],  # 左（绿色）
            'R': [['B'] * 3 for _ in range(3)],  # 右（蓝色）
        }

    def rotate_face(self, face):
        self.faces[face] = [
            [self.faces[face][2][0], self.faces[face][1][0], self.faces[face][0][0]],
            [self.faces[face][2][1], self.faces[face][1][1], self.faces[face][0][1]],
            [self.faces[face][2][2], self.faces[face][1][2], self.faces[face][0][2]],
        ]
        if face == 'F':
            self._rotate_sides(['U', 'R', 'D', 'L'])
        elif face == 'B':
            self._rotate_sides(['U', 'L', 'D', 'R'])
        elif face == 'L':
            self._rotate_sides(['U', 'F', 'D', 'B'])
        elif face == 'R':
            self._rotate_sides(['U', 'B', 'D', 'F'])
        elif face == 'U':
            self._rotate_sides(['B', 'R', 'F', 'L'])
        elif face == 'D':
            self._rotate_sides(['F', 'R', 'B', 'L'])

    def _rotate_sides(self, sides):
        temp = self.faces[sides[0]][2]
        self.faces[sides[0]][2] = [self.faces[sides[3]][2][2], self.faces[sides[3]][1][2], self.faces[sides[3]][0][2]]
        self.faces[sides[3]][0][2], self.faces[sides[3]][1][2], self.faces[sides[3]][2][2] = \
            self.faces[sides[2]][0][::-1]
        self.faces[sides[2]][0] = [self.faces[sides[1]][0][2], self.faces[sides[1]][1][2], self.faces[sides[1]][2][2]]
        self.faces[sides[1]][0][2], self.faces[sides[1]][1][2], self.faces[sides[1]][2][2] = temp[::-1]

    def print_cube(self):
        for face in 'ULFRBD':
            print(f"{face} face:")
            for row in self.faces[face]:
                print(' '.join(row))
            print()

cube = RubiksCube()
cube.print_cube()
cube.rotate_face('F')
print("After rotating the front face:")
cube.print_cube()

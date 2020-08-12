#Maria Jose Castro Lemus 
#181202
#Graficas por Computadora - 10
#Lab 3: SR3 Models

import struct 
from obj import Obj
from collections import namedtuple
import random

V2 = namedtuple('Vertex2', ['x', 'y'])
V3 = namedtuple('Vertex3', ['x', 'y', 'z'])

def char(c):
    return struct.pack('=c', c.encode('ascii'))
def word(c):
    return struct.pack('=h', c)
def dword(c):
    return struct.pack('=l', c)
def color(r, g, b):
    return bytes([b, g, r])

def sum(v0, v1):
    """
      Input: 2 size 3 vectors
      Output: Size 3 vector with the per element sum
    """
    return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)


def sub(v0, v1):
    """
      Input: 2 size 3 vectors
      Output: Size 3 vector with the per element substraction
    """
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)


def mul(v0, k):
    """
      Input: 2 size 3 vectors
      Output: Size 3 vector with the per element multiplication
    """
    return V3(v0.x * k, v0.y * k, v0.z * k)


def dot(v0, v1):
    """
      Input: 2 size 3 vectors
      Output: Scalar with the dot product
    """
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z


def length(v0):
    """
      Input: 1 size 3 vector
      Output: Scalar with the length of the vector
    """
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5


def norm(v0):
    """
      Input: 1 size 3 vector
      Output: Size 3 vector with the normal of the vector
    """
    v0length = length(v0)

    if not v0length:
        return V3(0, 0, 0)

    return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)


def cross(u, w):
    # print(u, w)
    return V3(
        u.y * w.z - u.z * w.y,
        u.z * w.x - u.x * w.z,
        u.x * w.y - u.y * w.x,
    )


def bbox(*vertices):
    xs = [vertex.x for vertex in vertices]
    ys = [vertex.y for vertex in vertices]
    xs.sort()
    ys.sort()

    xMin = xs[0]
    xMax = xs[-1]
    yMin = ys[0]
    yMax = ys[-1]

    return xMin, xMax, yMin, yMax


def barycentric(A, B, C, P):
    cx, cy, cz = cross(
        V3(B.x - A.x, C.x - A.x, A.x - P.x),
        V3(B.y - A.y, C.y - A.y, A.y - P.y)
    )

    if abs(cz) < 1:
        return -1, -1, -1

    u = cx / cz
    v = cy / cz
    w = 1 - (cx + cy) / cz
    
    return  w, v, u

class Render(object):
    def __init__(self):
        self.framebuffer =[]
        self.zbuffer =[]

    def glInit(self):
        pass

    def clear(self, r, g,b):
        self.framebuffer= [
        [color(r,g,b) for x in range(self.width)]
        for y in range(self.height)
        ]

        self.zbuffer = [
            [-float('inf') for x in range(self.width)]
            for y in range(self.height)
        ]

    def  glClear(self):
        self.clear()

    def glClearcolor(self, r, g, b):
        r = round(r*255)
        g = round(g*255)
        b = round(b*255)
        self.clear(r, g, b)

    def glColor(self, r,g,b):
        r = round(r*255)
        g = round(g*255)
        b = round(b*255)
        return color(r, g, b)
        
    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height
        #r = Render(width,height)

    def glViewport(self, x, y, width, height):
        self.viewPortWidth = width
        self.viewPortHeight = height
        self.xViewPort = x
        self.yViewPort = y

    def glVertex(self, x,y):
        calcX = round((x+1)*(self.viewPortWidth/2)+self.xViewPort)
        calcY = round((y+1)*(self.viewPortHeight/2)+self.yViewPort)
        self.point(calcX, calcY)


    def write(self, filename):
        f = open(filename, 'bw')
        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14 + 40 + self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(14 + 40))

        #image header 
        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))

        #pixel data
        for x in range(self.width):
            for y in range(self.height):
                    f.write(self.framebuffer[y][x])

        f.close()

    #function dot
    def point(self, x, y,color):
        try:
            #print(color)
            self.framebuffer[x][y] = self.glColor(color[0],color[1],color[2])
        except:
            pass  

    def glLine(self,x0, y0, x1, y1):
        '''x0 = round((x0+1)*(self.viewPortWidth/2)+self.xViewPort)
        y0 = round((y0+1)*(self.viewPortHeight/2)+self.yViewPort)
        x1 = round((x1+1)*(self.viewPortWidth/2)+self.xViewPort)
        y1 = round((y1+1)*(self.viewPortHeight/2)+self.yViewPort)'''
        #print('coordenadas',x0, y0, x1, y1)
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        offset = 0
        threshold = dx

        y = y0
        for x in range(x0, x1):
            if steep:
                self.point(y, x)
            else:
                self.point(x, y)
            
            offset += dy * 2
            if offset >= threshold:
                y += 1 if y0 < y1 else -1
                threshold += dx * 2

        #Referencia del repositorio ejemplo de dennis
    def glFinish(self, filename='out.bmp'):
        self.write(filename)

    def triangle(self, A, B, C, selectColor):
        xMin, xMax, yMin, yMax = bbox(A, B, C)
        for x in range(xMin, xMax + 1):
            for y in range(yMin, yMax + 1):
                P = V2(x, y)
                w, v, u = barycentric(A, B, C, P)
                if w < 0 or v < 0 or u < 0:
                    continue
                
                z = A.z * w + B.z * u + C.z * v
                
                try:
                    if z > self.zbuffer[x][y]:
                        self.point(x, y,selectColor)
                        '''
                            Para z's Color Map
                            z = round(z % 255)
                            zColor = color(z, z, z)
                            self.point(x, y, zColor)
                        '''
                        self.zbuffer[x][y] = z
                except:
                    pass

    def load(self, filename, translate, scale):
        model = Obj(filename)

        light = V3(0, 0, 1)
        
        for face in model.faces:
            vcount = len(face)

            if vcount == 3:
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1

                v1 = V3(model.vertices[f1][0], model.vertices[f1][1], model.vertices[f1][2])
                v2 = V3(model.vertices[f2][0], model.vertices[f2][1], model.vertices[f2][2])
                v3 = V3(model.vertices[f3][0], model.vertices[f3][1], model.vertices[f3][2])

                x1 = round((v1.x * scale.x) + translate.x)
                y1 = round((v1.y * scale.y) + translate.y)
                z1 = round((v1.z * scale.z) + translate.z)

                x2 = round((v2.x * scale.x) + translate.x)
                y2 = round((v2.y * scale.y) + translate.y)
                z2 = round((v2.z * scale.z) + translate.z)

                x3 = round((v3.x * scale.x) + translate.x)
                y3 = round((v3.y * scale.y) + translate.y)
                z3 = round((v3.z * scale.z) + translate.z)

                A = V3(x1, y1, z1)
                B = V3(x2, y2, z2)
                C = V3(x3, y3, z3)

                normal = cross(sub(B, A), sub(C, A))
                intensity = dot(normal, light)
                grey = round(intensity / 255)
                if grey < 0:
                    # Ignorar esta cara
                    continue
                intensityColor = color(grey, grey, grey)
                self.triangle(A, B, C, intensityColor)
                #self.triangle(A, B, C, [0,0,0])

            else:
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1
                f4 = face[3][0] - 1   

                v1 = V3(model.vertices[f1][0], model.vertices[f1][1], model.vertices[f1][2])
                v2 = V3(model.vertices[f2][0], model.vertices[f2][1], model.vertices[f2][2])
                v3 = V3(model.vertices[f3][0], model.vertices[f3][1], model.vertices[f3][2])
                v4 = V3(model.vertices[f4][0], model.vertices[f4][1], model.vertices[f4][2])

                x1 = round((v1.x * scale.x) + translate.x)
                y1 = round((v1.y * scale.y) + translate.y)
                z1 = round((v1.z * scale.z) + translate.z)

                x2 = round((v2.x * scale.x) + translate.x)
                y2 = round((v2.y * scale.y) + translate.y)
                z2 = round((v2.z * scale.z) + translate.z)

                x3 = round((v3.x * scale.x) + translate.x)
                y3 = round((v3.y * scale.y) + translate.y)
                z3 = round((v3.z * scale.z) + translate.z)

                x4 = round((v4.x * scale.x) + translate.x)
                y4 = round((v4.y * scale.y) + translate.y)
                z4 = round((v4.z * scale.z) + translate.z)

                A = V3(x1, y1, z1)
                B = V3(x2, y2, z2)
                C = V3(x3, y3, z3)
                D = V3(x4, y4, z4)

                normal = cross(sub(B, A), sub(C, A))
                intensity = dot(normal, light)
                grey = round(intensity  *255)
                if grey < 0:
                    # Ignorar esta cara
                    continue
                intensityColor = color(grey, grey, grey)
                
                #self.triangle(A, B, C, intensityColor)

                self.triangle(A, D, C, intensityColor)

r = Render()
r.glCreateWindow(1000, 1000)
r.glClearcolor(0, 0, 0)
r.load('sphere.obj', V3(500, 500, 0), V3(500, 500, 500))
r.glFinish()
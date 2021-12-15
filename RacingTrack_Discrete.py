#!/usr/bin/env python3.8
import pygame
import math
import numpy as np
from numpy.linalg import norm
from functools import cached_property


dim = (400, 400)

cWhite = (255, 255, 255)
cBlack = (0, 0, 0)
cRed = (255, 0, 0)
cGreen = (0, 255, 0)
cBlue = (0, 0, 255)
cYellow = (255, 255, 0)
cBorderRed = (255, 49, 49)


def rotate(v, theta):
    M = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    return M@v

def insideWindow(p):
    # assert(p.shape == (2,))
    return (0 <= p[0] < dim[0]) and (0 <= p[1] < dim[1])

def lerp(p1, p2, t):
    return (1-t)*p1 + (t)*p2

def getNormal(p: np.array):
    assert(p.shape == (2,))
    return np.array([-p[1], p[0]])

def getNormals(ps: np.array):
    assert(ps.shape[1] == 2)
    return np.array([getNormal(p) for p in ps])

def normalize(p: np.array):
    assert(p.ndim == 1)
    return p/norm(p)

def normalizes(ps: np.array):
    assert(ps.ndim == 2)
    return np.array([p/norm(ps) for p in ps])


class Bezier:
    def __init__(self, ps: np.array):
        self.ps = ps
        assert(self.n >= 2)

    def continuousClosure(self):
        # For the curve to be closed and C1:
        #       ps[-1] == ps[0] and 
        #       (ps[-1] == ps[0]) needs to be equidistant between ps[1] and ps[-2]
        # So, this function adds ps[-2], ps[-1] to ps
        # See: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html

        pBeforeLast = self.ps[0] + (self.ps[0] - self.ps[1])
        return Bezier(np.vstack((self.ps, [pBeforeLast, self.ps[0]])))

    @property
    def n(self):
        return self.ps.shape[0]

    @cached_property
    def derivative(self):
        return Bezier(self.n * (self.ps[1:] - self.ps[:-1]))

    def __call__(self, t):
        # Compute at t using
        # De Casteljau's Algorithm

        pss = np.copy(self.ps)
        for i in range(1, self.n):
            for j in range(self.n - i):
                pss[j] = lerp(pss[j], pss[j+1], t)
        return pss[0]

    def split(self, t=0.5):
        # split bezier curve into two curves at t

        l1 = []
        l2 = []

        pss = np.copy(self.ps)
        l1.append(1*pss[0]); l2.append(pss[-1])
        for i in range(1, self.n):
            for j in range(self.n - i):
                pss[j] = lerp(pss[j], pss[j+1], t)

            l1.append(1*pss[0]); l2.append(pss[self.n - i - 1])
        
        return (Bezier(np.array(l1)),
                Bezier(np.array(l2[::-1])))

    def dot(self, other):
        # dot producting two Bernstein Polynomial of R2 coefficients
        # to get a Bernstein Polynomial of R2 coefficients
        # See: https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node10.html

        assert(self.ps.shape[1] == 2)

        m = self.n - 1
        n = other.n - 1
        l = []
        for i in range(m+n+1):
            l.append(sum([
                (math.comb(m, j) * math.comb(n, i-j) / math.comb(m+n, i)) * self.ps[j].dot(other.ps[i-j])
                for j in range(max(0, i-n), min(m, i)+1)
            ]))
        
        return Bezier(np.array(l))

    def __add__(self, other):
        if self.n > other.n:
            return Bezier(self.ps + other.elevation(self.n).ps)
        else:
            return Bezier(self.elevation(other.n).os + other.ps)

    def __neg__(self):
        return Bezier(-self.ps)

    def elevationStep(self):
        # raise the degree of the Bezier by 1
        # See: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-elev.html

        qs = [1*self.ps[0]]
        for i in range(1, self.n):
            qs.append(lerp(self.ps[i], self.ps[i-1], i/(self.n + 1)))
        qs.append(self.ps[-1])
        return Bezier(np.array(qs))

    def elevation(self, m):
        assert(m >= self.n)

        toReturn = self
        for _ in range(m-self.n):
            toReturn = toReturn.elevationStep()

        return toReturn    

    def normalLine(self, t):
        # returns a function that describes the normalized normal at t

        return lambda t1: self(t) + t1 * normalize(getNormal(self.derivative(t)))

    def zeros(self, accuracy=1e-3):
        # Calculates zeros of a Bernstein polynomial (of R1 coefficients)

        assert(self.ps.ndim == 1)

        queue = [([0, 1], self)]
        iter = 0
        while 2**(-iter) > accuracy:
            new_queue = []
            for (interval, bez) in queue:
                if np.all(bez.ps > 0) or np.all(bez.ps < 0):
                    continue
                bez1, bez2 = bez.split()
                mid = (interval[0] + interval[1])/2.0
                new_queue.append(([interval[0], mid], bez1))
                new_queue.append(([mid, interval[1]], bez2))
            
            queue = new_queue
            iter += 1

        return [s for ([s, _], _) in queue]

    def projectT(self, p):
        # Project the point p into the bezier curve
        # and return the parameter t of the projection
        # See: https://ieeexplore.ieee.org/document/4722219

        local_minima = [0, 1] + self.derivative.dot(Bezier(self.ps - p)).zeros()
        global_minimum = min(local_minima, key=lambda t: norm(self(t) - p))
        return global_minimum
        
    def project(self, p):
        # returns the closest point
        return self(self.projectT(p))

    def distance(self, p):
        return norm(p - self.project(p))

    # def intersections_old(self, p, v):
    def intersections_old(self, p1, p2):
        
        #? First Method
        # https://math.stackexchange.com/questions/2347733/intersections-between-a-cubic-b%C3%A9zier-curve-and-a-line
        # w = getNormal(v)
        # local_minima = Bezier(self.ps.dot(w) - p.dot(w)).zeros()
        # global_minimum = min(local_minima, key=lambda t: norm(self(t) - p))
        # return global_minimum

        #? Line is a bezier curve Idea
        # Assuming that the curve is contained in the 400x400 window
        # global_minimum = min(local_minima, key=lambda t: norm(self(t) - p))
        # return local_minima

        #? offset curve
        # https://math.stackexchange.com/questions/152996/shifting-a-quadratic-b%c3%a9zier-curve
        # https://stackoverflow.com/questions/4148831/how-to-offset-a-cubic-bezier-curve
        # is hard
        # so try find approximation methods

        #? maybe check this
        # https://pomax.github.io/bezierinfo/#offsetting

        #? problem
        # # normalize(getNormal(self.derivative(t))) !=
        # # normalizes(getNormals(self.derivative.ps))
        # w = getNormal(v)
        # b2 = self + Bezier(-20*normalizes(getNormals(self.derivative.ps)))
        # local_minima = Bezier(b2.ps.dot(w) - p.dot(w)).zeros() + [0, 1]

                
        #? Just experiementation (discard later)
        # b1 = Bezier(self.ps - p)
        # b2 = Bezier(20*normalizes(getNormals(self.derivative.ps)))
        # a = self.derivative.dot((b1 + b2))
        # local_minima = [0, 1] + a.zeros()


        # global_minimum = min(local_minima, key=lambda t: norm(self(t) - p))

        # return global_minimum
        pass

    def intersection_bruteforce(self, p, v):
        # assert(norm(v) == 1)

        thePoint = p*1.0
        while insideWindow(thePoint):
            if window.get_at(tuple(thePoint.astype(int)))[:3] == cBorderRed:
                return [thePoint]

            thePoint += v

        return []


beziers = [
    np.array([(0, 0), (135, 135), (270, 270)]) + np.array([50, 50]),
    np.array([(0, 0), (270-45, 0+45), (270, 270)]) + np.array([50, 50]),
    np.array([(0, 0), (180, 0), (90, 270), (270, 270)]) + np.array([50, 50]),
    np.array([(0+25, 85*3+25), (0+25, 85+25), (85*4+25, 85*3+25), (0+25, 0+25), (4*85+25, 85+25)]),
    Bezier(np.array([(100, 200),
                    (200, 0),
                    (450, 50),
                    (450, 300)])).continuousClosure().ps,
    np.array([
        (   0, -100),
        (  50, -100),
        ( 100,  -50),
        ( 100,    0),
        ( 100,   50),
        (  50,  100),
        (   0,  100),
        ( -50,  100),
        (-100,   50),
        (-100,    0),
        (-100,  -50),
        ( -50, -100),
        (   0, -100),
    ])*1.5 + np.array([200, 220])
]


class Track:
    def __init__(self, ps, radius):
        self.bezier = Bezier(1.0 * ps) # (x1.0) to convert to array of doubles
        self.r = radius
        
    @cached_property
    def trackSurface(self):
        toReturn = pygame.Surface(dim)
        toReturn.fill(cWhite)

        step = 1/1000

        for i in np.arange(0, 1 + step, step):
            pygame.draw.circle(toReturn, cBorderRed, self.bezier(i), self.r*1.2)
        for i in np.arange(0, 1 + step, step):
            pygame.draw.circle(toReturn, cBlack, self.bezier(i), self.r)

        return toReturn

    def drawCheckpoints(self, t, n, clrSucc, clrFail):
        def drawCheckpoint(t, clr):
            normalF = self.bezier.normalLine(t)
            pygame.draw.line(window, clr, normalF(self.r), normalF(-self.r))

        for i in np.arange(0, 1, 1/n):
            drawCheckpoint(i, clr=[clrFail, clrSucc][bool(t>=i)])

    def isHit(self, p, r):
        return (self.bezier.distance(p) > self.r - r)

    def getVisionPoints(self, p, v, n, maxAngle):
        v = normalize(v)

        angles = np.arange(-(n-1)/2, (n-1)/2+1) * (maxAngle / ((n-1)/2))
        rotationVs = [rotate(v, theta) for theta in angles]

        return [self.bezier.intersection_bruteforce(p, w) for w in rotationVs]

    def getVisionDistances(self, p, v, n, maxAngle):
        l = []
        for w in self.getVisionPoints(p, v, n, maxAngle):
            if w:
                l.append(norm(w[0] - p))
            else:
                l.append(dim[0]) # or any big number

        return np.array(l)

    def drawVisionPoints(self, p, v, n, maxAngle, clr):
        for w in self.getVisionPoints(p, v, n, maxAngle):
            if w:
                pygame.draw.line(window, clr, p, w[0])


class Car:
    def __init__(self, track: Track, radius, accelerMax, steeringMax, dragCoefficient):
        self.track = track
        self.radius = radius

        self.p = track.bezier(0)
        self.v = normalize(track.bezier.derivative(0))
        
        self.accelerMax = accelerMax
        self.steeringMax = steeringMax

        self.dragCoefficient = dragCoefficient

        #! Stuff that could be changed too
        self.dt = 0.01

    def step(self, action_hotbinary4):
        assert(action_hotbinary4.shape == (4,))
        
        acceler = 0
        steering = 0
        if (action_hotbinary4 == np.array([1, 0, 0, 0])).all(): # Do Nothing
            pass
        elif (action_hotbinary4 == np.array([0, 1, 0, 0])).all(): # Acc
            acceler = self.accelerMax
        elif (action_hotbinary4 == np.array([0, 0, 1, 0])).all(): # left
            steering = -self.steeringMax
        elif (action_hotbinary4 == np.array([0, 0, 0, 1])).all(): # right
            steering = self.steeringMax
        else:
            raise ValueError("argument should be hot-binary encoded np-array with length 4")

        self.v += self.dt * (acceler - self.dragCoefficient * norm(self.v)**2) * normalize(self.v)
        # self.v += self.dt * (acceler) * normalize(self.v)
        self.v = rotate(self.v, steering*self.dt)
        self.p += self.dt * self.v


class Env:
    def __init__(self, checkpointsNumber, visionpointsNumber, visionpointsMaxAngle, accelerMax, steeringMax, trackRadius, bezier, carRadius, dragCoefficient=0.001):
        global window

        self.checkpointsNumber = checkpointsNumber
        self.visionpointsNumber = visionpointsNumber
        self.visionpointsMaxAngle = visionpointsMaxAngle
        self.accelerMax = accelerMax
        self.steeringMax = steeringMax

        self.trackRadius = trackRadius
        self.bezier = bezier
        self.carRadius = carRadius

        self.dragCoefficient = dragCoefficient
    
        pygame.init()
        window = pygame.display.set_mode(dim)

        self.track = Track(self.bezier, self.trackRadius)
        self.car = Car(self.track, self.carRadius, self.accelerMax, self.steeringMax, self.dragCoefficient)


        self.t = 0

    def step(self, action_hotbinary4):
        # Takes action, outputs state

        self.car.step(action_hotbinary4)
        
        # [return]
        # 1. (np.array) vision distances
        visionDistances = self.track.getVisionDistances(self.car.p, self.car.v, self.visionpointsNumber, self.visionpointsMaxAngle)

        # # 2. (double) completion factor [0, 1]
        # t = self.track.bezier.projectT(self.car.p)

        # 2. (double) CHANGE in completion factor [0, 1]
        new_t = self.track.bezier.projectT(self.car.p)
        t = new_t - self.t
        self.t = new_t

        # 3. (double) speed
        speed = norm(self.car.v)
        
        # 4. (bool) is hit
        shouldReset = self.track.isHit(self.car.p, self.car.radius)

        return (visionDistances, t, speed, shouldReset)

    def render(self):
        # Note: all colors can be changed from here and (Track.trackSurface)

        # draw track
        window.blit(self.track.trackSurface, (0, 0))
        # draw visionpoints
        self.track.drawVisionPoints(self.car.p, self.car.v, self.visionpointsNumber, self.visionpointsMaxAngle, cYellow)
        # draw car
        pygame.draw.circle(window, [cRed, cGreen][not self.track.isHit(self.car.p, self.car.radius)], self.car.p, self.car.radius)
        # draw checkpoints
        self.track.drawCheckpoints(self.track.bezier.projectT(self.car.p), self.checkpointsNumber, cGreen, cRed)
        # draw projection
        # pygame.draw.circle(window, cBlue, self.track.bezier.project(self.car.p), self.car.radius)


        pygame.display.flip()

        for event in pygame.event.get():
            # if event.type == pygame.QUIT:
            #     run = False
            pass # empties the queue

    def reset(self):
        self.car = Car(self.track, self.carRadius, self.accelerMax, self.steeringMax, self.dragCoefficient)

    def exit(self):
        pygame.quit()




def demo1():
    env = Env(20, 9, math.pi/8, 250, math.pi/2.5, 20, beziers[2], 5)

    while True:
        if env.step(1, 0.75)[3] == True:
            env.reset()

        env.render()

def demo2():
    env = Env(20, 9, math.pi/2, 5, math.pi/2.5, 20, beziers[2], 5)

    mouseOld = env.car.p
    while True:
        mouseNew = np.array(pygame.mouse.get_pos())
        v = mouseNew - mouseOld
        env.car.p = np.array(mouseNew)
        if norm(v) > 0.5:
            env.car.v = 20*normalize(env.car.v) + normalize(v)
        mouseOld = mouseNew

        env.render() 

def demo3():
    env = Env(20, 9, math.pi/2, 25, math.pi/2.5, 20, beziers[2], 2)

    a, b = 1, 0.75
    while True:
        sigmoid = lambda x: 1/(1 + math.exp(-50*(x-0.5)))
        feedback = env.step(a, b)
        print(feedback)
        l, _, _, res = feedback
        a = 0 if l[4] < 15 else min(l[3]/100, 1)
        b1 = l[3] + l[2] + l[1] + l[0]
        b2 = l[5] + l[6] + l[7] + l[8]
        b = sigmoid(2*(b2/(b1+b2))-1/2)
        if res:
            env.reset()

        env.render()

def demoKeys():
    env = Env(20, 9, math.pi/2, 100, math.pi, 20, beziers[3], 2)

    while True:
        action = np.array([1, 0, 0, 0])
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = np.array([0, 1, 0, 0])
        elif keys[pygame.K_LEFT]:
            action = np.array([0, 0, 1, 0])
        elif keys[pygame.K_RIGHT]:
            action = np.array([0, 0, 0, 1])

        state = env.step(action)
        if state[3] == True:
            env.reset()

        env.render()

        


if __name__ == "__main__":
    # demoKeys()
    env = Env(20, 9, math.pi/2, 100, math.pi, 20, beziers[3], 2)
    while True:
        env.render()

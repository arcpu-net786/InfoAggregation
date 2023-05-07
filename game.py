import pygame
import numpy as np
from math import sin, cos, radians
from random import randint, sample, choice
from car import Car
from ped import Ped
from calc import poly_points, distance

pygame.init()

colors = {   
            "black"  : (0, 0, 0),
            "brown"  : (145, 109, 103),
            "grey"   : (70, 70, 70), 
            "blue"   : (0, 0, 255),
            "cyan"   : (0, 255, 255), 
            "green"  : (0, 255, 0), 
            "yellow" : (253, 218, 22), 
            "red"    : (255, 0, 0), 
            "beige"  : (247, 233, 210),
            "orange" : (255, 128, 0), 
            "pink"   : (255, 0, 255), 
            "white"  : (255, 255, 255),

        }


class CarGameAI:
    def __init__(self, w=1000, h=300):
    	#initialize the Pygame GUI frame
        self.w = w
        self.h = h

        self.max_iters = self.w * 2
        self.goal = (self.w, self.h//2)

        self.gameDisplay = pygame.display.set_mode((self.w, self.h))
        self.clock=pygame.time.Clock()
        self.reset()


    def reset(self):
    	#initialize the car object on the frame
        self.car = Car("red", 140, 138, 140, 158, 100, 158, 100, 138, vel=2, deg=0)
        self.peds  = [Ped(i, choice([True, False])) for i in sorted(sample(range(int(self.w*0.1), int(self.w*0.9)), 20))]
        self.curr_peds = []
        self.min_dist = float("inf")
        self.frame_iteration = 0


    def is_collision(self): 
        #checks if vertices of car has collision
        for x, y in self.car.vertices:

            #boundary of frame or with off-road
            if x >= self.w or x <= 0 or y >= 211 or y <= 90:
                return True
        
        #checks if any point in  car has collision
        for x, y in poly_points(self.car.vertices):

            #any pedestrian
            for p in self.curr_peds:
                if distance((x, y), (p.x, p.y)) <= 5:
                    return True
        return False


    def view_ahead_pt(self, dist, angle): #checks if a point ahead of car FRONT has off-road collision or frame boundary
        x, y = self.car.front
        x_new = x + cos(radians(self.car.deg + angle)) * dist
        y_new = y - sin(radians(self.car.deg + angle)) * dist

        if x_new >= self.w or x_new <= 0 or y_new >= 211 or y_new <= 90: #boundaries of road
            return 1
        else:
            for p in self.curr_peds:
                if distance((x_new, y_new), (p.x, p.y)) <= 5: #(x_new, y_new) detects pedestrian
                    return 2
        return 0


    def draw_point(self, center, size=1, color="green"): #draw a point at position (x, y)
        x, y = center
        pygame.draw.polygon(self.gameDisplay, colors[color], [(x-size, y-size), (x+size, y-size), (x+size, y+size), (x-size, y+size)])


    def draw_road(self):
        y = 100
        pygame.draw.polygon(self.gameDisplay, colors["grey"], [(0, y-10), (self.w, y-10), (self.w, y+110), (0, y+110)])
        pygame.draw.polygon(self.gameDisplay, colors["white"], [(0, y), (self.w, y), (self.w, y+1), (0, y+1)])
        for i in range(0, self.w, 40):
            pygame.draw.polygon(self.gameDisplay, colors["white"], [(i, y+31), (i+10, y+31), (i+10, y+32), (i, y+32)])
            pygame.draw.polygon(self.gameDisplay, colors["white"], [(i+5, y+64), (i+15, y+64), (i+15, y+65), (i+5, y+65)])
        pygame.draw.polygon(self.gameDisplay, colors["white"], [(0, y+100), (self.w, y+100), (self.w, y+101), (0, y+101)])


    def update_ui(self, perception=True): #generate the GUI on Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.clock.tick(30)
        self.gameDisplay.fill(colors["white"]) #draw background
        self.draw_road() #draw road/infrastructure
        pygame.draw.polygon(self.gameDisplay, colors[self.car.color], [(self.car.x_fl,self.car.y_fl),(self.car.x_fr,self.car.y_fr),(self.car.x_br,self.car.y_br),(self.car.x_bl,self.car.y_bl)])
        
        ##############################################
        #Pedestrian management

        for p in self.peds:
            if 0 <= p.x - self.car.front[0] <= 50 and p not in self.curr_peds:
                self.curr_peds.append(p)

        for p in self.curr_peds:
            if 80 <= p.y <= 220:
                pygame.draw.circle(self.gameDisplay, colors[p.color], (p.x, p.y), 5)
                p.update()

        ##############################################

        #if user enables perception, draw perception samples
        if perception:
            x, y = self.car.front
            for angle in range(-90, 91, 10):
                for freq in range(20, 121, 20):
                    x_new = x + cos(radians(self.car.deg+angle))*freq
                    y_new = y - sin(radians(self.car.deg+angle))*freq
                    # if x_new >= self.w or x_new <= 0 or y_new >= 211 or y_new <= 90:
                    #     pass
                    # else:
                    for p in self.curr_peds:
                        if distance((x_new, y_new), (p.x, p.y)) <= 10:
                         self.draw_point((x_new, y_new), 1, "red")
                        else:
                            self.draw_point((x_new, y_new), 1)

        pygame.display.flip()


    def play_step(self, action):
    	#given an action passed by the agent, perform that action
        self.frame_iteration += 1
        self.move(action)
        
        reward = 0
        game_over = False

        #end the game if car takes too long
        if self.frame_iteration > self.max_iters:
            game_over = True
            reward = -15
            return reward, game_over

        #end the game if car reaches end of road
        if self.car.front[0] >= self.w or self.car.x_fl >= self.w or self.car.x_fr >= self.w:
            game_over = True
            reward = 15*((max(0, self.max_iters - self.frame_iteration))/self.max_iters)
            return reward, game_over

        #end the game upon collision
        if self.is_collision():
            game_over = True
            reward = -15*((max(0, self.max_iters - self.frame_iteration))/self.max_iters)
            return reward, game_over

        else:
            dst = distance(self.car.front, self.goal)
            if dst < self.min_dist:
                self.min_dist = dst 
                reward = 2
            else:
                reward = -5

        self.update_ui()
        return reward, game_over


    def move(self, action):
        self.car.forward(action.index(1)/14)

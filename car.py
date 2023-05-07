from math import sin, cos, radians, floor, ceil

class Car:
    def __init__(self, color, x_fl, y_fl, x_fr, y_fr, x_br, y_br, x_bl, y_bl, vel=2, deg=0):
    	#initialize the position of the car based on given coordinates
        self.x_fl, self.y_fl, self.x_fr, self.y_fr, self.x_br, self.y_br, self.x_bl, self.y_bl = x_fl, y_fl, x_fr, y_fr, x_br, y_br, x_bl, y_bl
        self.front = (self.x_fl + self.x_fr)/2, (self.y_fl + self.y_fr)/2
        self.back = (self.x_bl + self.x_br)/2, (self.y_bl + self.y_br)/2
        self.vertices = [(self.x_fl, self.y_fl), (self.x_fr, self.y_fr), (self.x_br, self.y_br), (self.x_bl, self.y_bl)]

        #initialize the orientation, velocity, and color of the car
        self.deg = deg
        self.vel = vel
        self.color = color
    
    def forward(self, vel=None):
    	#determine if linear motion is forward or backward
        if vel == None:
            vel = self.vel

        #linearly move the car by its velocity and update its coordinates
        self.x_fl, self.y_fl = self.x_fl + cos(radians(self.deg))*vel, self.y_fl - sin(radians(self.deg))*vel
        self.x_fr, self.y_fr = self.x_fr + cos(radians(self.deg))*vel, self.y_fr - sin(radians(self.deg))*vel
        self.x_bl, self.y_bl = self.x_bl + cos(radians(self.deg))*vel, self.y_bl - sin(radians(self.deg))*vel
        self.x_br, self.y_br = self.x_br + cos(radians(self.deg))*vel, self.y_br - sin(radians(self.deg))*vel
        
        self.front = (self.x_fl + self.x_fr)/2, (self.y_fl + self.y_fr)/2
        self.back = (self.x_bl + self.x_br)/2, (self.y_bl + self.y_br)/2
        
        #obtain the vertices of the car
        self.vertices = [(self.x_fl, self.y_fl), (self.x_fr, self.y_fr), (self.x_br, self.y_br), (self.x_bl, self.y_bl)]

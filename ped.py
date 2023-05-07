from random import random, choice

class Ped:
    def __init__(self, x, top):
        self.x = x
        if top:
            self.y = 80
        else:
            self.y = 220
        self.top = top
        self.vel = random()*3 + 0.5
        self.color = choice(["blue", "green", "yellow", "red", "orange", "pink", "brown"])

    def update(self):
        if self.top:
            self.y += self.vel
        else:
            self.y -= self.vel
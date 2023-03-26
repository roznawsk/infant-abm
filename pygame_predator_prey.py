import math
import pygame
import random
import time

# Define constants for the screen width and height
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440
CONSUME_SQ_DIST = 14 ** 2

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()


class Agent(pygame.sprite.Sprite):
    def __init__(self, size, color, x=None, y=None):
        super().__init__()

        # draw agent
        self.surf = pygame.Surface((2*size, 2*size), pygame.SRCALPHA, 32)
        pygame.draw.circle(self.surf, color, (size, size), size)
        self.rect = self.surf.get_rect()

        # default values
        self.vmax = 2.0

        # initial position
        self.x = x if x else random.randint(0, SCREEN_WIDTH)
        self.y = y if y else random.randint(0, SCREEN_HEIGHT)

        # initial velocity
        self.dx = 0
        self.dy = 0

        # inital values
        self.is_alive = True
        self.target = None
        self.age = 0
        self.energy = 0

        # move agent on screen
        self.rect.centerx = int(self.x)
        self.rect.centery = int(self.y)

    def update(self, screen, food=()):
        # target is dead, don't chase it further
        if self.target and not self.target.is_alive:
            self.target = None

        # eat the target if close enough
        if self.target:
            squared_dist = (self.x-self.target.x)**2 + (self.y-self.target.y)**2
            if squared_dist < CONSUME_SQ_DIST:
                self.target.is_alive = False
                self.energy = self.energy + 1

        # agent doesn't have a target, find a new one
        if not self.target:
            min_dist = 9999999
            min_agent = None
            for a in food:
                if a is not self and a.is_alive:
                    sq_dist = (self.x - a.x) ** 2 + (self.y - a.y) ** 2
                    if sq_dist < min_dist:
                        min_dist = sq_dist
                        min_agent = a

            if min_dist < 100000:
                self.target = min_agent

        # initalize forces to zero
        fx = 0
        fy = 0

        # move in the direction of the target, if any
        if self.target:
            # calculate speed vector and vector to the food
            food_vec_x = self.target.x - self.x
            food_vec_y = self.target.y - self.y

            vector_lengths = (math.sqrt(food_vec_x ** 2 + food_vec_y ** 2) *
                              math.sqrt(self.dx ** 2 + self.dy ** 2))

            if vector_lengths != 0:
                angle_sine = 1 - ((food_vec_x * self.dx + food_vec_y * self.dy) / vector_lengths) ** 2
            else:
                angle_sine = 0

            fx += 0.1 * (self.target.x - self.x) - angle_sine * self.dx
            fy += 0.1 * (self.target.y - self.y) - angle_sine * self.dy

        # update our direction based on the 'force'
        self.dx = self.dx + 0.05*fx
        self.dy = self.dy + 0.05*fy

        # slow down agent if it moves faster than it max velocity
        velocity = math.sqrt(self.dx ** 2 + self.dy ** 2)
        if velocity > self.vmax:
            self.dx = (self.dx / velocity) * (self.vmax)
            self.dy = (self.dy / velocity) * (self.vmax)

        # update position based on delta x/y
        self.x = self.x + self.dx
        self.y = self.y + self.dy

        # ensure it stays within the screen window
        self.x = max(self.x, 0)
        self.x = min(self.x, SCREEN_WIDTH)
        self.y = max(self.y, 0)
        self.y = min(self.y, SCREEN_HEIGHT)

        self.age = self.age + 1

        # update graphics
        self.rect.centerx = int(self.x)
        self.rect.centery = int(self.y)
        screen.blit(self.surf, self.rect)


class Predator(Agent):
    def __init__(self, x=None, y=None):
        size = 3
        color = (255, 0, 0)
        super().__init__(size, color)
        self.vmax = 2.5


class Prey(Agent):
    def __init__(self, x=None, y=None):
        size = 2
        color = (255, 255, 255)
        super().__init__(size, color)
        self.vmax = 2.0


class Plant(Agent):
    def __init__(self, x=None, y=None):
        size = 2
        color = (0, 128, 0)
        super().__init__(size, color)
        self.vmax = 0

# Initial agent lists go here
# ...


preys = [Prey() for i in range(100)]
predators = [Predator() for i in range(100)]
plants = [Plant() for _ in range(1000)]

iteration = 0
iteration_times = []
PRINT_EVERY = 24 * 5

while True:
    start = time.time()

    # Process inputs
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    screen.fill("black")  # Fill the display with a solid color

    # Do agent updates here
    # ...

    [a.update(screen) for a in plants]
    [a.update(screen, food=plants) for a in preys]
    [a.update(screen, food=preys) for a in predators]

    # Agent housekeeping here
    # ...

    # handle eaten and create new plant
    plants = [p for p in plants if p.is_alive is True]
    plants = plants + [Plant() for _ in range(20)]

    # handle eaten and create new preys
    preys = [p for p in preys if p.is_alive is True]

    for p in preys[:]:
        if p.energy > 5:
            p.energy = 0
            preys.append(Prey(x=p.x + random.randint(-20, 20), y=p.y + random.randint(-20, 20)))

    # handle old and create new predators
    predators = [p for p in predators if p.age < 2000]

    for p in predators[:]:
        if p.energy > 10:
            p.energy = 0
            predators.append(Predator(x=p.x + random.randint(-20, 20), y=p.y + random.randint(-20, 20)))

    pygame.display.flip()  # Refresh on-screen display

    end = time.time()
    iteration_times.append(end-start)

    if iteration % PRINT_EVERY == 0:
        avg_time = round(sum(iteration_times) / len(iteration_times) * 1000, 2)
        iteration_times = []

        print(
            f'Entity count: {len(plants) + len(preys) + len(predators)}, avg iteration time: {avg_time}ms')

    iteration += 1

    clock.tick(24)         # wait until next frame (at 24 FPS)

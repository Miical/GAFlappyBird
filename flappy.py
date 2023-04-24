from itertools import cycle
import random
import sys
import pygame
from pygame.locals import *

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

models_pool = []
models_number = 12
models_score = []
generation = 0
load_models = True

def init_models():
    for i in range(models_number):
        model = Sequential([
            Dense(7, activation="sigmoid", name="layer2"),
            Dense(1, activation="sigmoid", name="layer3"),
        ])
        model(np.array([[0, 0, 0]]))
        models_pool.append(model)
    if load_models:
        for i in range(models_number):
            models_pool[i].load_weights("models/model" + str(i) + ".keras")

def save_models():
    for i in range(models_number):
        models_pool[i].save_weights("models/model" + str(i) + ".keras")

def model_mutate(weights):
    new_weights = weights[:]
    for xi in range(len(new_weights)):
        for yi in range(len(new_weights[xi])):
            if random.uniform(0, 1) > 0.5:
                change = random.uniform(-0.3,0.3)
                new_weights[xi][yi] += change
    return new_weights

def model_crossover(weights1, weights2):
    weightsnew1 = weights1[:]
    weightsnew2 = weights2[:]
    weightsnew1[0] = weights2[0]
    weightsnew2[0] = weights1[0]
    return [weightsnew1, weightsnew2]

def predict(model, delta_x, delta_y, delta_y2):
    inputs = np.array([[delta_x, delta_y, delta_y2]])
    outputs = model(inputs)

    if outputs <= 0.5:
        return True
    return False


def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    init_models()

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.flip(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hitmask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = showWelcomeAnimation()

        while True:
            crashInfo = mainGame(movementInfo)
            showGameOverScreen(crashInfo)


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # index of player to blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                # make first flap sound and return values for mainGame
                SOUNDS['wing'].play()
                return {
                    'playery': playery + playerShmVals['val'],
                    'basex': basex,
                    'playerIndexGen': playerIndexGen,
                }

        # adjust playery, playerIndex, basex
        if (loopIter + 1) % 5 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)
        playerShm(playerShmVals)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))
        SCREEN.blit(IMAGES['player'][playerIndex],
                    (playerx, playery + playerShmVals['val']))
        SCREEN.blit(IMAGES['message'], (messagex, messagey))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def mainGame(movementInfo):
    global models_score

    score = [0] * models_number
    loopIter = [0] * models_number
    playerIndex = [0] * models_number
    playerIndexGen = [movementInfo['playerIndexGen']] * models_number
    playerx = [int(SCREENWIDTH * 0.2)] * models_number
    playery = [movementInfo['playery']] * models_number
    crashTest = [False] * models_number

    # player velocity, max velocity, downward acceleration, acceleration on flap
    playerVelY    =  [-9] * models_number   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  [10] * models_number   # max vel along Y, max descend speed
    playerMinVelY =  [-8] * models_number   # min vel along Y, max ascend speed
    playerAccY    =   [1] * models_number   # players downward acceleration
    playerRot     =  [45] * models_number   # player's rotation
    playerVelRot  =   [3] * models_number   # angular speed
    playerRotThr  =  [20] * models_number   # rotation threshold
    playerFlapAcc =  [-9] * models_number   # players speed on flapping
    playerFlapped = [False] * models_number # True when player flaps

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    dt = FPSCLOCK.tick(FPS)/1000
    pipeVelX = -128 * dt

    models_score = [-1] * models_number
    last_player = None
    iter_num = 0

    while True:
        iter_num += 1
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        for i in range(models_number):
            if models_score[i] != -1:
                continue
            delta_x = lowerPipes[0]['x'] - playerx[i]
            delta_y = lowerPipes[0]['y'] - playery[i] - PIPEGAPSIZE / 2
            delta_y2 = lowerPipes[1]['y'] - playery[i] - PIPEGAPSIZE / 2
            if delta_x <= -IMAGES['pipe'][0].get_width() / 2 and len(lowerPipes) > 2:
                delta_x = lowerPipes[1]['x'] - playerx[i]
                delta_y = lowerPipes[1]['y'] - playery[i] - PIPEGAPSIZE / 2
                delta_y2 = lowerPipes[2]['y'] - playery[i] - PIPEGAPSIZE / 2
            if predict(models_pool[i], delta_x, delta_y, delta_y2):
                if playery[i] > -2 * IMAGES['player'][0].get_height():
                    playerVelY[i] = playerFlapAcc[i]
                    playerFlapped[i] = True

            # check for crash here
            crashTest[i] = checkCrash({'x': playerx[i], 'y': playery[i], 'index': playerIndex[i]},
                                upperPipes, lowerPipes)
            if crashTest[i][0]:
                models_score[i] = iter_num
                continue

            # check for score
            playerMidPos = playerx[i] + IMAGES['player'][0].get_width() / 2
            for pipe in upperPipes:
                pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
                if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                    score[i] += 1

            # playerIndex basex change
            if (loopIter[i] + 1) % 3 == 0:
                playerIndex[i] = next(playerIndexGen[i])
            loopIter[i] = (loopIter[i] + 1) % 30


            # rotate the player
            if playerRot[i] > -90:
                playerRot[i] -= playerVelRot[i]

            # player's movement
            if playerVelY[i] < playerMaxVelY[i] and not playerFlapped[i]:
                playerVelY[i] += playerAccY[i]
            if playerFlapped[i]:
                playerFlapped[i] = False

                # more rotation to cover the threshold (calculated in visible rotation)
                playerRot[i] = 45

            playerHeight = IMAGES['player'][playerIndex[i]].get_height()
            playery[i] += min(playerVelY[i], BASEY - playery[i] - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # print(upperPipes)
        # add new pipe when first pipe is about to touch left of screen
        if 3 > len(upperPipes) > 0 and upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if len(upperPipes) > 0 and upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        basex = -((-basex + 100) % baseShift)
        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(max(score))

        game_over = True
        for i in range(models_number):
            if models_score[i] != -1:
                continue
            # Player rotation has a threshold
            visibleRot = playerRotThr[i]
            if playerRot[i] <= playerRotThr[i]:
                visibleRot = playerRot[i]

            game_over = False
            last_player = i
            playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex[i]], visibleRot)
            SCREEN.blit(playerSurface, (playerx[i], playery[i]))

        pygame.display.update()
        FPSCLOCK.tick(FPS)
        if game_over:
            return {
                'y': playery[last_player],
                'groundCrash': crashTest[last_player][1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score[last_player],
                'playerVelY': playerVelY[last_player],
                'playerRot': playerRot[last_player]
            }

def showGameOverScreen(crashInfo):
    """crashes the player down and shows gameover image"""
    global models_pool, models_number, models_score, generation
    generation += 1
    if generation % 6 == 0:
        save_models()
    print(models_score)

    best_models = []
    for i in range(4):
        best_score, best_id = -1, None
        for id in range(models_number):
            if models_score[id] > best_score:
                best_score, best_id = models_score[id], id
        models_score[best_id] = -1
        best_models.append(models_pool[best_id])

    weights1 = best_models[0].get_weights()
    weights2 = best_models[1].get_weights()

    models_pool[0].set_weights(weights1)
    models_pool[2].set_weights(model_mutate(weights1))
    models_pool[3].set_weights(model_mutate(weights1))
    models_pool[1].set_weights(model_mutate(weights2))

    for select in range(2):
        weights3 = best_models[2 + select].get_weights()
        new_weights1 = model_crossover(weights1, weights3)
        new_weights2 = model_crossover(weights2, weights3)
        models_pool[select * 4 + 4].set_weights(model_mutate(new_weights1[0]))
        models_pool[select * 4 + 5].set_weights(model_mutate(new_weights1[1]))
        models_pool[select * 4 + 6].set_weights(model_mutate(new_weights2[0]))
        models_pool[select * 4 + 7].set_weights(model_mutate(new_weights2[1]))


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collides with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()

import numpy as np
import cv2
import heapq as hq
import math

result = cv2.VideoWriter('result.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         10, (400, 250))


def boomerang(x, y):
    line_1 = (0.316 * x + 173.608 - y) >= 0
    line_2 = (0.857 * x + 111.429 - y) <= 0
    line_mid = (-0.114 * x + 189.091 - y) <= 0
    line_3 = (-3.2 * x + 436 - y) >= 0
    line_4 = (-1.232 * x + 229.348 - y) <= 0

    return (line_1 and line_2 and line_mid) or (line_3 and line_4 and not line_mid)


def hexagon(x, y):
    line_1 = (-0.571 * x + 174.286 - y) <= 0
    line_2 = (165 - x) <= 0
    line_3 = (0.571 * x + 25.714 - y) >= 0
    line_4 = (-0.571 * x + 254.286 - y) >= 0
    line_5 = (235 - x) >= 0
    line_6 = (0.571 * x - 54.286 - y) <= 0

    return line_1 and line_2 and line_3 and line_4 and line_5 and line_6


def circle(x, y):
    circ_eq = ((x - 300) ** 2 + (y - 185) ** 2 - 40 * 40) <= 0
    return circ_eq

def boomerang_space(x, y):
    line_1 = (0.316 * x + 178.608 - y) >= 0
    line_2 = (0.857 * x + 106.429 - y) <= 0
    line_mid = (-0.114 * x + 189.091 - y) <= 0
    line_3 = (-3.2 * x + 450 - y) >= 0
    line_4 = (-1.232 * x + 220.348 - y) <= 0

    # line_1_dist = (abs(0.316 * x + 173.608 - y) / math.sqrt(0.316 * 0.316 + 1)) >= 5
    # line_2_dist = (abs(0.857 * x + 111.429 - y) / math.sqrt(0.857 * 0.857 + 1)) >= 5
    # # line_mid_dist = abs(-0.114 * x + 189.091 - y) / math.sqrt(0.114 * 0.114 + 1) >= 5
    # line_3_dist = (abs(-3.2 * x + 436 - y) / math.sqrt(3.2 * 3.2 + 1)) >= 5
    # line_4_dist = (abs(-1.232 * x + 229.348 - y) / math.sqrt(1.232 * 1.232 + 1)) >= 5



    return ((line_1 and line_2 and line_mid) or (line_3 and line_4 and not line_mid)) \
        # and (line_1_dist and line_2_dist and line_3_dist and line_4_dist)


def hexagon_space(x, y):
    line_1 = (-0.575 * x + 169 - y) <= 0
    line_2 = (160 - x) <= 0
    line_3 = (0.575 * x + 31 - y) >= 0
    line_4 = (-0.575 * x + 261 - y) >= 0
    line_5 = (240 - x) >= 0
    line_6 = (0.575 * x - 61 - y) <= 0

    return line_1 and line_2 and line_3 and line_4 and line_5 and line_6

def circle_space(x, y):
    circ_eq = ((x - 300) ** 2 + (y - 185) ** 2 - 45 * 45) <= 0
    return circ_eq


def check_obs(x_pos, y_pos):
    return boomerang(x_pos, y_pos) or hexagon(x_pos, y_pos) or circle(x_pos, y_pos)

def check_obs_space(x_pos, y_pos):
    return boomerang_space(x_pos, y_pos) or hexagon_space(x_pos, y_pos) or circle_space(x_pos, y_pos)


blank_image = np.zeros((250, 400, 3), np.uint8)


def draw_map():
    for x_itr in range(0, blank_image.shape[1]):
        for y_itr in range(0, blank_image.shape[0]):
            if check_obs(x_itr, y_itr):
                blank_image[y_itr][x_itr] = [0, 0, 255]

    return blank_image


obstacle_map = draw_map()

visited = []

cost = {}
parent = {}

not_visited = []
hq.heapify(not_visited)


def action_left(pos, goal):
    global not_visited
    global cost
    global obstacle_map
    global parent
    global visited
    moved = False
    new_x = pos[0] - 1
    new_y = pos[1]
    if pos[0] > 0:
        if not check_obs_space(new_x, new_y) and not ((new_x, new_y) in visited):
            hq.heappush(not_visited, (cost[(pos[0], pos[1])] + 1, (new_x, new_y)))
            cost[(new_x, new_y)] = cost[(pos[0], pos[1])] + 1
            obstacle_map[new_y][new_x] = [255, 255, 255]
            parent[(new_x, new_y)] = (pos[0], pos[1])
            visited.append((new_x, new_y))

            if (new_x, new_y) == (goal[0], goal[1]):
                moved = True

    return moved


def action_right(pos, goal):
    global not_visited
    global cost
    global visited
    global obstacle_map
    global parent
    moved = False
    new_x = pos[0] + 1
    new_y = pos[1]
    if pos[0] < (obstacle_map.shape[1] - 1):
        if not check_obs_space(new_x, new_y) and not ((new_x, new_y) in visited):
            hq.heappush(not_visited, (cost[(pos[0], pos[1])] + 1, (new_x, new_y)))
            cost[(new_x, new_y)] = cost[(pos[0], pos[1])] + 1
            obstacle_map[new_y][new_x] = [255, 255, 255]
            parent[(new_x, new_y)] = (pos[0], pos[1])
            visited.append((new_x, new_y))

            if (new_x, new_y) == (goal[0], goal[1]):
                moved = True

    return moved


def action_down(pos, goal):
    global not_visited
    global cost
    global visited
    global obstacle_map
    global parent
    moved = False
    new_x = pos[0]
    new_y = pos[1] - 1
    if pos[1] > 0:
        if not check_obs_space(new_x, new_y) and not ((new_x, new_y) in visited):
            hq.heappush(not_visited, (cost[(pos[0], pos[1])] + 1, (new_x, new_y)))
            cost[(new_x, new_y)] = cost[(pos[0], pos[1])] + 1
            obstacle_map[new_y][new_x] = [255, 255, 255]
            parent[(new_x, new_y)] = (pos[0], pos[1])
            visited.append((new_x, new_y))

            if (new_x, new_y) == (goal[0], goal[1]):
                moved = True

    return moved


def action_up(pos, goal):
    global not_visited
    global cost
    global visited
    global obstacle_map
    global parent
    moved = False
    new_x = pos[0]
    new_y = pos[1] + 1
    if pos[1] < (obstacle_map.shape[0] - 1):
        if not check_obs_space(new_x, new_y) and not ((new_x, new_y) in visited):
            hq.heappush(not_visited, (cost[(pos[0], pos[1])] + 1, (new_x, new_y)))
            cost[(new_x, new_y)] = cost[(pos[0], pos[1])] + 1
            obstacle_map[new_y][new_x] = [255, 255, 255]
            parent[(new_x, new_y)] = (pos[0], pos[1])
            visited.append((new_x, new_y))

            if (new_x, new_y) == (goal[0], goal[1]):
                moved = True

    return moved


def action_up_left(pos, goal):
    global not_visited
    global cost
    global visited
    global obstacle_map
    global parent
    moved = False
    new_x = pos[0] - 1
    new_y = pos[1] + 1
    if pos[0] > 0 and pos[1] < (obstacle_map.shape[0] - 1):
        if not check_obs_space(new_x, new_y) and not ((new_x, new_y) in visited):
            hq.heappush(not_visited, (cost[(pos[0], pos[1])] + 1.4, (new_x, new_y)))
            cost[(new_x, new_y)] = cost[(pos[0], pos[1])] + 1.4
            obstacle_map[new_y][new_x] = [255, 255, 255]
            parent[(new_x, new_y)] = (pos[0], pos[1])
            visited.append((new_x, new_y))

            if (new_x, new_y) == (goal[0], goal[1]):
                moved = True

    return moved


def action_down_left(pos, goal):
    global not_visited
    global cost
    global visited
    global obstacle_map
    global parent
    moved = False
    new_x = pos[0] - 1
    new_y = pos[1] - 1
    if pos[0] > 0 and pos[1] > 0:
        if not check_obs_space(new_x, new_y) and not ((new_x, new_y) in visited):
            hq.heappush(not_visited, (cost[(pos[0], pos[1])] + 1.4, (new_x, new_y)))
            cost[(new_x, new_y)] = cost[(pos[0], pos[1])] + 1.4
            obstacle_map[new_y][new_x] = [255, 255, 255]
            parent[(new_x, new_y)] = (pos[0], pos[1])
            visited.append((new_x, new_y))

            if (new_x, new_y) == (goal[0], goal[1]):
                moved = True

    return moved


def action_up_right(pos, goal):
    global not_visited
    global cost
    global visited
    global obstacle_map
    global parent
    moved = False
    new_x = pos[0] + 1
    new_y = pos[1] + 1
    if pos[0] < (obstacle_map.shape[1] - 1) and pos[1] < (obstacle_map.shape[0] - 1):
        if not check_obs_space(new_x, new_y) and not ((new_x, new_y) in visited):
            hq.heappush(not_visited, (cost[(pos[0], pos[1])] + 1.4, (new_x, new_y)))
            cost[(new_x, new_y)] = cost[(pos[0], pos[1])] + 1.4
            obstacle_map[new_y][new_x] = [255, 255, 255]
            parent[(new_x, new_y)] = (pos[0], pos[1])
            visited.append((new_x, new_y))

            if (new_x, new_y) == (goal[0], goal[1]):
                moved = True

    return moved


def action_down_right(pos, goal):
    global not_visited
    global cost
    global visited
    global obstacle_map
    global parent
    moved = False
    new_x = pos[0] + 1
    new_y = pos[1] - 1
    if pos[0] < (obstacle_map.shape[1] - 1) and pos[1] > 0:
        if not check_obs_space(new_x, new_y) and not ((new_x, new_y) in visited):
            hq.heappush(not_visited, (cost[(pos[0], pos[1])] + 1.4, (new_x, new_y)))
            cost[(new_x, new_y)] = cost[(pos[0], pos[1])] + 1.4
            obstacle_map[new_y][new_x] = [255, 255, 255]
            parent[(new_x, new_y)] = (pos[0], pos[1])
            visited.append((new_x, new_y))

            if (new_x, new_y) == (goal[0], goal[1]):
                moved = True

    return moved


def dijkstras_solution(start, goal):
    global not_visited
    global cost
    global visited
    global obstacle_map
    global parent

    reached = False

    if check_obs_space(goal[0], goal[1]) or check_obs_space(start[0], start[1]):
        return "Start/Goal cannot be in the obstacle space"

    current = start
    hq.heappush(not_visited, (0, (start[0], start[1])))
    parent[(start[0], start[1])] = None

    cost[(start[0], start[1])] = 0
    obstacle_map[start[1]][start[0]] = [255, 255, 255]

    while current != goal and not reached:
        current = hq.heappop(not_visited)[1]
        reached_1 = action_left(current, goal)
        reached_2 = action_right(current, goal)
        reached_3 = action_up(current, goal)
        reached_4 = action_down(current, goal)
        reached_5 = action_down_right(current, goal)
        reached_6 = action_down_left(current, goal)
        reached_7 = action_up_left(current, goal)
        reached_8 = action_up_right(current, goal)

        reached = reached_1 or reached_2 or reached_3 or reached_4 or reached_5 \
                  or reached_6 or reached_7 or reached_8

        obstacle_map_flip = cv2.flip(obstacle_map, 0)
        cv2.imshow('img', obstacle_map_flip)
        result.write(obstacle_map_flip)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    backtrack = []
    itr_node = (goal[0], goal[1])

    if reached:
        while parent[itr_node] is not None:
            backtrack.append(itr_node)
            obstacle_map[itr_node[1]][itr_node[0]] = [255, 0, 0]
            obstacle_map_flip = cv2.flip(obstacle_map, 0)
            cv2.imshow('img', obstacle_map_flip)
            result.write(obstacle_map_flip)
            if cv2.waitKey(25) & 0xFF == ord('q'):
               break
            itr_node = parent[itr_node]

        result.release()
        return "The algorithm worked!"
        

# def input_start():


if __name__ == '__main__':

    ## Scenario 1
    # start = (36, 175)
    # goal = (0, 249) 

    start = (0, 0)
    goal = (399, 249) 

    # start_x = input("Enter a start position \n x = ")
    # start_y = input("\n y = ")

    message = dijkstras_solution(start, goal)

    print(message)

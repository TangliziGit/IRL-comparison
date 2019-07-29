import os
import random
import math

from PIL import Image
import pygame
import numpy as np
import pandas as pd
import pickle as pkl

# 设置游戏屏幕大小
SCREEN_WIDTH = 408  # 480
SCREEN_HEIGHT = 408  # 800



class OWObject(object):
    """
    Object in objectworld.
    """

    def __init__(self, inner_colour, outer_colour):
        """
        inner_colour: Inner colour of object. int.
        outer_colour: Outer colour of object. int.
        -> OWObject
        """

        self.inner_colour = inner_colour
        self.outer_colour = outer_colour

    def __str__(self):
        """
        A string representation of this object.

        -> __str__
        """

        return "<OWObject (In: {}) (Out: {})>".format(self.inner_colour,
                                                      self.outer_colour)

# 子弹类
class Bullet(pygame.sprite.Sprite):
    def __init__(self, bullet_img, init_pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = bullet_img
        self.rect = self.image.get_rect()
        self.rect.midbottom = init_pos
        self.speed = 10

    def move(self):
        self.rect.top -= self.speed


# 玩家飞机类
class Player(pygame.sprite.Sprite):
    def __init__(self, plane_img, player_rect, init_pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = []  # 用来存储玩家飞机图片的列表
        for i in range(len(player_rect)):
            self.image.append(plane_img.subsurface(player_rect[i]).convert_alpha())
        self.rect = player_rect[0]  # 初始化图片所在的矩形
        self.rect.topleft = init_pos  # 初始化矩形的左上角坐标
        self.speed = 8  # 初始化玩家飞机速度，这里是一个确定的值
        self.bullets = pygame.sprite.Group()  # 玩家飞机所发射的子弹的集合
        self.img_index = 0  # 玩家飞机图片索引
        self.is_hit = False  # 玩家是否被击中

    # 发射子弹
    def shoot(self, bullet_img):
        bullet = Bullet(bullet_img, self.rect.midtop)
        self.bullets.add(bullet)

    # 向上移动，需要判断边界
    def moveUp(self):
        if self.rect.top <= 0:
            self.rect.top = 0
        else:
            self.rect.top -= self.speed

    # 向下移动，需要判断边界
    def moveDown(self):
        if self.rect.top >= SCREEN_HEIGHT - self.rect.height:
            self.rect.top = SCREEN_HEIGHT - self.rect.height
        else:
            self.rect.top += self.speed

    # 向左移动，需要判断边界
    def moveLeft(self, UNIT):
        if self.rect.left - UNIT < 0:
            pass  # self.rect.left = 0
        else:
            self.rect.left -= UNIT  # self.speed

    # 向右移动，需要判断边界
    def moveRight(self, UNIT):
        if self.rect.left + UNIT >= SCREEN_WIDTH:  # - self.rect.width:
            pass  # self.rect.left = SCREEN_WIDTH - self.rect.width
        else:
            self.rect.left += UNIT  # self.speed


# 敌机类
class Enemy(pygame.sprite.Sprite):
    def __init__(self, enemy_img, enemy_down_imgs, init_pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = enemy_img
        self.rect = self.image.get_rect()
        self.rect.topleft = init_pos
        self.down_imgs = enemy_down_imgs
        self.speed = 10
        self.down_index = 0

    # 敌机移动，边界判断及删除在游戏主循环里处理
    def move(self):
        self.rect.top += self.speed
        # print(self.rect.top)


class Env:
    def __init__(self, n_objects=0, n_colours=0, prepare_tp=True):
        # 初始化 pygame
        pygame.init()
        # 设置游戏界面大小、背景图片及标题
        # 游戏界面像素大小
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        # 游戏界面标题
        pygame.display.set_caption('飞机大战')

        # 背景图
        self.background = pygame.image.load('./resources/image/background2.png').convert()

        # Game Over 的背景图
        self.game_over = pygame.image.load('./resources/image/gameover.png')

        # 飞机及子弹图片集合
        self.plane_img = pygame.image.load('./resources/image/shoot2.png')

        # 设置玩家飞机不同状态的图片列表，多张图片展示为动画效果
        self.player_rect = []
        offsetX, offsetY = 23, 0  # 42
        self.player_rect.append(pygame.Rect(0 + offsetX, 99 + offsetY, 57, 43))  # 玩家飞机图片
        self.player_pos = [0, SCREEN_HEIGHT - 43]  # [200, 200] # 600]
        self.player = Player(self.plane_img, self.player_rect, self.player_pos)

        # 子弹图片
        self.bullet_rect = pygame.Rect(1004, 987, 9, 21)
        self.bullet_img = self.plane_img.subsurface(self.bullet_rect)

        # 敌机不同状态的图片列表，多张图片展示为动画效果
        self.enemy1_rect = pygame.Rect(534, 612, 57, 43)
        self.enemy1_img = self.plane_img.subsurface(self.enemy1_rect)
        self.enemy1_down_imgs = []
        self.enemy1_down_imgs.append(self.plane_img.subsurface(pygame.Rect(534, 612, 57, 43)))
        self.enemy1_down_imgs.append(self.plane_img.subsurface(pygame.Rect(534, 612, 57, 43)))

        # 存储敌机，管理多个对象
        self.enemies1 = pygame.sprite.Group()

        # 存储被击毁的飞机，用来渲染击毁动画
        self.enemies_down = pygame.sprite.Group()

        # 初始化射击及敌机移动频率
        self.shoot_frequency = 0
        self.enemy_frequency = 0

        # 玩家飞机被击中后的效果处理
        self.player_down_index = 0  # 16

        # 初始化分数
        self.score = 0

        # 游戏循环帧率设置
        self.clock = pygame.time.Clock()
        self.UNIT = (SCREEN_WIDTH - self.enemy1_rect.width) // 3
        self.clock.tick(60)
        # self.enemy_x_list=[0, self.UNIT, 3*self.UNIT, 2*self.UNIT, 0]

        self.do_nothing = 2
        self.do_right = 1
        self.do_left = 0
        self.dir=[-1, 1, 0]

        self.n_time_hashing=25600
        self.n_sec_time_hashing=400
        self.n_states=4*self.n_sec_time_hashing
        self.n_actions=3
        self.n_objects=n_objects
        self.n_colours=n_colours

        self.enemy_x_list = [self.UNIT * i for i in [0, 1, 3, 2, 1, 3, 1, 3]]
        self.time_hash=pkl.load(open("time_hash.pkl", 'rb'))
        self.sec_time_hash=pkl.load(open("sec_time_hash.pkl", 'rb'))

        self.reverse_time_hash = [-1] * self.n_time_hashing
        self.reverse_sec_time_hash = [-1] * self.n_sec_time_hashing
        for idx, hash_code in enumerate(self.time_hash):
            self.reverse_time_hash[hash_code]=idx
        for idx, hash_code in enumerate(self.sec_time_hash):
            self.reverse_sec_time_hash[hash_code] = idx

        self.starting_state = (0, self.sec_time_hash[self.time_hash[0]])
        self.ending_state = (3, self.sec_time_hash[self.time_hash[268]])

        self.reset(False)
        self.init_expert_action_frame()

        # Generate objects.
        self.objects = {}
        for _ in range(self.n_objects):
            obj = OWObject(random.randint(0, self.n_colours),
                           random.randint(0, self.n_colours))

            while True:
                x = random.randint(0, 4)
                y = random.randint(0, self.n_sec_time_hashing)

                if (x, y) not in self.objects:
                    break

            self.objects[x, y] = obj

        self.prepare_tp=prepare_tp
        if prepare_tp:
            if not os.path.exists("transition_probability.pkl"):
                print("prepare for transition_probability")
                self.transition_probability = np.array(
                    [[[self._transition_probability(i, j, k)
                       for k in range(self.n_states)]
                      for j in range(self.n_actions)]
                     for i in range(self.n_states)])
                pkl.dump(self.transition_probability, open("transition_probability.pkl", 'wb'))
            else:
                self.transition_probability=pkl.load(open("transition_probability.pkl", 'rb'))

    def _next_time_hash_code(self, time_hash_code):
        # print(time_hash_code)
        time=self.reverse_time_hash[self.reverse_sec_time_hash[time_hash_code]]+1
        if time==300: return -1
        return self.sec_time_hash[self.time_hash[time]]

    def _next_position(self, position, action):
        if action==self.do_nothing:
            return position
        elif action==self.do_right:
            return min(position+1, 3)
        elif action==self.do_left:
            return max(position-1, 0)
        return -1

    def _transition_probability(self, state, action, result_state):
        state = self._get_real_state_by_state_code(state)
        result_state = self._get_real_state_by_state_code(result_state)

        if state[1]+1!=result_state[1]:
            return 0

        next_pos=state[0]+self.dir[action]
        if next_pos<0: next_pos=0
        if next_pos>3: next_pos=3

        if next_pos == result_state[0]:
            return 1
        return 0

    def init_expert_action_frame(self):
        self.expert_action = [2]+[int(x) for x in pkl.load(open("ops.pkl", "rb"))]# [1:]
        self.init_expert()
        self.expert_action_frame=pd.DataFrame(columns=["action"])
        self.origin_expert_states=[self.starting_state] + self.expert_states[:-1]
        self.origin_expert_actions=[self.do_nothing] + self.expert_action

        for state, action in zip(self.origin_expert_states, self.origin_expert_actions):
            state_code=self._get_state_code_by_state(state)
            # print("state_code", state_code,
            #     "state", state,
            #     "expert real state", self._get_real_state_by_state_code(state_code),
            #     "action", action)
            self.expert_action_frame.loc[state_code]=action
        pkl.dump(self.expert_action_frame, open("expert_action_frame.pkl", 'wb'))

    def check_state_code_in_expert_path(self, state_code):
        return state_code in self.expert_action_frame.index

    # state_code: 0~4*400, but time is from 0 to 268
    # state: (4, 400)
    # real time: state_code % 400 # self.n_sec_time_hashing

    # state_code's time -> sec -> first -> state
    # 0 ~ 299(now is 399) -> time_hash -> 0 ~ 25600
    # 0 ~ 25600 -> sec_time_hash -> 0 ~ 399
    def _get_state_by_state_code(self, state_code):
        return (state_code//self.n_sec_time_hashing,
                self.sec_time_hash[self.time_hash[state_code%self.n_sec_time_hashing]])

    def _get_state_code_by_state(self, state):
        # print(self.reverse_sec_time_hash[state[1]])
        # print(self.reverse_time_hash[self.reverse_sec_time_hash[state[1]]])
        return state[0]*self.n_sec_time_hashing+\
               self.reverse_time_hash[self.reverse_sec_time_hash[state[1]]]

    def _get_real_state_by_state_code(self, state_code):
        return (state_code//self.n_sec_time_hashing,
                state_code%self.n_sec_time_hashing)

    def _get_state_code_by_real_state(self, real_state):
        return real_state[0]*self.n_sec_time_hashing+real_state[1]

    def _get_logical_state(self, s):
        return [int(s[0]), int(s[1]+380)]

    def _get_ob_str(self, observation, time):
        ob=self._get_logical_state(observation)
        return np.array([ob[0], self.sec_time_hash[self.time_hash[time]]])

    def _get_state_code(self, logical_state):
        return (logical_state[0]-1)*self.n_sec_time_hashing+logical_state[1]

    def init_expert(self, load=True):
        if load and os.path.exists("expert_states.pkl") and os.path.exists("expert_imgs.pkl"):
            self.expert_states = pkl.load(open("expert_states.pkl", 'rb'))
            self.expert_imgs = pkl.load(open("expert_imgs.pkl", 'rb'))
            return

        print("init expert data")
        img, observation, termial=self.frame_step(self.do_nothing)
        self.expert_states = [self._get_ob_str(observation, 0)]
        self.expert_imgs=[img]

        for _i, action in enumerate(self.expert_action):
            img, observation_, termial=self.frame_step(action)
            ob=self._get_ob_str(observation, _i)
            ob_=self._get_ob_str(observation_, _i + 1)

            observation = observation_
            self.expert_states.append(ob_)
            self.expert_imgs.append(img)

            if termial==True:
                break
        pkl.dump(self.expert_states, open("expert_states.pkl", 'wb'))
        pkl.dump(self.expert_imgs, open("expert_imgs.pkl", 'wb'))
        print("init expert data done")

    def reset(self, hard=True):
        # 判断游戏循环退出的参数
        if hard:
            self.__init__(self.prepare_tp)
        self.running = True
        self.enemy_num = 0
        self.step = 0

    def frame_step(self, action):
        self.step += 1
        # 获取键盘事件（上下左右按键）
        # key_pressed = pygame.key.get_pressed()
        # key = input()
        # print(action)
        key = action
        # action = [0, 1, 2]
        # left, right, none

        if key == 0:  # key_pressed[K_a] or key_pressed[K_LEFT]:
            # print("left")
            self.player.moveLeft(self.UNIT)
        elif key == 1:  # key_pressed[K_d] or key_pressed[K_RIGHT]:
            # print("right")
            self.player.moveRight(self.UNIT)
        elif key == 2:
            pass

        # 生成敌机，需要控制生成频率
        if self.enemy_frequency % 20 == 0:
            # enemy1_pos = [random.randint(0, SCREEN_WIDTH - enemy1_rect.width), 0]
            self.enemy_num += 1
            enemy1_pos = [self.enemy_x_list[self.enemy_num % len(self.enemy_x_list)], 0]
            enemy1 = Enemy(self.enemy1_img, self.enemy1_down_imgs, enemy1_pos)
            self.enemies1.add(enemy1)
        self.enemy_frequency += 1
        if self.enemy_frequency >= 100:
            self.enemy_frequency = 0

        for bullet in self.player.bullets:
            # 以固定速度移动子弹
            bullet.move()
            # 移动出屏幕后删除子弹
            if bullet.rect.bottom < 0:
                self.player.bullets.remove(bullet)

        for enemy in self.enemies1:
            # 2. 移动敌机
            enemy.move()
            # 3. 敌机与玩家飞机碰撞效果处理
            # if pygame.sprite.collide_circle(enemy, self.player):
            if pygame.sprite.collide_rect(enemy, self.player):
                self.enemies_down.add(enemy)
                self.enemies1.remove(enemy)
                self.player.is_hit = True
                break
            # 4. 移动出屏幕后删除飞机
            if enemy.rect.top < 0 or enemy.rect.top > SCREEN_HEIGHT:
                self.enemies1.remove(enemy)
                self.score += 1

        # 敌机被子弹击中效果处理
        # 将被击中的敌机对象添加到击毁敌机 Group 中，用来渲染击毁动画
        self.enemies1_down = pygame.sprite.groupcollide(self.enemies1, self.player.bullets, 1, 1)
        for enemy_down in self.enemies1_down:
            self.enemies_down.add(enemy_down)

        # 绘制背景
        self.screen.fill(0)
        self.screen.blit(self.background, (0, 0))

        # 绘制玩家飞机
        if not self.player.is_hit:
            self.screen.blit(self.player.image[self.player.img_index], self.player.rect)
            # 更换图片索引使飞机有动画效果
            self.player.img_index = self.shoot_frequency // 8
        else:
            # 玩家飞机被击中后的效果处理
            self.player.img_index = self.player_down_index // 8
            # self.screen.blit(self.player.image[self.player.img_index], self.player.rect)
            self.screen.blit(self.player.image[0], self.player.rect)
            self.running = False

        # 敌机被子弹击中效果显示
        for enemy_down in self.enemies_down:
            if enemy_down.down_index == 0:
                pass
            if enemy_down.down_index > 7:
                self.enemies_down.remove(enemy_down)
                # self.score += 1000
                continue
            # self.screen.blit(enemy_down.down_imgs[enemy_down.down_index // 2], enemy_down.rect)
            self.screen.blit(enemy_down.down_imgs[0], enemy_down.rect)
            enemy_down.down_index += 1

        # 显示子弹
        self.player.bullets.draw(self.screen)
        # 显示敌机
        self.enemies1.draw(self.screen)

        # 绘制得分
        score_font = pygame.font.Font(None, 36)
        score_text = score_font.render(str(self.score), True, (128, 128, 128))
        text_rect = score_text.get_rect()
        text_rect.topleft = [10, 10]
        self.screen.blit(score_text, text_rect)

        # 更新屏幕
        pygame.display.update()

        # 处理游戏退出
        pygame.image.save(self.screen, "screenshot.png")
        img = Image.open("screenshot.png")
        ob = self._get_observation()

        if not self.running:
            # self.__init__(self.prepare_tp)
            return img, ob, True
        return img, ob, False

    def _get_observation(self):
        ob = [0, 0]
        ob[0] = self.player.rect.left // self.UNIT
        tmp = [0, 0, 0, 0]

        for enemy in self.enemies1:
            x = enemy.rect.left // self.UNIT
            tmp[x] = max(tmp[x], int(enemy.rect.top / 10 - 1))
        # print("[game states]", tmp)

        # for i, x in enumerate(tmp):
        #     ob[1] += x * pow(40, i)
        # ob[1] = int(ob[1])
        ob[1]=self.sec_time_hash[self.time_hash[self.step]]
        return ob

    def optimal_policy_deterministic(self, state_code):
        # in ..._frame.index
        if state_code in self.expert_action_frame.index:
            return self.expert_action_frame.loc[state_code].data[0]
        print("not exsits", state_code, self._get_state_by_state_code(state_code),
              '- real state:', self._get_real_state_by_state_code(state_code))
        return random.randint(0, self.n_actions - 1)

    def get_policy(self):
        return [self.optimal_policy_deterministic(s) for s in range(self.n_states)]

    def reward(self, state_code):
        if state_code == self._get_state_code_by_state(self.ending_state):
            return 1
        return 0

    def reward_deep_maxent(self, state_code):
        """
        Get the reward for a state int.

        state_int: State int.
        -> reward float
        """

        x, y = self._get_real_state_by_state_code(state_code)

        near_c0 = False
        near_c1 = False
        # for (dx, dy) in product(range(-3, 4), range(-3, 4)):
        for dx in range(-3, 4):
            if 0 <= x + dx < 4: # and 0 <= y + dy < self.grid_size:
                next_state_code=self._get_state_code_by_real_state((x+dx, y+1))
                next_state=self._get_state_by_state_code(next_state_code)
                dy=next_state[1]-y

                if (abs(dx) <= 3 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 0):
                    near_c0 = True
                if (abs(dx) <= 2 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 1):
                    near_c1 = True

        if near_c0 and near_c1:
            return 1
        if near_c0:
            return -1
        return 0

    def generate_trajectories(self, n_trajectories, trajectory_length, policy):
        trajectories = []
        for _ in range(n_trajectories):
            self.reset()
            sx, sy = self.starting_state

            trajectory = []
            for _ in range(trajectory_length):
                # Follow the given policy.
                action = policy(self._get_state_code_by_state((sx, sy)))

                img, next_state, terminal = self.frame_step(action)

                if terminal:
                    break

                state_int = self._get_state_code_by_state((sx, sy))
                action_int = action
                next_state_int = self._get_state_code_by_state(next_state)
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_state[0]
                sy = next_state[1]

            # print(self.expert_states)
            trajectories.append(np.array(trajectory))
        return np.array(trajectories)

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        # if feature_map == "coord":
        #     f = np.zeros(self.grid_size)
        #     x, y = i % self.grid_size, i // self.grid_size
        #     f[x] += 1
        #     f[y] += 1
        #     return f
        # if feature_map == "proxi":
        #     f = np.zeros(self.n_states)
        #     x, y = i % self.grid_size, i // self.grid_size
        #     for b in range(self.grid_size):
        #         for a in range(self.grid_size):
        #             dist = abs(x - a) + abs(y - b)
        #             f[self.point_to_int((a, b))] = dist
        #     return f

        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def feature_matrix_deep_maxent(self, discrete):
        return np.array([self.feature_vector_deep_maxent(i, discrete)
                         for i in range(self.n_states)])

    def feature_vector_deep_maxent(self, i, discrete):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> Feature vector.
        """

        sx, sy = self._get_state_by_state_code(i)

        nearest_inner = {}  # colour: distance
        nearest_outer = {}  # colour: distance

        for y in range(self.n_sec_time_hashing):
            for x in range(4):
                if (x, y) in self.objects:
                    dist = math.hypot((x - sx), (y - sy))
                    obj = self.objects[x, y]
                    if obj.inner_colour in nearest_inner:
                        if dist < nearest_inner[obj.inner_colour]:
                            nearest_inner[obj.inner_colour] = dist
                    else:
                        nearest_inner[obj.inner_colour] = dist
                    if obj.outer_colour in nearest_outer:
                        if dist < nearest_outer[obj.outer_colour]:
                            nearest_outer[obj.outer_colour] = dist
                    else:
                        nearest_outer[obj.outer_colour] = dist

        # Need to ensure that all colours are represented.
        for c in range(self.n_colours):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if discrete:
            # Temporary don't have to use
            #
            # state = np.zeros((2 * self.n_colours * self.grid_size,))
            # i = 0
            # for c in range(self.n_colours):
            #     for d in range(1, self.grid_size + 1):
            #         if nearest_inner[c] < d:
            #             state[i] = 1
            #         i += 1
            #         if nearest_outer[c] < d:
            #             state[i] = 1
            #         i += 1
            # assert i == 2 * self.n_colours * self.grid_size
            # assert (state >= 0).all()
            pass
        else:
            # Continuous features.
            state = np.zeros((2 * self.n_colours))
            i = 0
            for c in range(self.n_colours):
                state[i] = nearest_inner[c]
                i += 1
                state[i] = nearest_outer[c]
                i += 1

        return state

# # 游戏 Game Over 后显示最终得分
# font = pygame.font.Font(None, 48)
# text = font.render('Score: '+ str(score), True, (255, 0, 0))
# text_rect = text.get_rect()
# text_rect.centerx = screen.get_rect().centerx
# text_rect.centery = screen.get_rect().centery + 24
# screen.blit(game_over, (0, 0))
# screen.blit(text, text_rect)
#
# # 显示得分并处理游戏退出
# while 1:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             exit()
#     pygame.display.update()

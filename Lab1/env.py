import numpy as np
from gymnasium import Env, spaces, utils
from io import StringIO

# Actions
UP_RIGHT = 0
DOWN_RIGHT = 1

GRID_WORLD = ["XXXXX", "FFXFF", "FFXFF", "FFFFG", "FXFFF", "FXFFF", "XXXXX"]


class JumpToTheGoalEnv(Env):
    """
    Jump To The Goal involves crossing a maze from start to goal without crashing with any obstacles
    by walking over the grid.
    In the stochastic version, the player may not always move in the intended direction.

    ## Description
    The game starts with the player at location [1, 0] of the maze grid world with the
    goal located at [3, 4] for the 7x5 environment.

    Obstacles in the maze are distributed in set locations.

    The player makes moves until they reach the goal or crashes with an obstacle.

    ## Action Space
    The action shape is (1,) in the range {0, 1} indicating
    which direction to move the player.

    - 0: Move up-right
    - 1: Move down-right

    ## Observation Space
    The observation is a value representing the player's current position.
    States are numbered from 0 to 34 starting at [0, 0] and going from left to right and up to down.
    For example, row 0 is formed by s0, s1, s2, s3 and s4, row 1 by s5, s6, s7, s8, s9, and so on.

    ## Starting State
    The episode starts with the player in state [5] (location [1, 0]).

    ## Rewards
    Reward schedule:
    - Reach goal: +5
    - Crash with an obstacle: -5
    - Reach any other state: -1

    ## Episode End
    The episode ends if the following happens:

    - Termination:
        1. The player crashes with an obstacle.
        2. The player reaches the goal.

    ## Information
    step() and reset() return a dict with the following keys:
    - p - transition probability for the state.

    ## Arguments
    render_mode (str): "human" by default to visualize the grid.
    deterministic (bool): whether the env is deterministic (True) or stochastic (False)
    prob(float): if the env is stochastic, the probability of performing the action chosen

    For example, if action is up-right and deterministic is False, then:
    - P(up-right) = prob
    - P(down-right) = 1-prob

    ## Version History
    v0: Initial version release
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, deterministic=True, prob=1.0):
        super(JumpToTheGoalEnv, self).__init__()
        self.desc = np.asarray(GRID_WORLD, dtype="c")
        self.nrow, self.ncol = self.desc.shape
        self.deterministic = deterministic
        self.prob = prob

        self.nA = 2  # Two actions: UP_RIGHT and DOWN_RIGHT
        self.nS = self.nrow * self.ncol

        self.initial_state = 5
        self.state = self.initial_state
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        def to_s(row, col):
            return row * self.ncol + col

        def inc(row_ini, col_ini, a):
            if a == UP_RIGHT:
                row, col = row_ini - 1, col_ini + 1
            elif a == DOWN_RIGHT:
                row, col = row_ini + 1, col_ini + 1

            # Boundary conditions
            if row < 0 or row > self.nrow - 1:
                return 0 if row < 0 else self.nrow - 1, col
            if col > self.ncol - 1:
                return row_ini + 1, self.ncol - 1

            return row, col

        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = self.desc[new_row, new_col]
            if new_letter == b"X" or new_letter == b"G":
                terminated = True
            else:
                terminated = False
            reward = -1
            if new_letter == b"X":
                reward = -5
            elif new_letter == b"G":
                reward = 5
            return new_state, reward, terminated

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                for a in range(2):
                    li = self.P[s][a]
                    letter = self.desc[row, col]
                    if letter == b"G":
                        li.append((1.0, s, 0, True))
                    elif letter == b"X":
                        li.append((1.0, s, 0, True))
                    else:
                        if self.deterministic:
                            li.append((1.0, *update_probability_matrix(row, col, a)))
                        else:
                            prob_success = self.prob
                            prob_failure = 1 - self.prob
                            for b in range(2):
                                if b == a:
                                    li.append((prob_success, *update_probability_matrix(row, col, b)))
                                else:
                                    li.append((prob_failure, *update_probability_matrix(row, col, b)))

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def step(self, action):
        """Environment step."""
        transitions = self.P[self.state][action]
        if self.deterministic:
            p, s, r, t = transitions[0]  # Only one possible outcome
        else:
            i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
            p, s, r, t = transitions[i]
        self.state = s
        self.lastaction = action

        if self.render_mode == "human":
            self.render()
        return int(s), r, t, False, {"prob": p}

    def reset(self, seed=None, options=None):
        """Environment reset."""
        super().reset(seed=seed)
        self.state = self.initial_state
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.state), {"prob": 1}

    def render(self):
        """Environment render."""
        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    def _render_text(self):
        """Render in text format."""
        outfile = StringIO()
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        row, col = divmod(self.state, self.ncol)
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        outfile.write("\n".join("".join(line) for line in desc) + "\n")
        return outfile.getvalue()

    def _render_gui(self, mode):
        "Render with the gui."
        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is not installed, run `pip install gymnasium[toy_text]`")

        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Jump to the Goal")
            self.window = pygame.display.set_mode((self.ncol * 100, self.nrow * 100))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.ncol * 100, self.nrow * 100))
        canvas.fill((255, 255, 255))
        pix_square_size = 100

        # Grid draw
        for row in range(self.nrow):
            for col in range(self.ncol):
                pos = (col * pix_square_size, row * pix_square_size)
                rect = pygame.Rect(pos, (pix_square_size, pix_square_size))

                if self.desc[row, col] == b"X":
                    color = (0, 0, 0)
                elif self.desc[row, col] == b"G":
                    color = (6, 160, 27)
                else:
                    color = (79, 172, 234)

                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (180, 200, 230), rect, 1)

        # Agent draw
        agent_row, agent_col = divmod(self.state, self.ncol)
        agent_pos = (agent_col * pix_square_size, agent_row * pix_square_size)
        pygame.draw.circle(
            canvas,
            (255, 143, 0),
            (agent_pos[0] + pix_square_size // 2, agent_pos[1] + pix_square_size // 2),
            pix_square_size // 3,
        )

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        """Close the rendering window."""
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

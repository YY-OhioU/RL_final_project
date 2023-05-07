# Project 

## Environment:
    - source: 
        - flappy-bird-gymnasium
        - https://github.com/markub3327/flappy-bird-gymnasium
        - Modification:
            - flappy-bird-gymnasium/envs/game_logic.py
                - line 176: 
                    ```if self.player_y + PLAYER_HEIGHT >= self.base_y - 1 or self.player_y <= 0:
                        return True
                    ```
    - Other:
        - basic reward: 0 per step
        - penalty for dying: -2
        - reward for scoreing: 5

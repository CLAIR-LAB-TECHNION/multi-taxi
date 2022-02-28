TAXI_ENVIRONMENT_REWARDS = dict(
    step=-1,
    no_fuel=-1,
    bad_pickup=-1,
    bad_dropoff=-1,
    bad_refuel=-1,
    bad_fuel=-1,
    pickup=-1,
    standby_engine_off=-1,
    turn_engine_on=-1,
    turn_engine_off=-1,
    standby_engine_on=-1,
    intermediate_dropoff=2,
    final_dropoff=100,
    hit_wall=-1,
    collision=-1,
    collided=-1,
    unrelated_action=-1,
)

SIMPLIFIED_TAXI_ENVIRONMENT_REWARDS = dict(
    step=-1,
    no_fuel=-1,
    bad_pickup=-5,
    bad_dropoff=-2,
    bad_refuel=-1,
    bad_fuel=-5,
    pickup=-1,
    standby_engine_off=-1,
    turn_engine_on=-1,
    turn_engine_off=-1,
    standby_engine_on=-1,
    intermediate_dropoff=-1,
    final_dropoff=100,
    hit_wall=-5,
    collision=-5,
    collided=-5,
    unrelated_action=-1,
)

COLOR_MAP = {
    ' ': [0, 0, 102],  # Black background
    '_': [0, 0, 102],
    '0': [0, 0, 102],  # Black background beyond map walls
    '': [180, 180, 180],  # Grey board walls
    '|': [180, 180, 180],  # Grey board walls
    '+': [180, 180, 180],  # Grey board walls
    '-': [180, 180, 180],  # Grey board walls
    ':': [0, 0, 85],  # black passes board walls
    '@': [180, 180, 180],  # Grey board walls
    'P': [254, 151, 0],  # [254, 151, 0],  # Blue
    'P0': [254, 151, 0],  # [102, 51, 0],
    'P1': [254, 151, 0],  # [153, 76, 0],
    'P2': [254, 151, 0],  # [204, 102, 0],
    'P3': [254, 151, 0],  # [255, 128, 0],
    'P4': [254, 151, 0],  # [255, 153, 51],
    'D': [102, 0, 51],
    'D0': [102, 0, 51],
    'D1': [102, 0, 51],  # [153, 0, 76],
    'D2': [102, 0, 51],  # [204, 0, 102],
    'D3': [102, 0, 51],  # [255, 0, 127],
    'D4': [102, 0, 51],  # [255, 51, 153],
    'F': [250, 204, 255],  # Pink
    'G': [159, 67, 255],  # Purple
    'X': [0, 0, 102],

    # Colours for agents. R value is a unique identifier
    '1': [255, 255, 000],  # Yellow
    '2': [255, 000, 000],  # Red
    '3': [204, 204, 204],  # White
    '4': [51, 255, 000],  # Green
    '5': [100, 255, 255],  # Cyan
}

ALL_ACTIONS_NAMES = ['south', 'north', 'east', 'west', 'pickup', 'dropoff']

SIMPLIFIED_ALL_ACTIONS_NAMES = ['goto_src', 'goto_dst', 'pickup', 'dropoff']

BASE_AVAILABLE_ACTIONS = ['south', 'north', 'east', 'west',
                          'pickup', 'dropoff']

SIMPLIFIED_BASE_AVAILABLE_ACTIONS = ['goto_src', 'goto_dst', 'pickup', 'dropoff']

""",
                    'turn_engine_on', 'turn_engine_off',
                    'standby',
                    'refuel']"""

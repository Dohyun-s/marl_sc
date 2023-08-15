
from pettingzoo.mpe import (
    camrl
)

all_prefixes = ["atari", "classic", "butterfly", "mpe", "sisl"]

manual_environments = {
    "butterfly/knights_archers_zombies",
    "butterfly/pistonball",
    "butterfly/cooperative_pong",
    "sisl/pursuit",
}

all_environments = {
    "mpe/camrl": camrl,
}

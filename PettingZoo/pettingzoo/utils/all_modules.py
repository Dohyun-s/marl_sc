from pettingzoo.mpe import camrl_v3

all_prefixes = ["atari", "classic", "butterfly", "mpe", "sisl"]

manual_environments = {
    "butterfly/knights_archers_zombies",
    "butterfly/pistonball",
    "butterfly/cooperative_pong",
    "sisl/pursuit",
}

all_environments = {
    "mpe_v3/camrl": camrl_v3,
}

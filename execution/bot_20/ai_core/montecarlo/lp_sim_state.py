import numpy as np

class LPSimulatorState:
    def __init__(self, start_eth=1.0, start_usdc=0.0, price0=2000):
        self.eth = start_eth
        self.usdc = start_usdc
        self.price = price0

        # LP status
        self.in_position = False
        self.lower_tick = None
        self.upper_tick = None

        # Stats
        self.fees_earned = 0.0
        self.swap_count = 0
        self.portfolio_value = []

    def mark_to_market(self, price):
        return self.eth + (self.usdc / price)

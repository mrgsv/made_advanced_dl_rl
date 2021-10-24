"""Modified BlackJack env.
Reference: https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py"""
import gym
from gym import spaces
from gym.utils import seeding


def cmp(a, b):
    return float(a > b) - float(a < b)


plus_minus_compliance = {
    1: -1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 0,
    8: 0,
    9: 0,
    10: -1,
}


class BlackjackWithDoubleAndCountEnv(gym.Env):
    """Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards. (Virtually for all Blackjack games today).

    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).

    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.

    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.

    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """

    def __init__(self, natural=False, sab=False):
        # 0 - stick
        # 1 - hit
        # 2 - double
        # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
        self.dealer = None
        self.player = None
        self.plus_minus_sum = 0
        self.amount_of_cards_lower_than_15 = False
        self.deck = ([i for i in range(1, 11)] + [10] * 3) * 4
        self.folded_cards = []
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2), spaces.Discrete(41))
        )
        self.seed()
        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit: add a card to players hand and return
            card = self._draw_card(self.np_random)
            self.plus_minus_sum += self._get_card_plus_minus_cost(card)
            self.player.append(card)
            if self._is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
        elif action == 0:  # stick: play out the dealers hand, and score
            done = True
            while self._sum_hand(self.dealer) < 17:
                card = self._draw_card(self.np_random)
                self.plus_minus_sum += self._get_card_plus_minus_cost(card)
                self.dealer.append(card)
            reward = cmp(self._score(self.player), self._score(self.dealer))
            if self.sab and self._is_natural(self.player) and not self._is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                    not self.sab
                    and self.natural
                    and self._is_natural(self.player)
                    and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
        elif action == 2:  # double: double the bet and finish the game
            done = True
            card = self._draw_card(self.np_random)
            self.plus_minus_sum += self._get_card_plus_minus_cost(card)
            self.player.append(card)
            while self._sum_hand(self.dealer) < 17:
                card = self._draw_card(self.np_random)
                self.plus_minus_sum += self._get_card_plus_minus_cost(card)
                self.dealer.append(card)
            reward = 2 * cmp(self._score(self.player), self._score(self.dealer))
            if self.sab and self._is_natural(self.player) and not self._is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                    not self.sab
                    and self.natural
                    and self._is_natural(self.player)
                    and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self._sum_hand(self.player), \
               self.dealer[0], \
               self._usable_ace(self.player), \
               self._get_plus_minus_value()

    def reset(self):
        self.amount_of_cards_lower_than_15 = False
        self.deck = ([i for i in range(1, 11)] + [10] * 3) * 4
        self.folded_cards = []
        self.plus_minus_sum = 0
        dealer_hand = self._draw_hand(self.np_random)
        self.plus_minus_sum += self._get_card_plus_minus_cost(dealer_hand[0])
        self.dealer = dealer_hand
        player_hand = self._draw_hand(self.np_random)
        self.plus_minus_sum += self._get_card_plus_minus_cost(player_hand[0])
        self.plus_minus_sum += self._get_card_plus_minus_cost(player_hand[1])
        self.player = player_hand
        return self._get_obs()

    def _draw_card(self, np_random):
        if self.amount_of_cards_lower_than_15:
            self.deck = ([i for i in range(1, 11)] + [10] * 3) * 4
            self.folded_cards = []
            self.amount_of_cards_lower_than_15 = False
            self.plus_minus_sum = 0
        card = int(np_random.choice(self.deck))
        self.deck.remove(card)
        self.folded_cards.append(card)
        if len(self.deck) < 15:
            self.amount_of_cards_lower_than_15 = True
        return card

    def _draw_hand(self, np_random):
        return [self._draw_card(np_random), self._draw_card(np_random)]

    def _usable_ace(self, hand):  # Does this hand have a usable ace?
        return 1 in hand and sum(hand) + 10 <= 21

    def _sum_hand(self, hand):  # Return current hand total
        if self._usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    def _is_bust(self, hand):  # Is this hand a bust?
        return self._sum_hand(hand) > 21

    def _score(self, hand):  # What is the score of this hand (0 if bust)
        return 0 if self._is_bust(hand) else self._sum_hand(hand)

    def _is_natural(self, hand):  # Is this hand a natural blackjack?
        return sorted(hand) == [1, 10]

    def _get_plus_minus_value(self):  # get plus_minus sum
        return self.plus_minus_sum

    def _get_card_plus_minus_cost(self, card):
        return plus_minus_compliance[card]

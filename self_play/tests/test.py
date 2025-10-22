from diplomacy import Game
from ..models.self_play_agents import SelfPlayAgent, RandomStrategy, RLAgent
from ..models.game_playing import play_game


def set_up_game() -> Game:
    game = Game()
    return game
    
def set_up_agents(game, agent, strategy) -> dict:
    agents = {power[0] : agent(strategy()) for power in game.powers.items()}
    return agents

def test_play_game(agent=SelfPlayAgent, strategy=RandomStrategy):
    game = set_up_game()
    agents = set_up_agents(game, agent, strategy)
    play_game(game, agents, max_turns=10)

    print("Game played successfully up to phase:", game.phase)

def test_encode_board_state():
    game = set_up_game()
    agents = set_up_agents(game, RLAgent, RandomStrategy)
    board_state = agents['AUSTRIA']._encode_board_state(game)
    print("Encoded board state:", board_state())



from random import choice

class SelfPlayAgent:
    def __init__(self, strategy):
        self.strategy = strategy

    def choose_orders(self, game, power_name: str) -> list[str]:
        possible_orders = game.get_all_possible_orders()
        orderable_locations = game.get_orderable_locations(power_name)
        selected_orders = []
        for loc in orderable_locations:
            legal = possible_orders.get(loc, [])
            if legal:
                selected_orders.append(self.choose(legal))
        return selected_orders

    def choose(self, options) -> str:
        return self.strategy.select(options)
    
class RandomStrategy:
    def select(self, options: list[str]) -> str:
        return choice(options)
    
class RLAgent(SelfPlayAgent):
    def __init__(self, model):
        super().__init__(strategy=model)
        self.model = model

    def choose_orders(self, game, power_name):
        possible_orders = game.get_all_possible_orders()
        orderable_locations = game.get_orderable_locations(power_name)
        board_state = self._encode_board_state(game)

    def _encode_previous_orders(self, game):
        previous_orders = game.orders
        return previous_orders


    def _encode_board_state(self, game):
        state = {}
        map_ = game.map
        locs = map_.locs

        units_by_power = game.get_units(None)
        loc_to_unit = {
            u.split()[1].upper(): {
                "type": u.split()[0].strip('*'),
                "owner": power,
                "dislodged": u.startswith('*')
            }
            for power, units in units_by_power.items()
            for u in units
            }

        centers_by_power = game.get_centers(None)
        sc_owners = {
            loc.upper(): power
            for power, centers in centers_by_power.items()
            for loc in centers
        }

        dislodged_info = {}
        for unit_str, attacker in game.dislodged.items():
            parts = unit_str.strip('*').split()
            if len(parts) != 2:
                continue
            unit_type, loc = parts
            loc = loc.upper()
            owner = loc_to_unit.get(loc, {}).get("owner", None)
            dislodged_info[loc] = {
                "type": unit_type,
                "owner": owner,
                "attacker": attacker
            }


        for loc in locs:
            loc = loc.upper()
            data = {}

            unit_data = loc_to_unit.get(loc)
            data["unit"] = {
                "type": unit_data["type"],
                "owner": unit_data["owner"]
            } if unit_data and not unit_data.get("dislodged") else None

            data["dislodged_unit"] = dislodged_info.get(loc, None)
            data["supply_center_owner"] = sc_owners.get(loc, None)

            data["buildable"] = False
            data["removable"] = False

            data["loc_type"] = map_.loc_type.get(loc, None)

            if '/' in loc:
                parent = loc.split('/')[0].lower()
                if parent not in state:
                    state[parent] = {}
                if "coast_units" not in state[parent]:
                    state[parent]["coast_units"] = []
                if unit_data and unit_data["type"] == "F" and not unit_data.get("dislodged"):
                    state[parent]["coast_units"].append(loc)

            state[loc] = data

        return state

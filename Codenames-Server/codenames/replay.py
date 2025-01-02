from abc import ABC, abstractmethod
import json
import os
from typing import Literal, Dict, List
from players.codemaster import Codemaster
from players.guesser import Guesser


class Action(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def from_dict(role: Literal, data: dict) -> "Action":
        pass

    @abstractmethod
    def get_role(self) -> Literal[
        "blue_codemaster", "blue_guesser", "red_codemaster", "red_guesser"
    ]:
        pass


class GuessAction(Action):
    def __init__(self, word: str, color: Literal["red", "blue"], kept_guessing: bool = False):
        self.word = word
        self.role = f"{color}_guesser"
        self.kept_guessing = kept_guessing

    def to_dict(self):
        return {
            "word": self.word,
            "kept_guessing": self.kept_guessing
        }

    @staticmethod
    def from_dict(role: Literal[
        "blue_guesser", "red_guesser"
    ], data: dict) -> "GuessAction":
        return GuessAction(data["word"], role.split("_")[0], data["kept_guessing"])

    def get_role(self) -> Literal[
        "blue_guesser", "red_guesser"
    ]:
        return self.role
    
    def keep_guessing(self):
        self.kept_guessing = True


class HintAction(Action):
    def __init__(self, hint: str, num: int, color: Literal["red", "blue"], intentions: List[str] = None):
        self.hint = hint
        self.num = num
        self.intentions = intentions
        self.role = f"{color}_codemaster"

    def to_dict(self):
        return {
            **{
                "hint": self.hint,
                "num": self.num
            },
            **({} if self.intentions is None else {"intentions": self.intentions})
        }

    @staticmethod
    def from_dict(role: Literal[
        "blue_codemaster", "red_codemaster"
    ], data: dict) -> "HintAction":
        print(data)
        return HintAction(data["hint"], data["num"], role.split("_")[0], data.get("intentions", None))

    def get_role(self) -> Literal[
        "blue_codemaster", "red_codemaster"
    ]:
        return self.role


class Replay():
    def __init__(self, seed, one_team_game: bool = False, first_team: Literal["red", "blue"] = "red"):
        self.seed = seed
        self.actions: Dict[str, List[Action]] = {
            "blue_codemaster": [],
            "blue_guesser": [],
            "red_codemaster": [],
            "red_guesser": []
        }
        self.one_team_game = one_team_game
        self.first_team = first_team
        self.complete = False

    def add_action(
        self,
        action: Action,
    ):
        self.actions[action.get_role()].append(action)

    def to_json(self):
        return json.dumps({
            "seed": self.seed,
            "actions": {
                role: [action.to_dict() for action in role_actions]
                for role, role_actions in self.actions.items()
            },
            "one_team_game": self.one_team_game,
            "first_team": self.first_team,
            "complete": self.complete
        }, indent=2)

    def now_complete(self):
        self.complete = True

    @staticmethod
    def from_json(json_str) -> "Replay":
        data = json.loads(json_str)
        replay = Replay(
            data["seed"], data["one_team_game"], data["first_team"]
        )
        replay.actions = {
            role: [(
                GuessAction if "word" in action else HintAction
            ).from_dict(role, action) for action in role_actions]
            for role, role_actions in data["actions"].items()
        }
        print(replay.actions)
        replay.complete = data["complete"]
        return replay


class ReplayHandler(Codemaster, Guesser):
    def __init__(self, replay_id: str, seed=None, replay_folder: str = "replays", is_recording=False, **kwargs):
        self.replay_id = replay_id
        self.seed = seed
        self.replay_folder = replay_folder
        self.is_recording = is_recording
        self.replay = None
        self.is_broken = False

        self.action_pointer = -1
        self.num_guesses = {
            "red": 0,
            "blue": 0
        }
        self.current_actor = None

        if is_recording:
            self.setup_replay(**kwargs)
        else:
            self.get_replay()

    def get_replay_path(self):
        return f"{self.replay_folder}/{self.replay_id}.json"

    def get_replay(self):
        try:
            with open(self.get_replay_path(), "r") as f:
                json_str = f.read()
            self.replay = Replay.from_json(json_str)
            if not self.replay.complete:
                self.is_broken = True
                return
            self.seed = self.replay.seed
        except Exception as e:
            print(e)
            print("Could not load replay.")
            self.is_broken = True

    def setup_replay(self, **kwargs):
        try:
            if not os.path.exists(self.replay_folder):
                os.mkdir(self.replay_folder)
        except Exception as e:
            print(e)
            print("Could not create or access replay folder. Replays will not be saved.")
            self.is_broken = True
            return

        try:
            with open(self.get_replay_path(), "w") as f:
                f.write("")
        except Exception as e:
            print(e)
            print("Could not write to replay file. Replay will not be saved.")
            self.is_broken = True
            return

        self.replay = Replay(self.seed, **kwargs)
    
    def add_action(self, action: Action):
        self.replay.add_action(action)
    
    def save_replay(self, complete=False):
        if complete:
            self.replay.now_complete()
        try:
            with open(self.get_replay_path(), "w") as f:
                f.write(self.replay.to_json())
        except Exception as e:
            print(e)
            print("Could not write to replay file. Replay not saved.")
            self.is_broken = True
            return

    def set_game_state(self, words_on_board, key_grid):
        """A set function for wordOnBoard and keyGrid """
        pass

    def get_clue(self):
        """Function that returns a clue word and number of estimated related words on the board"""
        if self.is_broken:
            return None, None

        if self.current_actor is None:
            self.current_actor = self.replay.first_team
        else:
            self.current_actor = "blue" if self.current_actor == "red" else "red"

        if self.current_actor == self.replay.first_team:
            self.action_pointer += 1

        if self.action_pointer >= len(self.replay.actions[self.current_actor + "_codemaster"]):
            self.is_broken = True
            return None, None

        action = self.replay.actions[self.current_actor + "_codemaster"][self.action_pointer]
        return action.hint, action.num  # , action.intentions

    def set_board(self, words_on_board):
        """Set function for the current game board"""
        pass

    def set_clue(self, clue, num_guesses):
        """Set function for current clue and number of guesses this class should attempt"""
        pass

    def keep_guessing(self):
        """Return True if guess attempts remaining otherwise False"""
        if self.is_broken:
            return False

        if self.num_guesses[self.current_actor] >= len(self.replay.actions[self.current_actor + "_guesser"]):
            self.is_broken = True
            return False

        action = self.replay.actions[self.current_actor + "_guesser"][self.num_guesses[self.current_actor]]
        return action.kept_guessing

    def get_answer(self):
        """Return the top guessed word based on the clue and current game board"""
        if self.is_broken:
            return None
        
        if self.num_guesses[self.current_actor] >= len(self.replay.actions[self.current_actor + "_guesser"]):
            self.is_broken = True
            return None

        action = self.replay.actions[self.current_actor + "_guesser"][self.num_guesses[self.current_actor]]
        self.num_guesses[self.current_actor] += 1
        return action.word

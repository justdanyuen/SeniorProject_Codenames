from asyncio import iscoroutinefunction as is_async
from players.codemaster import Codemaster
from players.guesser import Guesser
import json


async def send(clientsocket, msg):
    await clientsocket.send(msg)

async def receive(clientsocket) -> str:
    return await clientsocket.recv()


class OnlineHumanCodemaster(Codemaster):
    """Codemaster derived class for human interaction"""

    def __init__(self, clientsocket):
        super().__init__()
        self.clientsocket = clientsocket

    def set_game_state(self, words_in_play, map_in_play):
        self.words = words_in_play
        self.maps = map_in_play
        
    async def get_clue(self):
        msg = {
            "prompt": {
                "type": "str",
                "message": "[Codemaster] Please enter a word and a number of guesses separated by a space"
            },
            "guesses_left": 0
        }
        await send(self.clientsocket, json.dumps(msg))
        clue = await receive(self.clientsocket)
        clue = clue.split(" ")

        if not self._is_valid(clue):
            msg = {"error": "Invalid input, please enter a valid word (one that is not in play) and a number of guesses, separated by a space"}
            await send(self.clientsocket, json.dumps(msg))
            return await self.get_clue()

        clue[0] = clue[0].lower().strip()
        clue[1] = int(clue[1].strip())

        return clue[0], clue[1]
    
    def _is_valid(self, clue):
        """Check if the clue is valid"""
        return not (
            len(clue) != 2 or
            not clue[1].strip().isnumeric() or
            int(clue[1].strip()) > 9 or
            int(clue[1].strip()) < 1 or
            len(clue[0]) < 1 or
            clue[0].upper().strip() in self.words
        )


class OnlineHumanGuesser(Guesser):
    """Guesser derived class for human interaction"""

    def __init__(self, clientsocket):
        super().__init__()
        self.clientsocket = clientsocket

    def set_clue(self, clue, num):
        print("The clue is:", clue, num)
        self.clue = clue
        self.num = num
        self.guesses_left = num + 1

    def set_board(self, words):
        self.words = words

    async def get_answer(self):
        msg = {
            "prompt": {
                "type": "str",
                "message": f"[Guesser] The clue is \"{self.clue}\" for {self.num}, please enter a guess"
            },
            "guesses_left": self.guesses_left
        }
        await send(self.clientsocket, json.dumps(msg))
        answer = await receive(self.clientsocket)

        if not self._is_valid(answer):
            msg = {"error": "Invalid input, please enter a valid word (one that is in play)"}
            await send(self.clientsocket, json.dumps(msg))
            return await self.get_answer()

        answer = answer.strip().lower()
        return answer

    async def keep_guessing(self):
        self.guesses_left -= 1
        msg = {
            "prompt": {
                "type": "bool",
                "message": "Would you like to keep guessing?"
            }
        }
        await send(self.clientsocket, json.dumps(msg))
        answer = await receive(self.clientsocket)

        return answer.lower() == "true"

    def _is_valid(self, result):
        return result.upper().strip() in self.words


class OnlineCodemaster(Codemaster):
    """Online codemaster container"""

    def __init__(self, clientsocket, codemaster, cm_kwargs={}):
        super().__init__()
        self.codemaster = codemaster(**cm_kwargs)
        self.clientsocket = clientsocket

    async def set_game_state(self, words_in_play, map_in_play):
        self.codemaster.set_game_state(words_in_play, map_in_play)
        msg = {"board": {"words": words_in_play, "key": map_in_play}}
        await send(self.clientsocket, json.dumps(msg))

    async def get_clue(self):
        clue = None
        if is_async(self.codemaster.get_clue):
            clue = await self.codemaster.get_clue()
        else:
            clue = self.codemaster.get_clue()
        msg = {"clue_success": True}
        await send(self.clientsocket, json.dumps(msg))
        return clue
    

class OnlineGuesser(Guesser):
    """Online guesser container"""

    def __init__(self, clientsocket, guesser, is_replaying=False, g_kwargs={}):
        super().__init__()
        if is_replaying:
            self.guesser = guesser
        else:
            self.guesser = guesser(**g_kwargs)
        self.clientsocket = clientsocket

    def set_clue(self, clue, num):
        self.guesser.set_clue(clue, num)

    async def set_board(self, words):
        self.words = words
        self.guesser.set_board(words)
        msg = {"board": {"words": words}}
        await send(self.clientsocket, json.dumps(msg))

    async def get_answer(self):
        anwer = None
        if is_async(self.guesser.get_answer):
            answer = await self.guesser.get_answer()
        else:
            answer = self.guesser.get_answer()
        msg = {"guess_success": self.words.index(answer.upper().strip())}
        await send(self.clientsocket, json.dumps(msg))
        return answer

    async def keep_guessing(self):
        if is_async(self.guesser.keep_guessing):
            return await self.guesser.keep_guessing()
        return self.guesser.keep_guessing()

from dotenv import load_dotenv
from anthropic import Anthropic
import os
from asyncio import iscoroutinefunction as is_async
from players.codemaster import Codemaster
from players.guesser import Guesser
import json



async def send(clientsocket, msg):
    await clientsocket.send(msg)

async def receive(clientsocket) -> str:
    return await clientsocket.recv()


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

#correlates to online.OnlineCodemaster
class AICodemaster(Codemaster):
    def __init__(self,clientsocket, brown_ic=None, glove_vecs=None, word_vectors=None, ):
        super().__init__()
        load_dotenv()
        self.claude_key = os.getenv('CLAUDE_KEY')  
        self.clientsocket = clientsocket
      
    async def set_game_state(self, words_in_play, maps_in_play):
        self.words_in_play = words_in_play
        self.maps_in_play = maps_in_play
        msg = {"board": {"words": words_in_play, "key": maps_in_play}}
        await send(self.clientsocket, json.dumps(msg))


    
    async def get_clue(self):
        codemaster_prompt = f"""
        You are a codemaster in the game Codenames. Your task is to provide a clue word that is semantically similar to as many 'Red' words as possible while avoiding words similar to 'Blue' and especially 'Assassin' words.

        Here is the list of 25 words on the board:
        <words>
        {self.words_in_play}
        </words>

        And here are their corresponding mappings:
        <mappings>
        {self.maps_in_play}
        </mappings>

        Your goal is to find a clue word that is semantically similar to as many 'Red' words as possible. The number of 'Red' words your clue relates to is your score. However, you must be careful:

        1. Avoid words similar to 'Blue', as they could help the opposing team.
        2. Absolutely avoid words similar to 'Assassin', as guessing this word results in an immediate loss.
        3. Words mapped to 'Civilian' are neutral, so they are less critical to avoid than 'Blue' or 'Assassin' words.

        Follow these steps:
        1. Analyze the words and their mappings.
        2. Identify the 'Red' words.
        3. Look for semantic connections between multiple 'Red' words.
        4. Consider potential clue words that relate to multiple 'Red' words.
        5. Evaluate each potential clue word to ensure it doesn't strongly relate to 'Blue' or 'Assassin' words.
        6. Choose the best clue word that maximizes the number of related 'Red' words while minimizing risk.

        Provide your response in the following format:
        <thinking>
        [Your thought process, including consideration of different options and why you chose your final clue word]
        </thinking>
        <answer>
        Clue word: [Your chosen clue word]
        Number: [The number of 'Red' words your clue relates to]
        Explanation: [Brief explanation of which 'Red' words your clue relates to and why you chose this clue]
        </answer>

        Important reminders:
        - Never include civilian words with red words in your count.
        - Ensure your clue word is not on the board or a variation of a word on the board.
        - Your clue must be a single word in English.
        - Avoid proper nouns unless they are common knowledge.
        - If a word contains a '*' character, it is a placeholder for a word that should not be guessed.
        - Never return a number higher than the number of unguessed red words (guessed red words are marked as *Red* in the word list).
        - Whenever you evaluate a potential clue word, always consider the risk of leading your team to 'Blue' or 'Assassin' words.
        """
        
        client = Anthropic(api_key=self.claude_key)
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0,
            system="This prompt will be used in a python script in a function whenever the AI Codemaster has get_clue called.",
            messages=[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": codemaster_prompt
                }]
            }]
        )
        print(message.content)
        clue_word, number = parse_answer(message.content)

        clue_word = clue_word.lower()
        
        return clue_word, number


def parse_answer(text_blocks):
    # Assume the relevant TextBlock is the last item in the list
    last_block = text_blocks[-1]
    
    # Extract the text content from the TextBlock
    content = last_block.text
    
    # Find the content within <answer> tags
    start = content.find("<answer>") + len("<answer>")
    end = content.find("</answer>")
    answer_content = content[start:end].strip()

    # Split the content by lines
    lines = answer_content.split("\n")

    # Extract Clue Word and Number
    clue_word = lines[0].split(": ")[1].strip()
    number = int(lines[1].split(": ")[1].strip())

    return (clue_word, number)


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
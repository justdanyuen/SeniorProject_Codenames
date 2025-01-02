import requests
import numpy as np
from players.guesser import Guesser

class ADAGuesser(Guesser):
    """Guesser class that mimics a field operative in the Codenames game"""

    def __init__(self):
        """Handle pretrained vectors and declare instance vars"""
        self.board = None
        self.clue = None
        self.num_guesses = 0
        self.model = "text-embedding-ada-002"
        try:
            with open("players/openai_api.key", "r") as f:
                self.api_key = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError("You must create a file called openai_api.key with your OpenAI API key in it")


    def set_board(self, words_on_board):
        """Set function for the current game board"""
        self.board = words_on_board

    def set_clue(self, clue, num_guesses):
        """Set function for current clue and number of guesses this class should attempt"""
        self.clue = clue
        self.num_guesses = num_guesses

    def keep_guessing(self):
        """Return True if guess attempts remaining otherwise False"""
        return self.num_guesses > 0

    def get_embeddings(self, words):
        """Get embeddings for a list of words using OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": words
        }
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            json=data,
            headers=headers
        )
        response.raise_for_status()
        embeddings = [embedding["embedding"] for embedding in response.json()["data"]]
        return embeddings

    def get_answer(self):
        """Return the top guessed word based on the clue and current game board"""
        if not self.board or not self.clue:
            raise ValueError("Board and clue must be set before getting an answer")

        # Get embeddings for the clue and words on the board
        clue_embedding = self.get_embeddings([self.clue])[0]
        board_embeddings = self.get_embeddings(self.board)

        # Calculate cosine similarity between clue and board words
        clue_norm = np.linalg.norm(clue_embedding)
        similarities = []
        for word_embedding, word in zip(board_embeddings, self.board):
            if (word in ["*Red*", "*Blue*", "*Civilian*", "*Assassin*"]):
                similarities.append(-np.inf)
                continue
            word_norm = np.linalg.norm(word_embedding)
            similarity = np.dot(clue_embedding, word_embedding) / (clue_norm * word_norm)
            similarities.append(similarity)

        # Get the index of the highest similarity
        best_guess_index = np.argmax(similarities)

        # Decrement the number of guesses
        self.num_guesses -= 1

        return self.board[best_guess_index]

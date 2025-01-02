export enum GameStatus {
  Pending,
  Ongoing,
  Won,
  Lost,
}

export type Board = {
  words: string[];
  key: string[];
  guessed: boolean[];
};

export type Prompt = {
  type: string;
  message: string;
};

export type GameState = {
  board: Board;
  prompt: Prompt | null;
  guesses_left: number;
  error: string | null;
  status: GameStatus;
};

export type GameStateStore = [
  GameState,
  {
    reset(): void;
    update(data: any): void;
    setStatus(status: GameStatus): void;
    clearPrompt(): void;
    getGrid(): { word: string; color: string; guessed: boolean }[][];
  }
];

export const defaultGameState: GameState = {
  board: { words: [], guessed: [], key: [] },
  prompt: null,
  guesses_left: 0,
  error: null,
  status: GameStatus.Pending,
};

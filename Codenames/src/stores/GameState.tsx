import { JSX, createContext, useContext } from "solid-js";
import { createStore } from "solid-js/store";
import {
  GameState,
  GameStateStore,
  GameStatus,
  defaultGameState,
} from "~/util/prototypes";

const GameStateContext = createContext<GameStateStore>();

export function GameStateProvider(props: {
  gameState?: GameState;
  children?: JSX.Element[] | JSX.Element;
}) {
  const [state, setState] = createStore(props.gameState || defaultGameState);
  const gameState: GameStateStore = [
    state,
    {
      reset() {
        setState(defaultGameState);
      },
      update(data: any) {
        for (const key in data) updateGameStateKV(key, data[key]);
        if (!data.hasOwnProperty("prompt")) setState({ prompt: null });
        if (!data.hasOwnProperty("error")) setState({ error: null });
      },
      setStatus(status: GameStatus) {
        setState({ status });
      },
      clearPrompt() {
        setState({ prompt: null });
      },
      getGrid() {
        const words = state.board.words;
        const key = state.board.key;
        const guessed = state.board.guessed;
        const grid: { word: string; color: string; guessed: boolean }[][] = [];
        if (!words || !key || !guessed) return [];
        for (let i = 0; i < words.length; i++) {
          const row = Math.floor(i / 5);
          if (grid.length <= row) {
            grid.push([]);
          }
          const item = {
            word: words[i],
            color: key[i],
            guessed: guessed[i],
          };
          grid[row].push(item);
        }
        return grid;
      },
    },
  ];

  const updateGameStateKV = (key: string, value: any) => {
    switch (key) {
      case "guess_success":
        setState("board", "guessed", (prev) => {
          prev[value] = true;
          return [...prev];
        });
        break;
      case "board":
        if (state.board.guessed.length === 0)
          setState("board", {
            guessed: new Array(value.words.length).fill(false),
          });
        if (state.board.words.length === 0)
          setState("board", { words: value.words });
        if (state.board.key.length === 0 && value.hasOwnProperty("key"))
          setState("board", { key: value.key });
        break;
      case "prompt":
        setState({ prompt: value });
        break;
      case "guesses_left":
        setState({ guesses_left: value });
        break;
      case "error":
        setState({ error: value });
        alert(value);
        break;
      case "game_over":
        setState({
          status: value === "won" ? GameStatus.Won : GameStatus.Lost,
        });
        break;
    }
  };

  return (
    <GameStateContext.Provider value={gameState}>
      {props.children}
    </GameStateContext.Provider>
  );
}

export function useGameState() {
  return useContext(GameStateContext);
}

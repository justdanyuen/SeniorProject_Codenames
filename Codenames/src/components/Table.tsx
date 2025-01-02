import { createSignal, For, Switch, Match, Show, Accessor } from "solid-js";
import Row from "./Row";
import Loading from "./Loading";
import WebSocket from "isomorphic-ws";
import "./Table.css";
import { useGameState } from "~/stores/GameState";
import { GameStatus } from "~/util/prototypes";

export default function Table(props: { socketState: Accessor<0 | 2 | 1 | 3> }) {
  const state = useGameState();
  if (!state) throw new Error("Store uninitialized");
  const [gameState, { reset: resetGS, getGrid: getGridGS }] = state;

  const gameStateGrid = () => getGridGS();
  const [reloading, setReloading] = createSignal(false);
  const reload = () => {
    location.reload();
    setReloading(true);
    resetGS();
  };

  return (
    <div class="mb-4">
      <Switch
        fallback={
          <div class="text-center">
            <div>
              <b>Disconnected</b>
            </div>
            <input
              type="button"
              value="Reconnect"
              class="btn btn-outline-primary"
              onClick={reload}
              disabled={reloading()}
            />
          </div>
        }
      >
        <Match when={gameState.status === GameStatus.Ongoing}>
          <Show
            when={gameStateGrid().length > 0}
            fallback={<Loading></Loading>}
          >
            <div class="container">
              <For each={gameStateGrid()}>
                {(row) => <Row words={row} isKey={false}></Row>}
              </For>
            </div>
          </Show>
        </Match>
        <Match when={gameState.status === GameStatus.Lost}>
          <div class="text-center">
            <div>
              <b>You lost!</b>
            </div>
            <input
              type="button"
              value="New Game"
              class="btn btn-outline-primary"
              onClick={reload}
              disabled={reloading()}
            />
          </div>
        </Match>
        <Match when={gameState.status === GameStatus.Won}>
          <div class="text-center">
            <div>
              <b>You won!</b>
            </div>
            <input
              type="button"
              value="New Game"
              class="btn btn-outline-primary"
              onClick={reload}
              disabled={reloading()}
            />
          </div>
        </Match>
        <Match
          when={
            gameState.status === GameStatus.Pending &&
            props.socketState() === WebSocket.CONNECTING
          }
        >
          <Loading text="Connecting..."></Loading>
        </Match>
      </Switch>
    </div>
  );
}

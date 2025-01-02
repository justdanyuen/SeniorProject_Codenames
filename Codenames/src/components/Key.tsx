import { Show, For, Accessor } from "solid-js";
import "./Key.css";
import { useGameState } from "~/stores/GameState";
import { GameStatus } from "~/util/prototypes";
import { cardTypeToColor } from "~/util/general";

export default function Key(props: { humanIsCodemaster: Accessor<boolean> }) {
  const state = useGameState();
  if (!state) throw new Error("Store uninitialized");
  const [gameState, { getGrid: getGridGS }] = state;
  const gameStateGrid = () => getGridGS();

  return (
    <Show
      when={
        gameState.status === GameStatus.Ongoing &&
        props.humanIsCodemaster() &&
        gameStateGrid().length > 0
      }
    >
      <div class="col mx-auto">
        <div class="card text-center border-0 shadow mx-auto" id="key-card">
          <div class="card-body">
            <div id="key-grid" class="d-grid mx-auto">
              <For each={gameState.board.key}>
                {(color) => (
                  <div
                    class="rounded"
                    id="key-cell"
                    style={`background-color: ${cardTypeToColor(color, true)}`}
                  ></div>
                )}
              </For>
            </div>
          </div>
        </div>
      </div>
    </Show>
  );
}

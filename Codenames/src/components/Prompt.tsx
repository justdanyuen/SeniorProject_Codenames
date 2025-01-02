import { Match, Show, Switch } from "solid-js";
import "./Prompt.css";
import { getRole } from "~/util/general";
import { useGameState } from "~/stores/GameState";
import Loading from "./Loading";
import { GameStatus } from "~/util/prototypes";

export default function Prompt(props: { send: (message: string) => void }) {
  const state = useGameState();
  if (!state) throw new Error("Store uninitialized");
  const [gameState, { clearPrompt: clearPromptGS }] = state;

  let textRef: any;

  function respond(event: Event) {
    event.preventDefault();
    clearPromptGS();
    props.send(textRef.value);
  }

  function respondYes(event: Event) {
    event.preventDefault();
    clearPromptGS();
    props.send("true");
  }

  function respondNo(event: Event) {
    event.preventDefault();
    clearPromptGS();
    props.send("false");
  }

  return (
    <Show
      when={
        gameState.status === GameStatus.Ongoing &&
        gameState.board.words.length > 0
      }
    >
      <Show
        when={gameState.hasOwnProperty("prompt") && gameState.prompt != null}
        fallback={<Loading text="Thinking..."></Loading>}
      >
        <div
          class="card text-center mx-auto shadow bg-body border-0"
          id="prompt"
        >
          <div class="card-body">
            <Switch>
              <Match when={getRole(gameState.prompt?.message) != null}>
                <h5 class="card-title">
                  {(getRole(gameState.prompt?.message) ?? ["_"])[0]}
                </h5>
                <Show when={gameState.guesses_left > 0}>
                  <h6 class="card-subtitle mb-2 text-muted">
                    {gameState.guesses_left} guess
                    {gameState.guesses_left > 1 ? "es" : ""} left (you get one
                    extra)
                  </h6>
                </Show>
                <p class="card-text">
                  {(getRole(gameState.prompt?.message) ?? ["", "_"])[1]}
                </p>
              </Match>
              <Match when={getRole(gameState.prompt?.message) == null}>
                <p class="card-text">{gameState.prompt?.message}</p>
              </Match>
            </Switch>
            <Switch>
              <Match when={gameState.prompt?.type === "str"}>
                <form onSubmit={respond} autocomplete="off">
                  <div class="input-group mb-3 mx-auto">
                    <input
                      type="text"
                      class="form-control"
                      aria-label="input-text"
                      aria-describedby="button-addon"
                      ref={textRef}
                    />
                    <input
                      type="submit"
                      class="btn btn-secondary"
                      id="button-addon"
                      value="Submit"
                    />
                  </div>
                </form>
              </Match>
              <Match when={gameState.prompt?.type === "bool"}>
                <div class="btn-group mx-auto">
                  <input
                    type="button"
                    class="btn border-end-0 btn-outline-primary"
                    value="Yes"
                    onClick={respondYes}
                  />
                  <input
                    type="button"
                    class="btn btn-outline-primary"
                    value="No"
                    onClick={respondNo}
                  />
                </div>
              </Match>
            </Switch>
          </div>
        </div>
      </Show>
    </Show>
  );
}

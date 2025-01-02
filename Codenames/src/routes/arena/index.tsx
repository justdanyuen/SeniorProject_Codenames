import Table from "~/components/Table";
import Prompt from "~/components/Prompt";
import Key from "~/components/Key";
import { Show, createEffect, createSignal } from "solid-js";
import WebSocket from "isomorphic-ws";
import "./index.css";
import { getRole } from "~/util/general";
import { useGameState } from "~/stores/GameState";
import { GameStatus } from "~/util/prototypes";
import { Title } from "solid-start";

export default function GameUI() {
  const state = useGameState();
  if (!state) throw new Error("Store uninitialized");
  const [
    gameState,
    { reset: resetGS, update: updateGS, setStatus: setStatusGS },
  ] = state;

  const [message, setMessage] = createSignal("");
  const socket = new WebSocket("ws://localhost:8001/");
  const [socketState, setSocketState] = createSignal(socket.readyState);
  // used to only show the key to the codemaster
  const [humanIsCodemaster, setHumanIsCodemaster] = createSignal(false);
  const codemasterRole = "Codemaster";

  const startGame = () => send("Connection established.");

  socket.onopen = () => setSocketState(socket.readyState);
  socket.onclose = () => setSocketState(socket.readyState);
  socket.onerror = (e) => {
    console.error(`Socket Error: ${e.message}`);
    setSocketState(socket.readyState);
  };
  socket.onmessage = (m) => {
    setMessage(m.data.toString());
    setSocketState(socket.readyState);
  };

  const send = (msg: string) => {
    socket.send(msg, (err) => {
      if (err != null) {
        console.error(`Socket Error: ${err.message}`);
        setSocketState(socket.readyState);
      }
    });
  };

  createEffect(() => {
    if (
      socketState() === WebSocket.CLOSED &&
      gameState.status !== GameStatus.Pending
    )
      resetGS();
    if (socketState() === WebSocket.OPEN) {
      startGame();
      setStatusGS(GameStatus.Ongoing);
    }
  });

  createEffect(() => {
    if (message() === "") return;
    const data: any = JSON.parse(message());
    updateGS(data);
  });

  createEffect(() => {
    console.log(gameState);
    if (gameState.prompt === null) return;
    const role = getRole(gameState.prompt.message);
    if (role != null) setHumanIsCodemaster(role[0] === codemasterRole);
  });

  return (
    <main>
      <Title>Arena</Title>
      <h1>Codenames Arena</h1>
      <div class="container mb-4">
        <Table socketState={socketState}></Table>
        <div class="row align-items-center justify-content-center gap-4">
          <Key humanIsCodemaster={humanIsCodemaster}></Key>
          <div class="col mx-auto">
            <Prompt send={send}></Prompt>
          </div>
        </div>
      </div>
    </main>
  );
}

import { cardTypeToColor } from "~/util/general";

export default function Cell(props: {
  word: string;
  color: string;
  guessed: boolean;
}) {
  return (
    <div class={props.word === "" ? "" : "p-1"} style="padding: 0.1rem">
      <div
        class={`col ${
          props.word === "" ? "p-1 rounded" : "p-4 rounded"
        } shadow-sm text-center overflow-visible`}
        style={{
          "background-color": cardTypeToColor(props.color, props.guessed),
          color: props.guessed ? "#fff" : "#000",
        }}
      >
        {props?.word}
      </div>
    </div>
  );
}

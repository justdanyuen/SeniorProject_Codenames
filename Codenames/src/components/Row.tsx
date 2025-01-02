import { For } from "solid-js";
import Cell from "./Cell";

export default function Row(props: {
  words: { word: string; color: string; guessed: boolean }[];
  isKey: boolean;
}) {
  return (
    <div class={`row row-cols-${props.isKey ? "auto" : "5"}`}>
      <For each={props.words}>
        {(word: any) => (
          <Cell
            word={props.isKey ? "" : word.word.toLowerCase()}
            color={word.color}
            guessed={props.isKey ? true : word.guessed}
          ></Cell>
        )}
      </For>
    </div>
  );
}

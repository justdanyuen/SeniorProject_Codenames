import { A } from "@solidjs/router";
import { Title } from "solid-start";

export default function Home() {
  return (
    <main>
      <Title>Codenames - AI Competition</Title>
      <h1>Codenames AI Competition</h1>
      <div id="description">
        <p>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed
          consectetur, nisl vitae ultricies lacinia, nisl nisl aliquet nisl, nec
          aliquam nisl nisl nec nunc. Nulla facilisi. Nulla facilisi.
        </p>
      </div>
      <A class="btn" href="/create-arena">
        Create an Arena
      </A>
    </main>
  );
}

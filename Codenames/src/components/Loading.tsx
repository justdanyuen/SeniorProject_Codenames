export default function Loading(props: { text?: string }) {
  return (
    <div class="d-flex justify-content-center align-items-center">
      <span
        class="spinner-border spinner-border-sm text-secondary mx-2"
        role="status"
      ></span>
      <b class="mb-0 text-secondary">{props.text ?? "Loading..."}</b>
    </div>
  );
}

import { Component, Show } from 'solid-js';

interface OutputPanelProps {
  label: string;
  value?: string | null;
  error?: string | null;
}

export const OutputPanel: Component<OutputPanelProps> = (props) => {
  return (
    <div class="output__section">
      <div class="output__label">{props.label}</div>
      <Show when={props.error}>
        <div class="output__value output__value--error">{props.error}</div>
      </Show>
      <Show when={!props.error && props.value}>
        <div class="output__value">{props.value}</div>
      </Show>
    </div>
  );
};

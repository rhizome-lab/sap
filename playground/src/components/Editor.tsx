import { Component } from 'solid-js';

interface EditorProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export const Editor: Component<EditorProps> = (props) => {
  return (
    <div class="editor">
      <textarea
        class="editor__input"
        value={props.value}
        onInput={(e) => props.onChange(e.currentTarget.value)}
        placeholder={props.placeholder}
        spellcheck={false}
      />
    </div>
  );
};

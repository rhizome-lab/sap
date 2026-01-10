import { Component, createSignal, For, Show } from 'solid-js';

interface AstNode {
  type: string;
  value?: string | number;
  children?: AstNode[];
}

interface AstViewerProps {
  ast: AstNode;
}

export const AstViewer: Component<AstViewerProps> = (props) => {
  return (
    <div class="ast-viewer">
      <AstNodeView node={props.ast} depth={0} />
    </div>
  );
};

interface AstNodeViewProps {
  node: AstNode;
  depth: number;
}

const AstNodeView: Component<AstNodeViewProps> = (props) => {
  const [expanded, setExpanded] = createSignal(props.depth < 3);
  const hasChildren = () => props.node.children && props.node.children.length > 0;

  return (
    <div class="ast-node" style={{ "margin-left": props.depth > 0 ? '16px' : '0' }}>
      <div class="ast-node__header" onClick={() => hasChildren() && setExpanded(!expanded())}>
        <span class="ast-node__toggle">
          <Show when={hasChildren()}>
            {expanded() ? '▼' : '▶'}
          </Show>
        </span>
        <span class="ast-node__type">{props.node.type}</span>
        <Show when={props.node.value !== undefined}>
          <span class="ast-node__value">
            {typeof props.node.value === 'string' ? `"${props.node.value}"` : props.node.value}
          </span>
        </Show>
      </div>
      <Show when={hasChildren() && expanded()}>
        <div class="ast-node__children ast-node__children--expanded">
          <For each={props.node.children}>
            {(child) => <AstNodeView node={child} depth={props.depth + 1} />}
          </For>
        </div>
      </Show>
    </div>
  );
};

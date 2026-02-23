import { useMemo } from 'react';
import {
  ReactFlow,
  type Node,
  type Edge,
  Background,
  Controls,
  Handle,
  Position,
  type NodeProps,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import type { ThoughtNode } from '../types/agent';

interface ThoughtNodeData {
  label: string;
  content: string;
  reasoning: string;
  score: number;
  expected_impact: number;
  confidence: number;
  selected: boolean;
  depth: number;
  isRoot?: boolean;
  [key: string]: unknown;
}

function ThoughtNodeComponent({ data }: NodeProps<Node<ThoughtNodeData>>) {
  const isSelected = data.selected;
  const isRoot = data.isRoot;

  if (isRoot) {
    return (
      <div className="px-5 py-3 rounded-xl border-2 border-purple-500/60 bg-purple-950/80 max-w-[320px] shadow-lg shadow-purple-500/10">
        <p className="text-sm text-purple-200 font-semibold text-center leading-snug">{data.content}</p>
        <Handle type="source" position={Position.Bottom} className="!bg-purple-500" />
      </div>
    );
  }

  const scoreColor =
    data.score > 0.6 ? 'text-green-400' : data.score > 0.3 ? 'text-yellow-400' : 'text-red-400';

  return (
    <div
      className={`px-4 py-3 rounded-xl border-2 max-w-[280px] transition-all ${
        isSelected
          ? 'border-blue-500 bg-blue-950/80 shadow-lg shadow-blue-500/20'
          : 'border-gray-700 bg-gray-900/90'
      }`}
    >
      <Handle type="target" position={Position.Top} className="!bg-gray-600" />
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className={`text-lg font-bold ${scoreColor}`}>{data.score.toFixed(2)}</span>
        {isSelected && (
          <span className="text-xs bg-blue-600 text-white px-2 py-0.5 rounded-full">
            Selected
          </span>
        )}
      </div>
      <p className="text-sm text-gray-200 font-medium leading-snug">{data.content}</p>
      {data.reasoning && (
        <p className="text-xs text-gray-400 mt-1 leading-snug line-clamp-2">{data.reasoning}</p>
      )}
      <div className="flex gap-3 mt-2 text-xs text-gray-500">
        <span>Impact: {(data.expected_impact * 100).toFixed(0)}%</span>
        <span>Conf: {(data.confidence * 100).toFixed(0)}%</span>
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-gray-600" />
    </div>
  );
}

const nodeTypes = { thought: ThoughtNodeComponent };

interface Props {
  thoughtNodes: ThoughtNode[];
  selectedPath: string[];
  rootThought?: string;
  isGenerating?: boolean;
}

export default function ThoughtTreeViz({ thoughtNodes, selectedPath, rootThought, isGenerating }: Props) {
  const { nodes, edges } = useMemo(() => {
    const flowNodes: Node<ThoughtNodeData>[] = [];
    const flowEdges: Edge[] = [];

    if (thoughtNodes.length === 0) {
      return { nodes: flowNodes, edges: flowEdges };
    }

    // Root node
    const totalWidth = Math.max(thoughtNodes.length * 320, 640);
    const centerX = totalWidth / 2;

    flowNodes.push({
      id: 'root',
      type: 'thought',
      position: { x: centerX - 160, y: 0 },
      data: {
        label: 'Root',
        content: rootThought || 'How to improve model performance?',
        reasoning: '',
        score: 0,
        expected_impact: 0,
        confidence: 0,
        selected: true,
        depth: 0,
        isRoot: true,
      },
    });

    // Layout: spread branches horizontally
    const branchWidth = 320;
    const startX = centerX - ((thoughtNodes.length - 1) * branchWidth) / 2 - 140;

    thoughtNodes.forEach((node, i) => {
      const x = startX + i * branchWidth;
      const y = 120;
      const isSelected = selectedPath.includes(node.id);

      flowNodes.push({
        id: node.id,
        type: 'thought',
        position: { x, y },
        data: {
          label: node.id,
          content: node.content,
          reasoning: node.reasoning,
          score: node.score,
          expected_impact: node.expected_impact,
          confidence: node.confidence,
          selected: isSelected,
          depth: 1,
        },
      });

      flowEdges.push({
        id: `root-${node.id}`,
        source: 'root',
        target: node.id,
        style: { stroke: isSelected ? '#3b82f6' : '#4b5563', strokeWidth: isSelected ? 3 : 1 },
        animated: isSelected,
      });

      // Children
      if (node.children && node.children.length > 0) {
        const childWidth = branchWidth / (node.children.length + 1);
        node.children.forEach((child, j) => {
          const cx = x - branchWidth / 2 + (j + 1) * childWidth;
          const cy = y + 200;
          const childSelected = selectedPath.includes(child.id);

          flowNodes.push({
            id: child.id,
            type: 'thought',
            position: { x: cx, y: cy },
            data: {
              label: child.id,
              content: child.content,
              reasoning: child.reasoning,
              score: child.score,
              expected_impact: child.expected_impact,
              confidence: child.confidence,
              selected: childSelected,
              depth: 2,
            },
          });

          flowEdges.push({
            id: `${node.id}-${child.id}`,
            source: node.id,
            target: child.id,
            style: {
              stroke: childSelected ? '#3b82f6' : '#4b5563',
              strokeWidth: childSelected ? 3 : 1,
            },
            animated: childSelected,
          });
        });
      }
    });

    return { nodes: flowNodes, edges: flowEdges };
  }, [thoughtNodes, selectedPath, rootThought]);

  if (thoughtNodes.length === 0) {
    return (
      <div className="h-[400px] flex flex-col items-center justify-center text-gray-500 text-sm border border-gray-800 rounded-xl gap-3">
        {isGenerating ? (
          <>
            <div className="w-8 h-8 border-2 border-pink-500/30 border-t-pink-500 rounded-full animate-spin" />
            <p>Generating Tree of Thought strategies...</p>
          </>
        ) : (
          <>
            <svg className="w-12 h-12 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
            </svg>
            <p>Tree of Thought visualization will appear during experiment</p>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="h-[500px] border border-gray-800 rounded-xl overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
        minZoom={0.3}
        maxZoom={1.5}
      >
        <Background color="#374151" gap={20} />
        <Controls />
      </ReactFlow>
    </div>
  );
}

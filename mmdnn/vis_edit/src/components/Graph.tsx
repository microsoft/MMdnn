import * as React from "react";
import "./Graph.css";
import { IRNode } from "../types";
import * as dagre from 'dagre';
import { Node, Edge, GraphEdge, NodeConfig } from 'dagre';
import { getColor } from "../helper";

// export interface Node {
//     class?:string
// }
const node_w: number = 110, node_h: number = 20, margin: number = 10;
export interface Props {
    nodes: IRNode[],
    selectedLayer:string|undefined,
    width:number
}
export interface State {
    x: number,
    y: number,
    scale: number,
    nodes: Node[],
    edges: GraphEdge[],
    h: number,
    w: number,
}
export default class Graph extends React.Component<Props, State> {
    public graphWindow: any; x0:number;y0:number;
    constructor(props: Props) {
        super(props)
        this.pan = this.pan.bind(this)
        this.drawNodes = this.drawNodes.bind(this)
        this.state = {
            x: 1,
            y: 1,
            scale: 1,
            nodes: [],
            edges: [],
            h: 0,
            w: 0
        }
    }
    getDag(IRnodes: IRNode[]) {

        const g = new dagre.graphlib.Graph();
        g.setGraph({ ranksep: 10, marginx: margin, marginy: margin, rankdir: "TB" });
        g.setDefaultEdgeLabel((edge: Edge) => { return {} });
        IRnodes.forEach((node: any) => {
            let name = node.name.split('/')
            let label = name[name.length-1]
            g.setNode(node.name, { label: label, width: node_w, height: node_h, op: node.op, full_label:node.name })
            if (node.input) {
                node.input.forEach((input: string) => {
                    g.setEdge(input, node.name)
                })
            }
        })
        dagre.layout(g)
        let nodes: Node[] = []
        let edges: GraphEdge[] = []
        g.nodes().forEach((v) => {
            if (g.node(v)) {
                nodes.push(g.node(v))
            }
        })
        g.edges().forEach((e) => {
            let edge: GraphEdge = g.edge(e)
            edge['from'] = e.v
            edge['to'] = e.w
            edges.push(edge)
        });
        let h = Number(g.graph().height),
            w = Number(g.graph().width)
        return { nodes, edges, h, w }
    }
    drawNodes(nodes: NodeConfig[]) {
        let strokeSize = 4
        return (<g className="nodes" >
            {nodes.map((node: Node) => {
                let selected:string = node.full_label==this.props.selectedLayer?"red":"none"
                // console.info(selected, node.full_label, this.props.selectedLayer)
                return <g key={node.full_label} transform={`translate (${node.x - node_w / 2}, ${node.y - node_h / 2})`}>
                    <rect width={node_w} height={node_h}
                        style={{ fill: getColor(node.op), strokeWidth: strokeSize }} />
                    <text textAnchor="middle"
                        fontSize={node_h * 0.5}
                        x={node_w / 2}
                        y={node_h * 0.6}>
                        {node.label}
                    </text>
                    <rect width={node_w + 2*strokeSize} height={node_h + 2*strokeSize}
                         transform={`translate (${-strokeSize}, ${-strokeSize})`}
                        style={{ fill: "none", stroke: selected, strokeWidth: strokeSize}} />
                </g>
            })}
        </g>)
    }
    oneEdge(edge: GraphEdge, i: number) {
        let { points, from, to } = edge
        let len = points.length
        if (len === 0) { return }
        let start = `M ${points[0].x} ${points[0].y}`
        let vias = [];
        for (let i = 0; i < len - 2; i += 2) {
            let cPath = [0, 1, 2].map(k => `${points[i + k].x} ${points[i + k].y}`)
            vias.push(`M ${points[i].x} ${points[i].y} C ${cPath}`)

        }
        let pathData = `${start}  ${vias.join(' ')}`
        return <g className='link' key={`${from}->${to}`}>
            <path
                key={`${edge.from}->${edge.to}`}
                d={pathData}
                stroke="gray"
                fill='transparent'
                strokeWidth="2"
            // markerEnd="url(#arrow)" 
            />
            {/* <path
                key={`${edge.from}->${edge.to}_mask`}
                d={pathData}
                stroke="transparent"
                fill='transparent'
                strokeWidth="6" /> */}
        </g>

    }
    drawEdges(edges: GraphEdge[]) {
        return (<g className="edges">
            {edges.map((edge: GraphEdge, i: number) => {
                return this.oneEdge(edge, i)

            })}
        </g>)
    }
    scroll(e: any) {
        if (e.shiftKey) {
            this.zoom(e.deltaY)
        } else {
            let { y } = this.state
            this.setState({ y: y - e.deltaY })
        }
    }
    zoom(delta: number) {
        let { scale } = this.state
        scale *= (delta > 0 ? 1.1 : 0.9)
        this.setState({ scale })
    }
    mouseDown(e:any){
        e.stopPropagation()
        e.preventDefault()
        console.info("graph mouse down")
        document.addEventListener("mousemove", this.pan)
        this.x0 = e.clientX
        this.y0 = e.clientY
    }
    pan(e:any){
        let {x, y } = this.state
        x += e.clientX - this.x0
        y += e.clientY - this.y0
        this.x0 = e.clientX
        this.y0 = e.clientY
        console.info("pan", e)
        this.setState({x, y})
    }
    mouseUp(e:any){
        e.stopPropagation()
        e.preventDefault()
        document.removeEventListener("mousemove", this.pan)
    }
    componentWillReceiveProps(nextProps: Readonly<Props>, nextContext: any) {
        if (this.props.nodes.length != nextProps.nodes.length) {
            let { nodes: IRnodes } = nextProps
            let { nodes, edges, w, h } = this.getDag(IRnodes)
            // let scale: number = Math.min((this.graphWindow.clientHeight - 2 * margin) / h, (this.props.width - 2 * margin) / w)
            let x: number = margin + 0.5 * this.props.width - 0.5 * w
            let y: number = margin
            this.setState({x, y, nodes, edges, w, h })
        }else if(this.props.selectedLayer != nextProps.selectedLayer){
            let {nodes, scale } = this.state
            let selectedNode = nodes.filter(
                (node:NodeConfig)=>node.full_label==nextProps.selectedLayer
            )[0]
            let { y:node_y } = selectedNode
            // let {x:node0_x, y:node0_y} = nodes[0]
            this.setState({
                // x: x-(node_x-node0_x)*scale , 
                y: - node_y * scale + 0.4 * this.graphWindow.clientHeight})
        }
    }
    render() {
        let { nodes, edges, x, y, scale, h, w } = this.state
        if (nodes.length > 0) {
            // let { nodes, edges} = this.getDag(IRnodes)
            let svg_h = Math.max(h, this.graphWindow.clientHeight)
            let svg_w = Math.max(w, this.props.width)
            // let svg_h = this.graphWindow.clientHeight
            // let svg_w = this.props.width
            return (
                <div className="graphWindow"
                    ref={(ref) => { this.graphWindow = ref }}
                    onWheel={this.scroll.bind(this)}
                    onMouseDown={this.mouseDown.bind(this)}
                    onMouseUp={this.mouseUp.bind(this)}
                    onMouseLeave = {this.mouseUp.bind(this)}
                    // onDragStart={}
                    
                    // onKeyDown={(e) => { this.shiftDown = e.shiftKey }}
                    // onKeyUp={(e) => { this.shiftDown = false }}
                    tabIndex={0}
                >
                    <svg width={svg_w} height={svg_h} >
                        <g className="graph"
                            transform={`translate(${x}, ${y}) scale(${scale})`}
                        >
                            {this.drawEdges(edges)}
                            {this.drawNodes(nodes)}
                        </g>
                    </svg>
                </div>)
        } else {
            return <div className="graphWindow" ref={(ref) => { this.graphWindow = ref }} />
        }

    }
}
import React, { Component } from 'react';
import { dictLayer } from '../../assets/def';
import {LAYER_FONT} from "../../assets/const"
export default class InsideLayer extends Component {
    constructor(props) {
        super(props)
        this.state = { b: -1 }
    }
    // onClick(e, d) {
    //     let { layer } = this.props
    //     let arr = dictLayer[layer.name]
    //     e.preventDefault()
    //     e.stopPropagation()
    //     let b = arr.indexOf(d)
    //     this.setState({ b })
    //     this.props.onSelectAct(this.props.a, b)
    // }
    render() {
        let { layer, a, onSelectAct, width } = this.props
        // console.info(layer)
        // let arr=dictLayer[layer.name]
        let arr = layer.acts.map(d => d.name)
        let height = `${arr.length * 40 + 30}px`
        return <rect height={height} width={width * 0.9}
            x={width * 0.05}
            y={LAYER_FONT*2}
            style={{fill: '#555'}}
        >
            {/* <InOut x={50} y={15} />
            {arr.map((d, i) => <g key={'Act' + i} onClick={e => this.onClick(e, d)}>
                <Act x={25} y={25 + i * 40} name={d} highlight={this.state.b == i} />
                <InOut x={50} y={55 + i * 40} />
            </g>)} */}
        </rect>
    }
}

class InOut extends Component {
    constructor(props) {
        super(props)
        this.state = { showSize: true }
    }
    onClick(e) {
        let { showSize } = this.state
        showSize = !showSize
        e.preventDefault()
        e.stopPropagation()
        this.setState({ showSize })
    }
    render() {
        let data = <g>
            <line x2='40' y2='0' stroke='yellow' />
            <text fill='yellow' x='40' fontSize='12px'>x, y, z</text>
        </g>
        let { showSize } = this.state
        return <g transform={`translate(${this.props.x},${this.props.y})`}>
            <line y1="-10" y2="10" stroke="yellow" />
            <circle
                r='3'
                fill="yellow" />
            <circle
                r='5'
                onClick={e => this.onClick(e)}
                opacity='0' />
            {showSize ? data : null}
            {showSize ? data : null}
        </g>
    }
}
class Act extends Component {
    constructor(props) {
        super(props)
    }
    render() {
        return <g
            transform={`translate(${this.props.x},${this.props.y})`}>
            <rect
                width='50'
                height='20'
                rx='3'
                stroke='white'
                strokeWidth={this.props.highlight ? 3 : 0}
                fill='#17a9a8' />
            <text fill='white' x='25' y='18' textAnchor="middle">
                {this.props.name}
            </text>
        </g>
    }
}

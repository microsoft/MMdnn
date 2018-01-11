import React, { Component } from "react";
import { getLayerColor } from "../../assets/def";
import { DOT_SIZE, THEME_COLOR, COLORS, LAYER_FONT, THEME_DARK } from "../../assets/const";
import { layerStyle } from '../../style/styleObj';
import ReactDOM from "react-dom";
import InsideLayer from './InsideLayer';


const MARGIN = 20;
export default class Layer extends Component {
  constructor(props) {
    super(props);
    // this.onDragStart = this.onDragStart.bind(this);
    // this.onDrag = this.onDrag.bind(this);
    // this.onDragEnd = this.onDragEnd.bind(this);
    this.onLClick = this.onLClick.bind(this)
    this.onRClick = this.onRClick.bind(this)
    let { x, y, width } = this.props
    this.state = {
      x,
      y,
      z: 1,
      layer: this.props.layer,
      // height:30,
      width,
      forCopy: false
    };


  }
  // onDragStart(e) {
  //   // this.setState({
  //   //   relX: e.pageX,
  //   //   relY: e.pageY,
  //   //   from: this.props.a,
  //   //   z: 100
  //   // });
  //   // document.addEventListener("mousemove", this.onDrag);
  //   // document.addEventListener("mouseup", this.onDragEnd);
  //   e.preventDefault();
  //   console.info("drag a dot")
  // }
  // onDragEnd(e) {
  //   const { from, to } = this.state
  //   this.onMoveLayers(from, to)
  //   document.removeEventListener("mousemove", this.onDrag);
  //   document.removeEventListener("mouseup", this.onDragEnd);
  //   e.preventDefault();
  //   this.setState({
  //     x: this.props.x,
  //     y: this.props.y,
  //     z: 1
  //   })
  // }
  // onMoveLayers(from, to) {
  //   this.props.onMoveLayers(from, to)
  // }
  // onDrag(e) {
  //   let { from, relX, relY } = this.state
  //   let to = Math.round((e.pageY - relY) / 35) + from
  //   this.setState({
  //     x: e.pageX - relX + this.props.x,
  //     y: e.pageY - relY + this.props.y,
  //     to
  //   });
  //   e.preventDefault();
  // }
  onLClick(e) {
    e.preventDefault()
    e.stopPropagation()
    let isMulti = e.altKey
    
    if (!isMulti) {
      // this.props.onSelectAct(this.props.a, -1)
      // this.props.onFolded(this.props.layer.id)
      
      this.props.onSelectLayer(this.props.layer.id)
    } else{
      this.props.addForCopy(this.props.layer.id, isMulti)
    }

  }
  onRClick(e){
    e.preventDefault()
    //e.stopPropagation()
    this.props.addForCopy(this.props.layer.id, false)
  }
  addLink(to) { }
  render() {
    const { z } = this.state
    const {width, height, layer, info } = this.props
    const {x, y} = layer.pos
    let fillLayer = (this.props.selected) ? COLORS[1] : COLORS[0]
    let strokeWidth = (this.props.isForCopy) ? "7" : "0"
    // let detail = fold ? <g /> : <InsideLayer
    //   width={width}
    //   layer={layer}
    //   a={a}
    // /* onSelectAct={self.props.onSelectAct} */
    // />
    return (
      <g transform={`translate(${x - width / 2}, ${y - height / 2})`}>
        {/* {(this.props.a == 0) ?
          <div style={{ width, transform: `translate(${this.props.width * 0.5}px,0px)` }}>
            <span
              className="glyphicon glyphicon-plus-sign"
              aria-hidden="true">
            </span>
            {this.props.inputShape}
          </div>
          : <span />} */}
          <g >
            <rect
              className={layer.name}
              onClick={this.onLClick}
              onContextMenu={this.onRClick}
              width={width}
              height={height}
              stroke={COLORS[3]}
              strokeWidth={strokeWidth}
              rx='3'
              ry='3'
              fill={fillLayer}
            >
            </rect>
            {/*{detail}*/}
            <text
              className='layerName'
              textAnchor="middle"
              fontSize={height*0.5}
              x={width / 2}
              y={height*0.6}
              onClick={this.onLClick}
            >
              {`${layer.name}`}
            </text>
            <text
              className='count_params'
              textAnchor="end"
              fontSize={height*0.4}
              x={width*0.95}
              y={height*0.95}
            >
              {`p:${layer.params==undefined?'':layer.params}`}
            </text>
          </g>
        <circle className='dot-in'
          /* onClick={e => this.props.addLinkTo(layer.id)} */
          r={DOT_SIZE / 2}
          cx={width / 2}
          style={{
            fill: THEME_COLOR
          }}
        />
        <circle className='dot-in-mask'
          onClick={e => { this.props.addLinkTo(layer.id) }}
          r={DOT_SIZE / 2 * 4}
          cx={width / 2}
          style={{ fill: 'transparent' }}
        />
        <circle className='dot-out'
          r={DOT_SIZE / 2}
          cx={width / 2}
          cy={height}
          style={{ fill: THEME_COLOR }}
        />
        <circle className='dot-out-mask'
          r={DOT_SIZE / 2 * 4}
          cx={width / 2}
          cy={height}
          style={{ fill: 'transparent' }}
          onClick={e => this.props.addLinkFrom(layer.id)}
        />
        <text
          transform={`translate(${this.props.width * 0.55},${this.props.height*1.5})`}
          fill='white'
        >
          {/* <span className="glyphicon glyphicon-circle-arrow-down" aria-hidden="true"></span> */}
          {(layer.outShape) ?
            `(${layer.outShape.map(d => d || 'none').join(',')})`
            : ''}
        </text>
      </g>
    );
  }
}

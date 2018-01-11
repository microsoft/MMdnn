import React, { Component } from "react";
import { normalize, denormalize } from 'normalizr';
import { layersSchema } from '../ulti/schema';
import { getLayerColor, dictLayer } from "../../assets/def";
import { THEME_COLOR, COLORS } from "../../assets/const";
import Layer from "./Layer";
import ParBoard from './ParBoard';
import * as dagre from 'dagre';

const { remote } = require('electron')
const { Menu, MenuItem } = remote
const menu = new Menu()

const MARGIN = 20, NODE_H = 30, NODE_W = 200;

export default class GraphView extends Component {
  constructor(props) {
    super(props);
    // this.onChangePar = this.onChangePar.bind(this)
    this.link = { from: undefined, to: undefined }
    // let { edges, SVGh, SVGw } = this.getDag(this.props.layers)
    this.state = {
      svgScale: 1,
      forCopys: [],
      selectionBox: {
        x: 0,
        y: 0,
        width: 0,
        height: 0
      },
      // svg: { edges, SVGh, SVGw }
    }
    this.drawLink = this.drawLink.bind(this)
    this.onDelLink = this.onDelLink.bind(this)
    this.addLinkFrom = this.addLinkFrom.bind(this)
    this.addLinkTo = this.addLinkTo.bind(this)
    this.onAddLink = this.onAddLink.bind(this)
    this.zoomSVG = this.zoomSVG.bind(this)
    this.drawLayers = this.drawLayers.bind(this)
    this.addForCopy = this.addForCopy.bind(this)

    this.selectionEnd = this.selectionEnd.bind(this)
    this.selectionOn = this.selectionOn.bind(this)
    this.selectionStart = this.selectionStart.bind(this)
    this.findInBox = this.findInBox.bind(this)
    let self = this

    menu.append(new MenuItem({
      label: 'copy',
      click() {
        self.props.onCopyLayers(self.state.forCopys)
        self.setState({ forCopys: [] })
      }
    }))
    menu.append(new MenuItem({
      label: 'delete', click() {
        self.props.onDelLayers(self.state.forCopys)
        self.setState({ forCopys: [] })
      }
    }))
    // menu.append(new MenuItem({
    //   label: 'shallow copy',
    //   click() {
    //     self.props.onCopyLayers(self.state.forCopys, false)
    //     self.setState({ forCopys: [] })
    //   }
    // }))
    // menu.append(new MenuItem({ type: 'separator' }))
    menu.append(new MenuItem({
      label: 'share',
      click() {
        self.props.onShareModule(self.state.forCopys)
        self.setState({ forCopys: [] })
      }
    }))
    menu.append(new MenuItem({
      label: 'build a submodule',
      click() {
        // console.info('copys for building', self.state)
        self.props.onBuildModule(self.state.forCopys)
        self.setState({ forCopys: [] })
      }
    }))


    window.addEventListener('contextmenu', (e) => {
      e.preventDefault()
      e.stopPropagation()
      // console.info(e.button)
      menu.popup(remote.getCurrentWindow())
    }, false)
  }
  //onMoveLayers(from, to) {
  //  this.props.onMoveLayers(from, to)
  //  this.setState({ update: 1 })
  //}

  addForCopy(id, isMulti) {
    if (isMulti) {
      let { forCopys } = this.state
      forCopys.push(id)
      this.setState({ forCopys })
    } else if (this.state.forCopys.length == 0) {
      this.setState({ forCopys: [id] })
    }
  }
  onAddLink(from, to) {
    this.props.onAddLink(from, to)
    this.link = {}
  }
  addLinkFrom(from) {
    this.link.from = from
    // console.info(from)
  }
  addLinkTo(to) {
    if (this.link.from != undefined && this.link.to == undefined) {
      this.onAddLink(this.link.from, to)
    }
    // console.info(to)
  }

  getDag = (layers) => {
    const g = new dagre.graphlib.Graph();
    g.setGraph({});
    g.setDefaultEdgeLabel(() => { return {}; });

    const checkInput = (parent, input, layersObj) => {
      let inLayer = layersObj[input]
      if (inLayer.parent != parent &&
        inLayer.parent != '' &&
        layersObj[inLayer.parent].folded) {
        return checkInput(parent, inLayer.parent, layersObj)
      }
      return input
    }

    const setDag = (layers, g, layersObj) => {
      layers.forEach(layer => {
        if (layer.func != "module") {
          g.setNode(layer.id, { label: `${layer.id}`, width: NODE_W, height: NODE_H })
          layer.inputs.forEach(input => {
            input = checkInput(layer.parent, input, layersObj)
            g.setEdge(input, layer.id)
          })
        } else if (layer.func == 'module') {
          if (layer.folded) {
            g.setNode(layer.id, { label: `${layer.id}`, width: NODE_W, height: NODE_H })
            layer.inputs.forEach(input => {
              input = checkInput(layer.parent, input, layersObj)
              g.setEdge(input, layer.id)
            })
          } else {
            setDag(layer.layers, g, layersObj)
          }
        }
      })
    }
    let layersObj = this.props.config.entities.layers || {}

    setDag(layers, g, layersObj)
    dagre.layout(g)

    let edges = []
    // add pos to layers
    const addPos = (layers, g) => {
      layers.forEach(layer => {
        if (layer.func == "module" && !layer.folded) {
          addPos(layer.layers, g)
        } else {
          layer.pos = g.node(layer.id)
        }
      })
    }
    addPos(layers, g)
    
    g.edges().forEach((e) => {
      edges.push({ points: g.edge(e).points, from: e.v, to: e.w })
      // console.log("Edge " + e.v + " -> " + e.w + ": " + JSON.stringify(g.edge(e)));
    });
    let SVGh = g.graph().height
    let SVGw = g.graph().width
    return { layers, edges, SVGh, SVGw }
  }
  onDelLink(from, to) {
    this.props.onDelLink(from, to)
  }
  drawLink(edge,i) {
    let { points, from, to } = edge
    let len = points.length
    if (len == 0) { return }
    let start = `M ${points[0].x} ${points[0].y}`
    let vias = [];
    for (let i = 0; i < len - 2; i += 2) {
      let cPath;
      cPath = [...Array(3).keys()].map(k => `${points[i + k].x} ${points[i + k].y}`)
      cPath = 'C'.concat(cPath)
      vias.push(`M ${points[i].x} ${points[i].y} ${cPath}`)

    }
    vias = vias.join(' ')
    let pathData = `${start}  ${vias}`
    return <g className='link' key={`${edge.from}->${edge.to}`}>
      <path
        key={`${edge.from}->${edge.to}`}
        d={pathData}
        stroke={THEME_COLOR}
        fill='transparent'
        strokeWidth="3"
        markerEnd="url(#arrow)" />
      <path onDoubleClick={e => this.onDelLink(from, to)}
        key={`${edge.from}->${edge.to}_mask`}
        d={pathData}
        stroke="transparent"
        fill='transparent'
        strokeWidth="6" />
    </g>
  }
  onFolded(ids) {
    this.props.onFolded(ids)
  }
  zoomSVG(e) {
    let oldScale = this.state.svgScale
    this.setState({ svgScale: oldScale * e })
  }
  selectionStart(e) {
    if (e.button == 0) {
      e.preventDefault()
      e.stopPropagation()
      this.setState({
        selectionBox: {
          x: e.clientX - 130,
          y: e.clientY + document.getElementById('GraphView').scrollTop,
          width: 0,
          height: 0
        }
      })
      document.getElementById('GraphView').addEventListener("mousemove", this.selectionOn)
      document.getElementById('GraphView').addEventListener("mouseup", this.selectionEnd)
    }
  }
  selectionOn(e) {
    let { x, y } = this.state.selectionBox
    let { svgScale } = this.state
    let deltaX = e.clientX - 130 - x,
      deltaY = e.clientY + document.getElementById('GraphView').scrollTop - y
    layers
    this.findInBox(
      this.props.layers,
      (x - MARGIN) * svgScale + MARGIN,
      (y - MARGIN) * svgScale + MARGIN,
      deltaX * svgScale,
      deltaY * svgScale
    )
    //console.info('selection on', this.state)
    this.setState({
      selectionBox: {
        x,
        y,
        width: deltaX,
        height: deltaY,
      }
    })

  }
  findInBox(layers, x0, y0, w, h) {
    layers.forEach(layer => {
      if (layer.func == 'module' && !layer.folded) {
        this.findInBox(layer.layers, x0, y0, w, h)
      } else {
        let { x, y } = layer.pos
        let xIn = (w > 0 && x0 < x && x < x0 + w) || (w < 0 && x0 + w < x && x < x0)
        let yIn = (h > 0 && y0 < y && y < y0 + h) || (h < 0 && y0 + h < y && y < y0)
        if (xIn && yIn) {
          if (this.state.forCopys.indexOf(layer.id) == -1) {
            this.setState({ forCopys: [...this.state.forCopys, layer.id] })
          }
        }
      }
    })
  }
  selectionEnd(e) {
    this.setState({
      selectionBox: {
        x: 0,
        y: 0,
        width: 0,
        height: 0
      }
    })
    //console.info('selection end', this.state)
    document.getElementById('GraphView').removeEventListener("mousemove", this.selectionOn)
    document.getElementById('GraphView').removeEventListener("mouseup", this.selectionEnd)
  }
  drawLayers(layers, boxMargin = MARGIN) {
    return layers.map(layer => {
      if (layer.func == 'module' && !layer.folded) {

        const searchRange = (layer, xArray, yArray) => {
          if (!layer.layers) { console.info(layer) }
          layer.layers.forEach(d => {
            if (d.func == 'module' && !d.folded) {
              searchRange(d, xArray, yArray)
            } else {
              xArray.push(d.pos.x)
              yArray.push(d.pos.y)
            }
          })
          return { xArray, yArray }
        }
        let { xArray, yArray } = searchRange(layer, [], [])
        // layer.layers.forEach(d => {
        //   if(d.pos){
        //     xArray.push(d.pos.x)
        //   yArray.push(d.pos.y)
        //   }else{
        //     console.info('no pos;', d)
        //   }    
        // })
        let x1 = Math.min(...xArray), x2 = Math.max(...xArray),
          y1 = Math.min(...yArray), y2 = Math.max(...yArray)
        return <g
          /* filter="url(#outline)" */
          className={layer.id}>
          {this.drawLayers(layer.layers, boxMargin * 0.7)}
          <use xlinkHref='#icon-minus'
            onClick={(e) => { e.preventDefault; this.onFolded(layer.id) }}
            transform={`translate(${x2 + 0.5 * NODE_W},
            ${ y1 - 1 * NODE_H})`} />
          <rect
            x={x1 - boxMargin - NODE_W / 2}
            y={y1 - MARGIN - NODE_H / 2}
            width={x2 - x1 + 2 * boxMargin + NODE_W}
            height={y2 - y1 + 2 * MARGIN + NODE_H}
            fill='none'
            stroke='white'
            strokeWidth='3'
            strokeDasharray='5 5'
          />
          {layer.shared ?
            <text
              x={x2 + 0.5 * NODE_W}
              y={y1 - NODE_H}
              fill='white'
            >{`shared`}
            </text>
            : ''}
        </g>

      } else if (layer.func == "module" && layer.folded) {
        let isForCopy = this.state.forCopys.indexOf(layer.id) != -1
        return <g className={layer.id}>
          <Layer
            key={layer.id}
            layer={layer}
            width={NODE_W}
            height={NODE_H}
            /* onFolded={this.onFolded.bind(this)} */
            selectionBox={this.state.selectionBox}
            addLinkFrom={this.addLinkFrom}
            addLinkTo={this.addLinkTo}
            onSelectLayer={this.props.onSelectLayer}
            selected={this.props.selectedLayer == layer.id}
            addForCopy={this.addForCopy}
            isForCopy={isForCopy}
          />
          <use xlinkHref='#icon-plus'
            onClick={(e) => { e.preventDefault; this.onFolded(layer.id) }}
            transform={`translate(${layer.pos.x + 0.5 * NODE_W},
            ${ layer.pos.y - 1 * NODE_H})`} />
          <rect
            x={layer.pos.x - boxMargin - NODE_W / 2}
            y={layer.pos.y - MARGIN - NODE_H / 2}
            width={2 * boxMargin + NODE_W}
            height={2 * MARGIN + NODE_H}
            fill='none'
            stroke='white'
            strokeWidth='3'
            strokeDasharray='5 5'
          />
          {layer.shared ?
            <text
              x={layer.pos.x + 0.5 * NODE_W}
              y={layer.pos.y - NODE_H}
              fill='white'
            >{`shared`}
            </text>
            : ''}
        </g>
      } else {
        let isForCopy = this.state.forCopys.indexOf(layer.id) != -1
        return <Layer
          key={layer.id}
          layer={layer}
          width={NODE_W}
          height={NODE_H}
          /* onFolded={this.onFolded.bind(this)} */
          addLinkFrom={this.addLinkFrom}
          addLinkTo={this.addLinkTo}
          onSelectLayer={this.props.onSelectLayer}
          selected={this.props.selectedLayer == layer.id}
          addForCopy={this.addForCopy}
          isForCopy={isForCopy}
        />
      }
    })
  }
  saveModel() {
    const { dialog } = require('electron').remote
    const fs = require('fs')
    let modelPath = dialog.showSaveDialog(
      {
        title: 'save model',
        defaultPath: 'exportModel',
        filters: [
          { name: 'JSON', extensions: ['json'] }
        ]
      }
    )
    let { config } = this.props
    let layers = denormalize(config.result, layersSchema, config.entities)
    console.info('save layers:', layers)
    fs.writeFile(modelPath, JSON.stringify(layers), function (err) {
      if (err) {
        console.info(err)
      }
    });
  }
  render() {
    let { layers: oldLayers } = this.props
    let { layers, edges, SVGh, SVGw } = this.getDag(oldLayers)
    // let { edges, SVGh, SVGw } = this.state.svg
    let { svgScale } = this.state
    let { x, y, width, height } = this.state.selectionBox
    let selectionBox = <rect
      x={width > 0 ? x : x + width}
      y={height > 0 ? y : y + height}
      width={Math.abs(width)}
      height={Math.abs(height)}
      rx='3'
      ry='3'
      stroke='white'
      fill='none'
    />
    return (
      <div className="GraphView" id="GraphView" >
        <svg
          width={Math.max(SVGw * svgScale + MARGIN * 2, 0.4 * window.innerWidth)}
          height={Math.max(SVGh * svgScale + MARGIN * 2, 0.7 * window.innerHeight)}
          onMouseDown={this.selectionStart}
        >
          <title>GraphView</title>
          <desc>graph of a model</desc>
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto" markerUnits="strokeWidth" viewBox="0 0 20 20">
              {/* <path d="M0,0 L1,5 L0,10 L5,5 z" fill={THEME_COLOR} /> */}
              <polyline points="0,0 5,5 0,10" stroke={THEME_COLOR} fill="none" strokeWidth='2' />
            </marker>
            <symbol id='icon-plus' >
              <path fill='white'
                transform='scale(0.02, 0.02)'
                d="M 512 81.92 c -237.517 0 -430.08 192.563 -430.08 430.08 s 192.563 430.08 430.08 430.08 s 430.08 -192.563 430.08 -430.08 c 0 -237.517 -192.563 -430.08 -430.08 -430.08 Z M 768 563.2 h -204.8 v 204.8 h -102.4 v -204.8 h -204.8 v -102.4 h 204.8 v -204.8 h 102.4 v 204.8 h 204.8 v 102.4 Z">
              </path>
              <circle r='18' cx='9' cy='9' fill='transparent' />
            </symbol>
            <symbol id='icon-minus' >
              <path fill='white'
                transform='scale(0.02, 0.02)'
                d="M 512 81.92 c -237.517 0 -430.08 192.563 -430.08 430.08 s 192.563 430.08 430.08 430.08 s 430.08 -192.563 430.08 -430.08 c 0 -237.517 -192.563 -430.08 -430.08 -430.08 Z M 768 563.2 h -512 v -102.4 h 512 v 102.4 Z">
              </path>
              <circle r='18' cx='9' cy='9' fill='transparent' />
            </symbol>
            <filter id="outline">
              <feFlood floodColor={THEME_COLOR} floodOpacity="0.4" result="base" />
              <feMorphology result="bigger" in="SourceGraphic" operator="dilate" radius={NODE_H} />
              {/* <feColorMatrix result="mask" in="bigger" type="matrix"
                values="0 0 0 0 0
                        0 0 0 0 0
                        0 0 0 0 0
                        0 0 0 1 0" /> */}
              <feComposite result="drop" in="base" in2="mask" operator="in" />
              <feGaussianBlur result="blur" in="drop" stdDeviation="10" />
              <feBlend in="SourceGraphic" in2="blur" mode="normal" />
            </filter>
          </defs>
          {selectionBox}
          <g className="graph"
            transform={`scale(${svgScale}) translate(${MARGIN}, ${MARGIN})`}>
            {/* {this.state.selectionBox.dom} */}
            <g className='links' >
              {edges.map((edge,i) => this.drawLink(edge,i))}
            </g>
            <g id='layers'>
              {this.drawLayers(layers)}
            </g>
          </g>
        </svg>
        <div className="icon btn-group-vertical"
          style={{
            position: "fixed",
            fontSize: '1.2em',
            right: '42.5%',
            bottom: '28.5%',
            color: '#333'
          }}
        >
          <button className="glyphicon glyphicon-save"
            aria-hidden="true"
            onClick={this.saveModel.bind(this)} />
          <br />
          <button className="glyphicon glyphicon-plus"
            aria-hidden="true"
            onClick={e => this.zoomSVG(1.2)} />
          <br />
          <button className="glyphicon glyphicon-minus"
            aria-hidden="true"
            onClick={e => this.zoomSVG(0.8)} />
        </div>
      </div>
    );
  }
}

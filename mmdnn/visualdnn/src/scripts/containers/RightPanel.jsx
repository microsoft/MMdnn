"use babel";

import React, { Component } from "react";
import { NavDropdown, NavItem } from 'react-bootstrap';
import CodePreviewContainer from '../containers/CodePreviewContainer';
import Plot from '../components/Plot';
import FeatureMap from "../components/FeatureMap";


export default class RightPanel extends Component {
  constructor(props) {
    super(props)
    this.code = <CodePreviewContainer selectMode={this.selectMode.bind(this)}/>
    this.plot = <Plot/>
    this.feature =<FeatureMap/>
    // this.feature =<FeatureMap/>
    this.state = { height: `${window.innerHeight-40}px`, mode: this.code }
    this.selectMode = this.selectMode.bind(this)
  }
  selectMode(mode){
    switch(mode){
      case "code":
        this.setState({mode: this.code})
        break
      case "plot":
        this.setState({mode:this.plot})
        break
      case "feature":
        this.setState({mode:this.feature})
    }
  }
  render() {
    return <div className='panel panel-code CodeView' ref={e => this.rightPanel = e}>

      <div className='panel-heading'>
        <select className='modeSelect' onChange={(e)=>this.selectMode(e.target.value)}>
          <option value="code">Code Preview</option>
          <option value="feature">Feature Inspect</option>
          <option value="plot">Plot</option>
        </select>
        {/* <span className="glyphicon glyphicon-remove-sign" aria-hidden="true"></span> */}
      </div>

      <div className='panel-body'>
        {this.state.mode}
      </div>
    </div>
  }
};

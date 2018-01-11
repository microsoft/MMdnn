import brace from "brace";
import AceEditor from "react-ace";
import "brace/mode/python";
import "brace/theme/tomorrow_night_bright";
import React, { Component } from "react";
const fs = require('fs')
const { exec } = require('child_process')
const node_ssh = require('node-ssh')
let ssh = new node_ssh()

const my_callback =`
`

export default class CodePreview extends Component {
    constructor(props) {
        super(props)
        this.code = this.props.code
        this.state = { height: `${window.innerHeight - 40}px` }
    }
    componentDidMount() {
        window.addEventListener('resize', this.resize.bind(this))
        
    }
    resize() {
        let height = `${window.innerHeight - 40}px`
        this.setState({ height })
    }
    onRunCode(){
        this.props.onRunCode(this.code)
    }
    
    changeCode(value) {
        this.code = value
        // console.info("code change")
    }
    render() {
        return <div><AceEditor
            ref={e => { this.aceCode = e }}
            style={{
                width: 0.5 * window.innerWidth,
                height: this.state.height
            }}
            mode="python"
            theme="tomorrow_night_bright"
            onChange={this.changeCode.bind(this)}
            name="ace"
            value={this.props.code}
            fontSize={15}
            editorProps={{ $blockScrolling: true }}
        />
            <button className='btn-run btn' onClick={this.onRunCode.bind(this)}>
                <span className="glyphicon glyphicon-send" aria-hidden="true"></span>
            </button>
        </div>
    }
}
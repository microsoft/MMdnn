import React, { Component } from 'react';
import { Tabs, Tab } from "react-bootstrap";
export default class ParBoard extends Component {
    constructor(props) {
        super(props)
        this.timeOutID;
        // this.state={width: `${0.5*window.innerWidth-100}px`}
    }
    // componentDidMount(){
    //     window.addEventListener('resize', this.resize.bind(this))
    // }
    // resize(){
    //     this.setState({width: `${0.5*window.innerWidth-100}px`})
    // }
    onChangePar(attr) {
        this.props.onChangePar(attr)
        window.clearTimeout(this.timeOutID)
        this.timeOutID = window.setTimeout(this.props.onFetchShapes, 2000)
    }
    render() {
        // let { width } = this.state
        let { attrs } = this.props
        return <Tabs className="panel panel-par" id="nav-par">
            {attrs.map((attr, i) => <Tab eventKey={i} title={attr.tag} key={i}>
                <Pars
                    onChangePar={this.onChangePar.bind(this)}
                    attr={attr} />
            </Tab>)}
        </Tabs>
    }
}

class Pars extends Component {
    constructor(props) {
        super(props)
        this.state = { morePars: false }
    }
    onChangePar(event, key) {
        let { attr } = this.props
        let {pars, parsDefault} = attr
        let value = event.target.value
        if (key in parsDefault) {
            delete parsDefault[key]
            pars[key] = value
        } else {
            pars[key] = value
        }
        this.props.onChangePar(attr)
        // this.setState({ update: 1 })
    }
    btnClick() {
        let { morePars } = this.state
        morePars = !morePars
        this.setState({ morePars })
    }
    render() {
        let { pars, parsDefault } = this.props.attr
        let inputs = []
        let inputsMore = []
        Object.keys(pars)
            .forEach((k, i) => {
                let input;
                if (Array.isArray(pars[k])) {
                    input = <select style={{ color: 'black' }}
                        className="form-control"
                        onChange={event => this.onChangePar(event, k)}>
                        {pars[k].map(d => <option value={d}>{d}</option>)}
                    </select>
                } else {
                    input = <input
                        type="text"
                        className="form-control"
                        value={pars[k]}
                        onChange={event => this.onChangePar(event, k)}
                    />
                }
                inputs.push(<div className="input-group input-group-sm" key={'par' + i}>
                    <span className="input-group-addon">{k}</span>
                    {input}
                </div>)
            })
        Object.keys(parsDefault)
            .forEach((k, i) => {
                let input;
                if (Array.isArray(parsDefault[k])) {
                    input = <select style={{ color: 'black' }}
                        className="form-control"
                        onChange={event => this.onChangePar(event, k)}>
                        {parsDefault[k].map(d => <option value={d}>{d}</option>)}
                    </select>
                } else {

                    input = <input
                        type="text"
                        className="form-control"
                        value={parsDefault[k]}
                        onChange={event => this.onChangePar(event, k)} />
                }
                inputsMore.push(<div className="input-group-sm input-group" key={'parDefault' + i}>
                    <span className="input-group-addon">{k}</span>
                    {input}
                </div>)
            })
        return <div className="pars">
            {inputs}
            {this.state.morePars ? inputsMore : null}
            <button
                type="button" className="btn btn-sm btn-primary"
                style={{ outline: 'none' }}
                onClick={this.btnClick.bind(this)}>
                More Pars
            </button>
        </div>
    }
}
import React, { Component } from 'react';
import {
    Nav,
    NavDropdown,
    NavItem,
    MenuItem,
    Glyphicon,
    Modal,
    InputGroup,
    FormControl,
    Button,
    Tabs,
    Tab
} from 'react-bootstrap';
import { dictLayer, KERAS_DATASETS, KERAS_MODELS, dictCate } from '../../assets/def';
import axios from 'axios';
const fs = require('fs')


export default class ToolBox extends Component {
    constructor(props) {
        super(props)
        this.state = {
            compileShow: false,
            runShow: false
        }
        this.compile = {}
        this.fit = {}
        this.callbacks = {}
        this.handleSubmit = this.handleSubmit.bind(this)
    }
    handleSelect(key) {
        let [type, name] = key.split('_')
        switch (type) {
            case 'layer':
                this.onAddLayer(name);
                break
            case 'dataset':
                this.onAddInput(name);
                break;
            case 'compile':
                let { compileShow } = this.state
                compileShow = !compileShow
                this.setState({ compileShow })
                break;
            case 'run':
                let { runShow } = this.state
                runShow = !runShow
                this.setState({ runShow })
            default:
                break
        }
    }
    onAddLayer(layer) {
        this.props.onAddLayer(layer)
    }
    onAddInput(name) {
        this.props.onAddKerasDataset(name)
    }
    compileClose() {
        this.setState({ compileShow: false })
    }
    runClose() {
        this.setState({ runShow: false })
    }
    getCompileModal() {
        let { compile } = this.props
        let type = 'compile'
        return <Modal show={this.state.compileShow} onHide={this.compileClose.bind(this)}>
            <Modal.Header closeButton>
                <Modal.Title>Compile Settings</Modal.Title>
                <Modal.Body>
                    <form onSubmit={event => this.handleSubmit(event, type)}>
                        {Object.keys(compile).map((key, i) => <InputGroup key={`${key}_${i}`} >
                            <InputGroup.Addon>{key}</InputGroup.Addon>
                            {Array.isArray(compile[key]) ?
                                <select
                                    style={{ color: 'black' }}
                                    ref={(input) => { this.compile[key] = input }}>
                                    {compile[key].map(d =>
                                        <option value={d} key={d}>{d}</option>
                                    )}
                                </select> :
                                <input
                                    type='text'
                                    defaultValue={compile[key]}
                                    ref={(input) => { this.compile[key] = input }} />}
                        </InputGroup>)}
                        <Button type='submit'>
                            Submit
                        </Button>
                    </form>
                </Modal.Body>
            </Modal.Header>
        </Modal>
    }
    getRunModal() {
        let { fit, callbacks } = this.props.run
        let type = 'run'
        // return <Modal show={this.state.runShow} onHide={this.runClose.bind(this)}>
        //     <Modal.Header closeButton>
        //         <Modal.Title>Run Settings</Modal.Title>
        //         <Modal.Body>
        //             {/* <form onSubmit={event => this.handleSubmit(event, type)}> */}
        //                 <Tabs id='run-tabs'>
        //                     <Tab eventkey='fit-tab'>
        //                         {Object.keys(fit).map(key => <InputGroup key={key}>
        //                             <InputGroup.Addon>{key}</InputGroup.Addon>
        //                             <input type='text'
        //                                 defaultValue={fit[key]}
        //                                 ref={(input) => { this.fit[key] = input }}
        //                             />
        //                         </InputGroup>)}
        //                     </Tab>
        //                     <Tab eventkey='callbacks-tab'>
        //                         {Object.keys(callbacks).map(key => <InputGroup key={key}>
        //                             <InputGroup.Addon>{key}</InputGroup.Addon>
        //                             <input type='text'
        //                                 defaultValue={fit[key]}
        //                                 ref={(input) => { this.callbacks[key] = input }}
        //                             />
        //                         </InputGroup>)}
        //                     </Tab>
        //                 </Tabs>

        //                 <Button type='submit'>
        //                     Submit
        //                 </Button>
        //             {/* </form> */}
        //         </Modal.Body>
        //     </Modal.Header>
        // </Modal>
        return <Modal show={this.state.runShow} onHide={this.runClose.bind(this)}>
            <Modal.Header closeButton >
            <Modal.Title>Run Settings</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Tabs id='run-tabs'>
                    <Tab eventKey='Callbacks' title='Fit'>
                         {Object.keys(fit).map(key => <InputGroup key={key}>
                            <InputGroup.Addon>{key}</InputGroup.Addon>
                            <input type='text'
                                defaultValue={fit[key]}
                                ref={(input) => { this.fit[key] = input }}
                            />
                        </InputGroup>)} 
                    </Tab>
                    <Tab eventKey='Callbacks' title='Callbacks'>
                         {Object.keys(callbacks).map(key => <InputGroup key={key}>
                            <InputGroup.Addon>{key}</InputGroup.Addon>
                            <input type='text'
                                defaultValue={callbacks[key]}
                                ref={(input) => { this.callbacks[key] = input }}
                            />
                        </InputGroup>)} 
                    </Tab>
                </Tabs>
            </Modal.Body>
            <Modal.Footer>
                <Button type='submit'
                    onClick={event => this.handleSubmit(event, type)}>
                    Submit
                        </Button>
            </Modal.Footer>
        </Modal>

    }
    handleSubmit(event, type) {
        event.preventDefault()
        if (type == 'run') {
            this.runClose()
            let newRun = { fit: {}, callbacks: {} }
            Object.keys(this.props.run.fit).map(key => {
                newRun.fit[key] = this.fit[key]['value']
            })
            Object.keys(this.props.run.callbacks).map(key => {
                newRun.callbacks[key] = this.callbacks[key]['value']
            })
            this.props.onAddRun(newRun)
        } else {
            this.compileClose()
            let newCompile = {}
            Object.keys(this.props.compile).map(key => {
                newCompile[key] = this.compile[key]['value']
            })
            this.props.onAddCompile(newCompile)
        }

    }
    onImportModel() {
        let file = this.file.files[0]
        let reader = new FileReader()
        reader.onload = e => {
            this.props.onImportModel(JSON.parse(e.target.result))
        }
        reader.readAsText(file)
        // for (var i = 0, f; f = files[i]; i++) {
        //     var reader = new FileReader();

        //     // Closure to capture the file information.
        // reader.onload = (function (theFile) {
        //     return function (e) {
        //         let json = JSON.parse(e.target.result)
        //         console.info(json)
        //     }
        // })(f);
        //     reader.readAsText(f);
        // }
    }
    render() {

        return <div id="ToolBox"><Nav 
            bsStyle="tabs"
            className="nav-toolbox" onSelect={this.handleSelect.bind(this)}
            stacked
            className="navbar navbar-inverse navbar-collapse"
            role="navigation"
            >
            <NavItem className="toolBox-heading">
                <Glyphicon glyph="pencil" />
            </NavItem>
            <NavDropdown title='Module' id='module' key="module">
                <NavDropdown title='From keras' id="input">
                    {
                        KERAS_MODELS.map((dataset, i) => <NavItem key={i} eventKey={`dataset_${dataset}`}>{dataset}</NavItem>)
                    }
                </NavDropdown>

                <li>
                    <label className='btn btn-default btn-input'>
                        From local
                    <input
                            id="inputModel"
                            type='file' label='Upload' accept='.json'
                            onChange={this.onImportModel.bind(this)}
                            ref={(ref) => this.file = ref}
                            hidden
                        />
                    </label>
                </li>
            </NavDropdown>
            <NavDropdown title="Datasets" id="Datasets" key="Datasets">
                {/*<NavItem eventKey={'input_local'}>From Local</NavItem>*/}
                <NavDropdown title='From keras' id="Datasets">
                    {
                        KERAS_DATASETS.map((dataset, i) => <NavItem key={i} eventKey={`dataset_${dataset}`}>{dataset}</NavItem>)
                    }
                </NavDropdown>

                <li><label className='btn btn-default btn-input'>
                    From Local
                    <input
                        type='file' label='Upload' accept='.json'
                        ref={(ref) => this.fileUpload = ref}
                        hidden
                    />
                </label></li>
            </NavDropdown>
            <NavDropdown title="Layers" id="layers" key="layers">
                {Object.keys(dictCate).map(cate => <NavDropdown
                    title={cate}
                    id={`layerCate_${cate}`}
                    key={`layerCate_${cate}`}>
                    {dictCate[cate].map((layer, i) => <NavItem
                        key={i} eventKey={`layer_${layer}`}>
                        {layer}
                    </NavItem>)}
                </NavDropdown>)}
                {/*{
                    Object.keys(dictLayer).map((layer, i) => <NavItem key={i} eventKey={`layer_${layer}`}>{layer}</NavItem>)
                }*/}
            </NavDropdown>
            <NavItem eventKey='compile_' id="compile" key="compile">Compile</NavItem>
            <NavItem eventKey='run_' id="run" key="run">Run</NavItem>
        </Nav>
            {this.getCompileModal()}
            {this.getRunModal()}
        </div>

    }
}


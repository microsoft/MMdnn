import { connect } from 'react-redux'
import { 
  addLayer, 
  addKerasDataset, 
  addCompile, 
  addRun, 
  importModel 
 } from '../actions/toolBox_actions'
import ToolBox from '../components/ToolBox';
import {COMPILE, FIT, TF_CB} from '../../assets/def'

const mapStateToProps = (state) => {
  return {
    compile: _.isEmpty(state.setting.compile)?COMPILE:state.setting.compile,
    run: _.isEmpty(state.setting.run)?{fit:FIT, callbacks:TF_CB}:state.setting.run
  }
}

const mapDispatchToProps = (dispatch) => {
  return {
    onAddLayer: (layer) => {
      dispatch(addLayer(layer))
    },
    onAddKerasDataset: (name) => {
      dispatch(addKerasDataset(name))
    },
    onAddCompile: (newCompile) => {
      dispatch(addCompile(newCompile))
    },
    onAddRun: (newRun) => {
      dispatch(addRun(newRun))
    },
     onImportModel:(json) => {
      dispatch(importModel(json))
    }
  }
}

const ToolBoxContainer = connect(
  mapStateToProps,
  mapDispatchToProps
)(ToolBox)

export default ToolBoxContainer
import {setting, dataset, selectedLayer} from "./settingReducer"
import {config} from "./layersReducer"
import { combineReducers } from 'redux'

const codeApp = combineReducers({
  setting, 
  dataset, 
  selectedLayer,
  config
})

export default codeApp